/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file pipeline_planning.cc
 * \brief Plan the software pipeline
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/node/repr_printer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../target/utils.h"
#include "../target/bm1690_lmem.h"
#include "tpu_op_access_model.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Check whether two regions have intersections.
 * \param region1 The first region.
 * \param region2 The second region.
 * \return Whether region1 and region2 have intersections.
 */
bool MayConflict(Region region1, Region region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); i++) {
    Range dim1 = region1[i];
    Range dim2 = region2[i];
    auto int_set1 = arith::IntSet::FromRange(dim1);
    auto int_set2 = arith::IntSet::FromRange(dim2);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

std::string BufferRegionToString(const BufferRegion &region) {
  std::ostringstream os;
  os << region->buffer->name << AsLegacyRepr(region->region);
  return os.str();
}

std::string BufferRegionsToString(const Array<BufferRegion> &regions) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < regions.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << BufferRegionToString(regions[i]);
  }
  os << "]";
  return os.str();
}

std::string ObjectRefToString(const ObjectRef &obj) {
  return AsLegacyRepr(obj);
}

int64_t DivUpInt64(int64_t value, int64_t factor) {
  ICHECK_GT(factor, 0);
  return (value + factor - 1) / factor;
}

int64_t AbsDiffInt64(int64_t lhs, int64_t rhs) {
  return lhs >= rhs ? lhs - rhs : rhs - lhs;
}

bool TryGetPositiveIntImm(const PrimExpr &expr, int64_t *value) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    if (imm->value > 0) {
      *value = imm->value;
      return true;
    }
  }
  return false;
}

bool TryNormalizeLocalShape4(const Array<PrimExpr> &shape,
                             std::vector<int64_t> *shape4) {
  shape4->clear();
  if (shape.empty()) {
    *shape4 = {1, 1, 1, 1};
    return true;
  }

  auto get_dim = [&](size_t idx, int64_t *value) {
    return TryGetPositiveIntImm(shape[idx], value);
  };

  int64_t d0 = 1, d1 = 1, d2 = 1, d3 = 1;
  if (shape.size() == 1) {
    if (!get_dim(0, &d3)) {
      return false;
    }
    *shape4 = {1, 1, 1, d3};
    return true;
  }
  if (shape.size() == 2) {
    if (!get_dim(0, &d1) || !get_dim(1, &d3)) {
      return false;
    }
    *shape4 = {1, d1, 1, d3};
    return true;
  }
  if (shape.size() == 3) {
    if (!get_dim(0, &d0) || !get_dim(1, &d1) || !get_dim(2, &d3)) {
      return false;
    }
    *shape4 = {d0, d1, 1, d3};
    return true;
  }
  if (shape.size() == 4) {
    if (!get_dim(0, &d0) || !get_dim(1, &d1) || !get_dim(2, &d2) ||
        !get_dim(3, &d3)) {
      return false;
    }
    *shape4 = {d0, d1, d2, d3};
    return true;
  }
  return false;
}

bool IsTpuLocalScope(const std::string &scope) {
  return scope == "shared" || scope == "shared.dyn" || scope == "local" ||
         scope == "local.fragment";
}

bool TryEstimateLocalBufferFootprintBytes(const Buffer &buffer,
                                          int64_t *bytes) {
  if (!IsTpuLocalScope(buffer.scope())) {
    return false;
  }
  std::vector<int64_t> shape4;
  if (!TryNormalizeLocalShape4(buffer->shape, &shape4)) {
    return false;
  }
  *bytes =
      bm1690::TpuAlignSizeBytesFromShape4(shape4, buffer->dtype);
  return true;
}

std::string GetPipelineCertificateSink() {
  const char *env = std::getenv("TL_TPU_PIPELINE_CERTIFICATE");
  if (env == nullptr) {
    return "";
  }
  std::string sink(env);
  if (sink.empty() || sink == "0" || sink == "false" || sink == "False") {
    return "";
  }
  return sink;
}

void EmitPipelineCertificate(const std::string &text) {
  std::string sink = GetPipelineCertificateSink();
  if (sink.empty()) {
    return;
  }
  if (sink == "1" || sink == "true" || sink == "True" || sink == "stderr") {
    std::cerr << text;
    return;
  }
  if (sink == "stdout") {
    std::cout << text;
    return;
  }
  std::ofstream file(sink, std::ios::app);
  file << text;
}

bool EnvFlagEnabled(const char *name, bool default_value) {
  const char *env = std::getenv(name);
  if (env == nullptr) {
    return default_value;
  }
  std::string value(env);
  if (value.empty() || value == "0" || value == "false" ||
      value == "False") {
    return false;
  }
  return true;
}

class TpuOpSummaryCollector : public StmtExprVisitor {
public:
  explicit TpuOpSummaryCollector(const Map<Var, Buffer> &buffer_data_to_buffer)
      : buffer_data_to_buffer_(buffer_data_to_buffer) {}

  std::vector<TpuCallExternAccessInfo> Collect(const Stmt &stmt) {
    VisitStmt(stmt);
    return std::move(op_infos_);
  }

private:
  std::string FindScopeFromExpr(const PrimExpr &expr) const {
    if (auto *var = expr.as<VarNode>()) {
      auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
      if (it != buffer_data_to_buffer_.end()) {
        return (*it).second.scope();
      }
      return "";
    }
    if (auto *load = expr.as<BufferLoadNode>()) {
      return load->buffer.scope();
    }
    if (auto *call = expr.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr()) &&
          call->args.size() >= 2) {
        if (auto *var = call->args[1].as<VarNode>()) {
          auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
          if (it != buffer_data_to_buffer_.end()) {
            return (*it).second.scope();
          }
        }
      }
      for (const PrimExpr &arg : call->args) {
        std::string scope = FindScopeFromExpr(arg);
        if (!scope.empty()) {
          return scope;
        }
      }
    }
    return "";
  }

  void VisitExpr_(const CallNode *op) final {
    TpuCallExternAccessInfo info = GetTpuCallExternAccessInfo(op);
    if (info.valid) {
      if (info.op_name == "ppl.copy" && op->args.size() >= 3) {
        std::string src_scope = FindScopeFromExpr(op->args[1]);
        std::string dst_scope = FindScopeFromExpr(op->args[2]);
        info.transaction =
            ClassifyTpuPplCopyTransaction(src_scope, dst_scope);
        PopulateTpuCallExternPhases(&info);
      }
      op_infos_.push_back(std::move(info));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::vector<TpuCallExternAccessInfo> op_infos_;
};

class TpuBufferAccessNameCollector : public StmtExprVisitor {
public:
  explicit TpuBufferAccessNameCollector(
      const Map<Var, Buffer> &buffer_data_to_buffer)
      : buffer_data_to_buffer_(buffer_data_to_buffer) {}

  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  const std::unordered_set<std::string> &reads() const { return reads_; }
  const std::unordered_set<std::string> &writes() const { return writes_; }

private:
  void MarkBuffer(const Buffer &buffer, TpuBufferAccessKind access_kind) {
    std::string name(buffer->name);
    if (access_kind == TpuBufferAccessKind::kRead) {
      reads_.insert(name);
    } else if (access_kind == TpuBufferAccessKind::kWrite) {
      writes_.insert(name);
    } else {
      reads_.insert(name);
      writes_.insert(name);
    }
  }

  void MarkExprAs(const PrimExpr &expr, TpuBufferAccessKind access_kind) {
    bool previous_collecting = collecting_operand_;
    TpuBufferAccessKind previous_access = operand_access_kind_;
    collecting_operand_ = true;
    operand_access_kind_ = access_kind;
    VisitExpr(expr);
    collecting_operand_ = previous_collecting;
    operand_access_kind_ = previous_access;
  }

  bool IsAccessPtrCall(const CallNode *op) const {
    return op->op.same_as(builtin::tvm_access_ptr());
  }

  void VisitExpr_(const VarNode *op) final {
    auto it = buffer_data_to_buffer_.find(GetRef<Var>(op));
    if (it != buffer_data_to_buffer_.end() && collecting_operand_) {
      MarkBuffer((*it).second, operand_access_kind_);
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (collecting_operand_) {
      MarkBuffer(op->buffer, operand_access_kind_);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (collecting_operand_) {
      if (IsAccessPtrCall(op) && op->args.size() >= 2) {
        if (auto *var = op->args[1].as<VarNode>()) {
          auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
          if (it != buffer_data_to_buffer_.end()) {
            MarkBuffer((*it).second, operand_access_kind_);
          }
        }
      }
      StmtExprVisitor::VisitExpr_(op);
      return;
    }

    TpuCallExternAccessInfo info = GetTpuCallExternAccessInfo(op);
    if (info.valid) {
      for (const auto &operand_access : info.operand_accesses) {
        if (operand_access.arg_index < op->args.size()) {
          MarkExprAs(op->args[operand_access.arg_index],
                     operand_access.access_kind);
        }
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_set<std::string> reads_;
  std::unordered_set<std::string> writes_;
  bool collecting_operand_ = false;
  TpuBufferAccessKind operand_access_kind_ = TpuBufferAccessKind::kConservative;
};

class TpuWorkspaceNameCollector : public StmtExprVisitor {
public:
  explicit TpuWorkspaceNameCollector(
      const Map<Var, Buffer> &buffer_data_to_buffer)
      : buffer_data_to_buffer_(buffer_data_to_buffer) {}

  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  const std::unordered_set<std::string> &workspaces() const {
    return workspaces_;
  }

  const std::unordered_map<std::string, int64_t> &workspace_footprints() const {
    return workspace_footprints_;
  }

private:
  void MarkBuffer(const Buffer &buffer) {
    if (IsTpuLocalScope(buffer.scope())) {
      std::string name(buffer->name);
      workspaces_.insert(name);
      int64_t bytes = 0;
      if (TryEstimateLocalBufferFootprintBytes(buffer, &bytes)) {
        workspace_footprints_[name] = bytes;
      }
    }
  }

  void MarkVar(const Var &var) {
    std::string name(var->name_hint);
    auto alias_it = var_aliases_.find(name);
    if (alias_it != var_aliases_.end() && !resolving_alias_) {
      resolving_alias_ = true;
      MarkExpr(alias_it->second);
      resolving_alias_ = false;
      return;
    }
    workspaces_.insert(name);
    auto it = allocation_footprints_.find(name);
    if (it != allocation_footprints_.end()) {
      workspace_footprints_[name] = it->second;
    }
  }

  void MarkExpr(const PrimExpr &expr) {
    bool previous_collecting = collecting_workspace_;
    collecting_workspace_ = true;
    VisitExpr(expr);
    collecting_workspace_ = previous_collecting;
  }

  bool IsAccessPtrCall(const CallNode *op) const {
    return op->op.same_as(builtin::tvm_access_ptr());
  }

  void VisitExpr_(const VarNode *op) final {
    if (collecting_workspace_) {
      auto it = buffer_data_to_buffer_.find(GetRef<Var>(op));
      if (it != buffer_data_to_buffer_.end()) {
        MarkBuffer((*it).second);
        return;
      }
      MarkVar(GetRef<Var>(op));
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (collecting_workspace_) {
      MarkBuffer(op->buffer);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (collecting_workspace_) {
      if (IsAccessPtrCall(op) && op->args.size() >= 2) {
        if (auto *var = op->args[1].as<VarNode>()) {
          auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
          if (it != buffer_data_to_buffer_.end()) {
            MarkBuffer((*it).second);
          } else {
            MarkVar(GetRef<Var>(var));
          }
        }
      }
      StmtExprVisitor::VisitExpr_(op);
      return;
    }

    TpuCallExternAccessInfo info = GetTpuCallExternAccessInfo(op);
    if (info.valid) {
      for (const auto &phase : info.phases) {
        if (!phase.hidden_workspace) {
          continue;
        }
        for (size_t arg_index : phase.scratch_args) {
          if (arg_index < op->args.size()) {
            MarkExpr(op->args[arg_index]);
          }
        }
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode *op) final {
    var_aliases_[op->var->name_hint] = op->value;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode *op) final {
    int64_t elements = 1;
    bool static_shape = true;
    std::vector<int64_t> shape;
    shape.reserve(op->extents.size());
    for (const PrimExpr &extent_expr : op->extents) {
      int64_t extent = 0;
      if (!TryGetPositiveIntImm(extent_expr, &extent)) {
        static_shape = false;
        break;
      }
      shape.push_back(extent);
      elements *= extent;
    }
    if (static_shape && elements > 0) {
      Array<PrimExpr> array_shape;
      for (int64_t extent : shape) {
        array_shape.push_back(IntImm(DataType::Int(32), extent));
      }
      std::vector<int64_t> shape4;
      if (TryNormalizeLocalShape4(array_shape, &shape4)) {
        allocation_footprints_[op->buffer_var->name_hint] =
            bm1690::TpuAlignSizeBytesFromShape4(shape4, op->dtype);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_set<std::string> workspaces_;
  std::unordered_map<std::string, int64_t> workspace_footprints_;
  std::unordered_map<std::string, int64_t> allocation_footprints_;
  std::unordered_map<std::string, PrimExpr> var_aliases_;
  bool collecting_workspace_ = false;
  bool resolving_alias_ = false;
};

struct TpuRegionResourceEstimate {
  bool valid = false;
  bool static_bytes_known = false;
  int64_t logical_bytes = 0;
  std::string buffer_name;
  std::string scope;
  DataType dtype;
};

struct TpuStageResourceEstimate {
  bool unknown_static_cost = false;
  bool phase_unknown_static_cost = false;
  int64_t s2l_bytes = 0;
  int64_t l2s_bytes = 0;
  int64_t local_copy_bytes = 0;
  int64_t bdc_ops = 0;
  int64_t phase_gdma_cost_units = 0;
  int64_t phase_bdc_cost_units = 0;
  int64_t phase_local_copy_cost_units = 0;
  int64_t local_footprint_bytes = 0;
};

class TpuStageResourceEstimateCollector : public StmtExprVisitor {
public:
  explicit TpuStageResourceEstimateCollector(
      const Map<Var, Buffer> &buffer_data_to_buffer)
      : buffer_data_to_buffer_(buffer_data_to_buffer) {}

  TpuStageResourceEstimate Collect(const Stmt &stmt) {
    VisitStmt(stmt);
    return estimate_;
  }

private:
  std::string FindScopeFromExpr(const PrimExpr &expr) const {
    if (auto *var = expr.as<VarNode>()) {
      auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
      if (it != buffer_data_to_buffer_.end()) {
        return (*it).second.scope();
      }
      return "";
    }
    if (auto *load = expr.as<BufferLoadNode>()) {
      return load->buffer.scope();
    }
    if (auto *call = expr.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr()) &&
          call->args.size() >= 2) {
        if (auto *var = call->args[1].as<VarNode>()) {
          auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
          if (it != buffer_data_to_buffer_.end()) {
            return (*it).second.scope();
          }
        }
      }
      if (call->op.same_as(Op::Get("tl.region")) && !call->args.empty()) {
        if (auto *load = call->args[0].as<BufferLoadNode>()) {
          return load->buffer.scope();
        }
      }
      for (const PrimExpr &arg : call->args) {
        std::string scope = FindScopeFromExpr(arg);
        if (!scope.empty()) {
          return scope;
        }
      }
    }
    return "";
  }

  TpuRegionResourceEstimate EstimateRegionExpr(const PrimExpr &expr) const {
    TpuRegionResourceEstimate result;
    const auto *call = expr.as<CallNode>();
    if (!call || !call->op.same_as(Op::Get("tl.region")) ||
        call->args.size() < 2) {
      return result;
    }
    const auto *load = call->args[0].as<BufferLoadNode>();
    if (!load) {
      return result;
    }
    result.valid = true;
    result.buffer_name = std::string(load->buffer->name);
    result.scope = load->buffer.scope();
    result.dtype = load->buffer->dtype;

    int64_t elements = 1;
    for (size_t i = 2; i < call->args.size(); ++i) {
      int64_t extent = 0;
      if (!TryGetPositiveIntImm(call->args[i], &extent)) {
        result.static_bytes_known = false;
        return result;
      }
      elements *= extent;
    }
    result.logical_bytes = elements * bm1690::DTypeBytes(result.dtype);
    result.static_bytes_known = true;
    return result;
  }

  bool TryEstimateBufferLogicalElements(const Buffer &buffer,
                                        int64_t *elements) const {
    int64_t result = 1;
    for (const PrimExpr &dim : buffer->shape) {
      int64_t extent = 0;
      if (!TryGetPositiveIntImm(dim, &extent)) {
        return false;
      }
      result *= extent;
    }
    *elements = result;
    return true;
  }

  bool TryEstimateAccessPtrLogicalElements(const PrimExpr &expr,
                                           int64_t *elements) const {
    if (const auto *call = expr.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr()) &&
          call->args.size() >= 2) {
        if (const auto *var = call->args[1].as<VarNode>()) {
          auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
          if (it != buffer_data_to_buffer_.end()) {
            return TryEstimateBufferLogicalElements((*it).second, elements);
          }
        }
      }
    }
    if (const auto *var = expr.as<VarNode>()) {
      auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
      if (it != buffer_data_to_buffer_.end()) {
        return TryEstimateBufferLogicalElements((*it).second, elements);
      }
    }
    return false;
  }

  void AccumulateCopyBytes(const TpuCallExternAccessInfo &info,
                           const CallNode *op) {
    if (op->args.size() < 3) {
      estimate_.unknown_static_cost = true;
      return;
    }
    TpuRegionResourceEstimate src = EstimateRegionExpr(op->args[1]);
    TpuRegionResourceEstimate dst = EstimateRegionExpr(op->args[2]);
    int64_t copy_bytes = 0;
    if (src.static_bytes_known && dst.static_bytes_known) {
      copy_bytes = std::max(src.logical_bytes, dst.logical_bytes);
    } else if (src.static_bytes_known) {
      copy_bytes = src.logical_bytes;
      estimate_.unknown_static_cost = true;
    } else if (dst.static_bytes_known) {
      copy_bytes = dst.logical_bytes;
      estimate_.unknown_static_cost = true;
    } else {
      estimate_.unknown_static_cost = true;
      return;
    }

    std::string transaction = info.transaction;
    if (transaction.empty()) {
      std::string src_scope =
          src.valid ? src.scope : FindScopeFromExpr(op->args[1]);
      std::string dst_scope =
          dst.valid ? dst.scope : FindScopeFromExpr(op->args[2]);
      transaction = ClassifyTpuPplCopyTransaction(src_scope, dst_scope);
    }
    if (transaction == "S2L") {
      estimate_.s2l_bytes += copy_bytes;
    } else if (transaction == "L2S") {
      estimate_.l2s_bytes += copy_bytes;
    } else if (transaction == "local_to_local") {
      estimate_.local_copy_bytes += copy_bytes;
    } else {
      estimate_.unknown_static_cost = true;
    }
  }

  bool TryEstimateCopyBytesAndTransaction(const TpuCallExternAccessInfo &info,
                                          const CallNode *op,
                                          int64_t *copy_bytes,
                                          std::string *transaction) const {
    if (op->args.size() < 3) {
      return false;
    }
    TpuRegionResourceEstimate src = EstimateRegionExpr(op->args[1]);
    TpuRegionResourceEstimate dst = EstimateRegionExpr(op->args[2]);
    if (src.static_bytes_known && dst.static_bytes_known) {
      *copy_bytes = std::max(src.logical_bytes, dst.logical_bytes);
    } else if (src.static_bytes_known) {
      *copy_bytes = src.logical_bytes;
    } else if (dst.static_bytes_known) {
      *copy_bytes = dst.logical_bytes;
    } else {
      return false;
    }

    *transaction = info.transaction;
    if (transaction->empty()) {
      std::string src_scope =
          src.valid ? src.scope : FindScopeFromExpr(op->args[1]);
      std::string dst_scope =
          dst.valid ? dst.scope : FindScopeFromExpr(op->args[2]);
      *transaction = ClassifyTpuPplCopyTransaction(src_scope, dst_scope);
    }
    return true;
  }

  int64_t CostUnitsFromOps(int64_t ops) const {
    return DivUpInt64(std::max<int64_t>(0, ops), 1024);
  }

  int64_t CostUnitsFromBytes(int64_t bytes) const {
    return DivUpInt64(std::max<int64_t>(0, bytes), bm1690::kEuBytes);
  }

  void AccumulatePhaseCostUnits(const TpuCallExternAccessInfo &info,
                                const CallNode *op) {
    if (!info.valid || info.phases.empty()) {
      return;
    }

    if (info.op_name == "ppl.copy") {
      int64_t copy_bytes = 0;
      std::string transaction;
      if (!TryEstimateCopyBytesAndTransaction(info, op, &copy_bytes,
                                              &transaction)) {
        estimate_.phase_unknown_static_cost = true;
        return;
      }
      int64_t units = CostUnitsFromBytes(copy_bytes);
      if (transaction == "S2L" || transaction == "L2S") {
        estimate_.phase_gdma_cost_units += units;
      } else if (transaction == "local_to_local") {
        estimate_.phase_local_copy_cost_units += units;
      } else {
        estimate_.phase_unknown_static_cost = true;
      }
      return;
    }

    if (info.op_name == "ppl.gemm") {
      if (op->args.size() < 9) {
        estimate_.phase_unknown_static_cost = true;
        return;
      }
      int64_t m = 0, n = 0, k = 0;
      if (!TryGetPositiveIntImm(op->args[6], &m) ||
          !TryGetPositiveIntImm(op->args[7], &n) ||
          !TryGetPositiveIntImm(op->args[8], &k)) {
        estimate_.phase_unknown_static_cost = true;
        return;
      }
      estimate_.phase_bdc_cost_units += CostUnitsFromOps(2 * m * n * k);
      return;
    }

    if (info.op_name == "ppl.fill") {
      int64_t elements = 0;
      if (!TryEstimateOperandElements(op, 1, &elements)) {
        estimate_.phase_unknown_static_cost = true;
        return;
      }
      estimate_.phase_bdc_cost_units += CostUnitsFromOps(elements);
      return;
    }

    if (IsTpuElementwisePplOp(info.op_name) ||
        IsTpuElementwiseConstPplOp(info.op_name)) {
      int64_t elements = 0;
      if (!TryEstimateOperandElements(op, 1, &elements)) {
        estimate_.phase_unknown_static_cost = true;
        return;
      }
      estimate_.phase_bdc_cost_units += CostUnitsFromOps(elements);
      return;
    }

    if (IsTpuReducePplOp(info.op_name)) {
      int64_t input_elements = 0;
      int64_t output_elements = 0;
      bool input_known = TryEstimateOperandElements(op, 1, &input_elements);
      bool output_known = TryEstimateOperandElements(op, 2, &output_elements);
      if (input_known) {
        estimate_.phase_bdc_cost_units += CostUnitsFromOps(input_elements);
      }
      if (output_known) {
        estimate_.phase_bdc_cost_units += CostUnitsFromOps(output_elements);
      }
      if (!input_known || !output_known) {
        estimate_.phase_unknown_static_cost = true;
      }
      return;
    }

    if (info.op_name == "ppl.exp" || info.op_name == "ppl.sigmoid") {
      int64_t elements = 0;
      if (TryEstimateOperandElements(op, 1, &elements)) {
        // Transcendental PPL ops are BDC-heavy and table/workspace-backed.
        // Keep the cost conservative but nonzero so attention softmax is not
        // scored as a generic unknown with no overlap weight.
        estimate_.phase_bdc_cost_units += CostUnitsFromOps(elements * 4);
      } else {
        estimate_.phase_unknown_static_cost = true;
      }
      estimate_.phase_unknown_static_cost = true;
      return;
    }

    bool saw_known_engine = false;
    for (const TpuOpPhaseInfo &phase : info.phases) {
      if (phase.engine == "bdc") {
        saw_known_engine = true;
        int64_t elements = 0;
        if (!phase.write_args.empty() &&
            TryEstimateOperandElements(op, phase.write_args.front(),
                                       &elements)) {
          estimate_.phase_bdc_cost_units += CostUnitsFromOps(elements);
        } else {
          estimate_.phase_unknown_static_cost = true;
        }
      } else if (phase.engine == "gdma") {
        saw_known_engine = true;
        estimate_.phase_unknown_static_cost = true;
      }
      if (phase.conservative) {
        estimate_.phase_unknown_static_cost = true;
      }
    }
    if (!saw_known_engine) {
      estimate_.phase_unknown_static_cost = true;
    }
  }

  void AccumulateGemmOps(const CallNode *op) {
    if (op->args.size() < 9) {
      estimate_.unknown_static_cost = true;
      return;
    }
    int64_t m = 0, n = 0, k = 0;
    if (!TryGetPositiveIntImm(op->args[6], &m) ||
        !TryGetPositiveIntImm(op->args[7], &n) ||
        !TryGetPositiveIntImm(op->args[8], &k)) {
      estimate_.unknown_static_cost = true;
      return;
    }
    estimate_.bdc_ops += 2 * m * n * k;
  }

  void AccumulateFillOps(const CallNode *op) {
    if (op->args.size() < 2) {
      estimate_.unknown_static_cost = true;
      return;
    }
    TpuRegionResourceEstimate dst = EstimateRegionExpr(op->args[1]);
    if (dst.static_bytes_known) {
      int64_t dtype_bytes = bm1690::DTypeBytes(dst.dtype);
      if (dtype_bytes <= 0) {
        estimate_.unknown_static_cost = true;
        return;
      }
      estimate_.bdc_ops += dst.logical_bytes / dtype_bytes;
      return;
    }
    int64_t elements = 0;
    if (!TryEstimateAccessPtrLogicalElements(op->args[1], &elements)) {
      estimate_.unknown_static_cost = true;
      return;
    }
    estimate_.bdc_ops += elements;
  }

  bool TryEstimateOperandElements(const CallNode *op, size_t arg_index,
                                  int64_t *elements) const {
    if (arg_index >= op->args.size()) {
      return false;
    }
    TpuRegionResourceEstimate region = EstimateRegionExpr(op->args[arg_index]);
    if (region.static_bytes_known) {
      int64_t dtype_bytes = bm1690::DTypeBytes(region.dtype);
      if (dtype_bytes <= 0) {
        return false;
      }
      *elements = region.logical_bytes / dtype_bytes;
      return true;
    }
    return TryEstimateAccessPtrLogicalElements(op->args[arg_index], elements);
  }

  void AccumulateElementwiseOps(const CallNode *op) {
    int64_t elements = 0;
    if (!TryEstimateOperandElements(op, 1, &elements)) {
      estimate_.unknown_static_cost = true;
      return;
    }
    estimate_.bdc_ops += elements;
  }

  void AccumulateReduceOps(const CallNode *op) {
    int64_t input_elements = 0;
    int64_t output_elements = 0;
    bool input_known = TryEstimateOperandElements(op, 1, &input_elements);
    bool output_known = TryEstimateOperandElements(op, 2, &output_elements);
    if (input_known) {
      estimate_.bdc_ops += input_elements;
    }
    if (output_known) {
      estimate_.bdc_ops += output_elements;
    }
    if (!input_known || !output_known) {
      estimate_.unknown_static_cost = true;
    }
  }

  void VisitExpr_(const CallNode *op) final {
    TpuCallExternAccessInfo info = GetTpuCallExternAccessInfo(op);
    if (info.valid) {
      AccumulatePhaseCostUnits(info, op);
      if (info.op_name == "ppl.copy") {
        AccumulateCopyBytes(info, op);
      } else if (info.op_name == "ppl.gemm") {
        AccumulateGemmOps(op);
      } else if (info.op_name == "ppl.fill") {
        AccumulateFillOps(op);
      } else if (IsTpuElementwisePplOp(info.op_name) ||
                 IsTpuElementwiseConstPplOp(info.op_name)) {
        AccumulateElementwiseOps(op);
      } else if (IsTpuReducePplOp(info.op_name)) {
        AccumulateReduceOps(op);
      } else if (info.engine == "bdc") {
        estimate_.unknown_static_cost = true;
      }
      estimate_.unknown_static_cost =
          estimate_.unknown_static_cost || estimate_.phase_unknown_static_cost;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  TpuStageResourceEstimate estimate_;
};

std::string TpuOpInfosToString(
    const std::vector<TpuCallExternAccessInfo> &op_infos) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < op_infos.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << TpuCallExternAccessInfoToString(op_infos[i]);
  }
  os << "]";
  return os.str();
}

std::string TpuStageResourceEstimateToString(
    const TpuStageResourceEstimate &estimate) {
  std::ostringstream os;
  os << "{s2l_bytes=" << estimate.s2l_bytes
     << ",l2s_bytes=" << estimate.l2s_bytes
     << ",local_copy_bytes=" << estimate.local_copy_bytes
     << ",bdc_ops=" << estimate.bdc_ops
     << ",phase_gdma_cost_units=" << estimate.phase_gdma_cost_units
     << ",phase_bdc_cost_units=" << estimate.phase_bdc_cost_units
     << ",phase_local_copy_cost_units="
     << estimate.phase_local_copy_cost_units
     << ",local_footprint_bytes=" << estimate.local_footprint_bytes
     << ",phase_unknown_static_cost="
     << (estimate.phase_unknown_static_cost ? "true" : "false")
     << ",unknown_static_cost="
     << (estimate.unknown_static_cost ? "true" : "false") << "}";
  return os.str();
}

std::string StringSetToString(const std::unordered_set<std::string> &values) {
  std::vector<std::string> ordered(values.begin(), values.end());
  std::sort(ordered.begin(), ordered.end());
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < ordered.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << ordered[i];
  }
  os << "]";
  return os.str();
}

std::string IntVectorToString(const std::vector<int> &values) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << values[i];
  }
  os << "]";
  return os.str();
}

std::string StringVectorToString(const std::vector<std::string> &values) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << values[i];
  }
  os << "]";
  return os.str();
}

std::vector<std::string> SortedIntersection(
    const std::unordered_set<std::string> &lhs,
    const std::unordered_set<std::string> &rhs) {
  std::vector<std::string> result;
  for (const auto &value : lhs) {
    if (rhs.count(value)) {
      result.push_back(value);
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

class PipelinePlanner : public StmtExprMutator {
public:
  static Stmt Substitute(const PrimFunc &f) {
    PipelinePlanner substituter;
    for (const auto &[_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "Pipeline_Planning: Require the target attribute";
    substituter.target_ = target.value();
    if (auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
      substituter.global_symbol_ = global_symbol.value();
    }
    return substituter.VisitStmt(f->body);
  }

private:
  PipelinePlanner() = default;

  struct PipelineStageInfo {
    Array<BufferRegion> reads, writes;
    int original_order;
    int order = -1, stage = -1;
    bool copy_stage = false;
    int last_use_stage = -1;
    std::string tpu_pipeline_role = "unknown";
    std::vector<TpuCallExternAccessInfo> tpu_ops;
    std::unordered_set<std::string> tpu_read_buffers;
    std::unordered_set<std::string> tpu_write_buffers;
    std::unordered_set<std::string> tpu_workspace_buffers;
    std::unordered_map<std::string, int64_t> tpu_workspace_footprints;
    TpuStageResourceEstimate tpu_resource;
  };

  struct PipelineDependency {
    int src = -1;
    int dst = -1;
    std::string kind;
    std::string buffer;
  };

  struct TpuPipelinePolicyDecision {
    std::string kind = "non_tpu";
    std::string reason = "not_tpu_target";
    std::string policy_family = "non_tpu";
    std::string evidence_tier = "tier_0";
    std::string known_unknowns = "";
    bool allow_stage_optimization = false;
    bool enable_buffer_version_hints = false;
    bool enable_stage_aware_lmem = false;
    bool phase_model_available = false;
  };

  struct TpuOverlapWindowInfo {
    std::string name;
    std::string kind;
    std::string status;
    std::string materialization_kind;
    std::string blocked_reason;
    std::string notes;
    int load_statement = -1;
    int consumer_statement = -1;
    std::vector<int> overlap_statements;
    std::vector<std::pair<std::string, int>> required_versions;
  };

  std::vector<int> GetManualIntArrayAnnotation(const ForNode *loop,
                                               const char *key,
                                               size_t expected_size) {
    std::vector<int> result;
    auto anno = loop->annotations.Get(key);
    if (!anno.defined()) {
      return result;
    }
    Array<Integer> values = Downcast<Array<Integer>>(anno);
    CHECK_EQ(values.size(), expected_size)
        << "Pipeline annotation " << key << " expects " << expected_size
        << " entries, got " << values.size();
    result.reserve(values.size());
    for (const Integer &value : values) {
      result.push_back(static_cast<int>(value->value));
    }
    return result;
  }

  void ApplyManualPipelineAnnotations(
      const ForNode *loop, std::vector<PipelineStageInfo> *stage_infos) {
    std::vector<int> manual_stages = GetManualIntArrayAnnotation(
        loop, "tl_pipeline_stage", stage_infos->size());
    std::vector<int> manual_orders = GetManualIntArrayAnnotation(
        loop, "tl_pipeline_order", stage_infos->size());

    if (!manual_stages.empty()) {
      for (size_t i = 0; i < stage_infos->size(); ++i) {
        CHECK_GE(manual_stages[i], 0)
            << "tl_pipeline_stage must be non-negative, got "
            << manual_stages[i] << " at index " << i;
        (*stage_infos)[i].stage = manual_stages[i];
      }
    }

    if (!manual_orders.empty()) {
      std::vector<bool> seen(stage_infos->size(), false);
      for (size_t i = 0; i < stage_infos->size(); ++i) {
        int order = manual_orders[i];
        CHECK_GE(order, 0)
            << "tl_pipeline_order must be non-negative, got " << order
            << " at index " << i;
        CHECK_LT(order, static_cast<int>(stage_infos->size()))
            << "tl_pipeline_order must be smaller than statement count, got "
            << order << " at index " << i;
        CHECK(!seen[order])
            << "tl_pipeline_order contains duplicate order " << order;
        seen[order] = true;
        (*stage_infos)[i].order = order;
      }
    }
  }

  bool HasManualStageOrOrder(const ForNode *loop) const {
    return loop->annotations.Get("tl_pipeline_stage").defined() ||
           loop->annotations.Get("tl_pipeline_order").defined();
  }

  bool IsTpuTarget() const {
    return target_.defined() && target_->kind.defined() &&
           target_->kind->name == "tpu";
  }

  bool IsSimpleGemmCompanionOp(const std::string &op_name) const {
    return op_name == "ppl.copy" || op_name == "ppl.fill" ||
           op_name == "ppl.gemm";
  }

  TpuPipelinePolicyDecision DecideTpuPipelinePolicy(
      const std::vector<PipelineStageInfo> &stage_infos,
      int num_stages) const {
    TpuPipelinePolicyDecision decision;
    if (!IsTpuTarget()) {
      return decision;
    }

    bool has_gemm = false;
    bool has_mixed_compute = false;
    bool has_reduce = false;
    bool has_exp_or_sigmoid = false;
    bool has_load = false;
    bool has_compute = false;
    bool has_unknown_tpu_role = false;
    for (const auto &pinfo : stage_infos) {
      has_load = has_load || pinfo.tpu_pipeline_role == "load";
      has_compute = has_compute || pinfo.tpu_pipeline_role == "compute" ||
                    pinfo.tpu_pipeline_role == "local_copy";
      has_unknown_tpu_role =
          has_unknown_tpu_role ||
          (!pinfo.tpu_ops.empty() &&
           (pinfo.tpu_pipeline_role == "unknown" ||
            pinfo.tpu_pipeline_role == "mixed_or_unknown"));
      for (const auto &op_info : pinfo.tpu_ops) {
        if (!op_info.valid || op_info.op_name.rfind("ppl.", 0) != 0) {
          continue;
        }
        if (op_info.op_name == "ppl.gemm") {
          has_gemm = true;
        }
        if (IsTpuReducePplOp(op_info.op_name)) {
          has_reduce = true;
        }
        if (op_info.op_name == "ppl.exp" || op_info.op_name == "ppl.sigmoid") {
          has_exp_or_sigmoid = true;
        }
        if (!IsSimpleGemmCompanionOp(op_info.op_name)) {
          has_mixed_compute = true;
        }
      }
    }

    if (has_unknown_tpu_role) {
      decision.kind = "unsupported_tpu_role";
      decision.policy_family = "generic";
      decision.evidence_tier = "tier_0_contract";
      decision.reason =
          "pipeline_contains_unknown_or_mixed_tpu_statement_role";
      decision.known_unknowns =
          "planner_keeps_legacy_pipeline_order_without_tpu_specific_optimization";
      decision.allow_stage_optimization = false;
      decision.enable_buffer_version_hints = false;
      decision.enable_stage_aware_lmem = false;
      decision.phase_model_available = false;
      return decision;
    }

    if (has_gemm && !has_mixed_compute && has_load && has_compute &&
        num_stages >= 2) {
      decision.kind = "gemm_heavy";
      decision.policy_family = "pure_gemm";
      decision.evidence_tier = "tier_2_model_local";
      decision.phase_model_available = true;
      decision.reason =
          "pure_gemm_pipeline_with_load_compute_pattern";
      decision.allow_stage_optimization = true;
      decision.enable_buffer_version_hints =
          EnvFlagEnabled("TL_TPU_BUFFER_VERSION_HINTS", true);
      decision.enable_stage_aware_lmem = true;
      return decision;
    }

    if (has_gemm && has_mixed_compute) {
      decision.kind = "mixed_gemm";
      decision.policy_family = "attention_mixed";
      decision.evidence_tier =
          (has_reduce || has_exp_or_sigmoid) ? "tier_1_5_phase_model"
                                             : "tier_0_contract";
      decision.phase_model_available = true;
      decision.known_unknowns =
          "mixed_gdma_bdc_windows_need_family_specific_materialization";
      decision.reason =
          "gemm_pipeline_with_elementwise_or_reduction_ops";
      decision.allow_stage_optimization = false;
      decision.enable_buffer_version_hints = false;
      decision.enable_stage_aware_lmem = false;
      return decision;
    }

    if (!has_gemm && has_reduce && has_load && has_compute &&
        num_stages >= 2) {
      decision.kind = "reduction_norm";
      decision.policy_family = "reduction_norm";
      decision.evidence_tier = "tier_2_family_model_local";
      decision.phase_model_available = true;
      decision.known_unknowns =
          "reduction_runtime_overlap_and_bank_effect_need_cmodel_or_board_validation";
      decision.reason =
          "reduction_norm_pipeline_with_load_compute_pattern";
      decision.allow_stage_optimization = true;
      decision.enable_buffer_version_hints =
          EnvFlagEnabled("TL_TPU_BUFFER_VERSION_HINTS", true);
      decision.enable_stage_aware_lmem = false;
      return decision;
    }

    if (has_reduce) {
      decision.kind = "elementwise_or_generic";
      decision.policy_family = "reduction_norm";
      decision.evidence_tier = "tier_1_5_phase_model";
      decision.phase_model_available = true;
      decision.known_unknowns =
          "reduction_workspace_and_loop_carried_state_need_family_policy";
      decision.reason = "reduction_pipeline_without_supported_lc_pattern";
      decision.allow_stage_optimization = false;
      decision.enable_buffer_version_hints = false;
      decision.enable_stage_aware_lmem = false;
      return decision;
    }

    decision.kind = "elementwise_or_generic";
    decision.policy_family = "generic";
    decision.evidence_tier = has_exp_or_sigmoid ? "tier_1_5_phase_model"
                                                : "tier_0_contract";
    decision.phase_model_available = has_exp_or_sigmoid;
    decision.reason = "no_gemm_heavy_pipeline_pattern";
    decision.allow_stage_optimization = false;
    decision.enable_buffer_version_hints = false;
    decision.enable_stage_aware_lmem = false;
    return decision;
  }

  std::string ClassifyTpuPipelineRole(PipelineStageInfo *pinfo) const {
    bool saw_tpu_op = false;
    bool saw_compute = false;
    bool saw_local_copy = false;
    bool saw_load = false;
    bool saw_store = false;
    bool saw_unknown = false;
    for (const auto &op_info : pinfo->tpu_ops) {
      if (!op_info.valid) {
        continue;
      }
      saw_tpu_op = true;
      if (op_info.op_name == "ppl.copy") {
        if (op_info.transaction == "S2L") {
          saw_load = true;
        } else if (op_info.transaction == "L2S") {
          saw_store = true;
        } else if (op_info.transaction == "local_to_local") {
          saw_local_copy = true;
        } else {
          saw_unknown = true;
        }
      } else if (op_info.engine == "bdc") {
        saw_compute = true;
      } else {
        saw_unknown = true;
      }
    }
    if (!saw_tpu_op) {
      return "unknown";
    }
    int role_count = 0;
    role_count += saw_load ? 1 : 0;
    role_count += saw_store ? 1 : 0;
    role_count += saw_compute ? 1 : 0;
    role_count += saw_local_copy ? 1 : 0;
    if (saw_unknown || role_count != 1) {
      return "mixed_or_unknown";
    }
    if (saw_load) {
      return "load";
    }
    if (saw_store) {
      return "store";
    }
    if (saw_local_copy) {
      return "local_copy";
    }
    return "compute";
  }

  bool TryApplyTpuStageOptimizerV0(
      int num_stages, std::vector<PipelineStageInfo> *stage_infos,
      std::string *optimizer_decision) const {
    if (!IsTpuTarget() || num_stages < 2) {
      return false;
    }

    bool has_load = false;
    bool has_compute = false;
    for (auto &pinfo : *stage_infos) {
      pinfo.tpu_pipeline_role = ClassifyTpuPipelineRole(&pinfo);
      if (pinfo.tpu_pipeline_role == "unknown" ||
          pinfo.tpu_pipeline_role == "mixed_or_unknown") {
        *optimizer_decision = "legacy_heuristic:unsupported_tpu_role";
        return false;
      }
      has_load = has_load || pinfo.tpu_pipeline_role == "load";
      has_compute = has_compute || pinfo.tpu_pipeline_role == "compute" ||
                    pinfo.tpu_pipeline_role == "local_copy";
    }
    if (!has_load || !has_compute) {
      *optimizer_decision = "legacy_heuristic:missing_lc_pattern";
      return false;
    }

    int compute_stage = std::min(1, num_stages - 1);
    int store_stage = std::min(2, num_stages - 1);
    for (auto &pinfo : *stage_infos) {
      if (pinfo.tpu_pipeline_role == "load") {
        pinfo.stage = 0;
      } else if (pinfo.tpu_pipeline_role == "store") {
        pinfo.stage = store_stage;
      } else {
        pinfo.stage = compute_stage;
      }
    }

    std::vector<size_t> indices(stage_infos->size());
    for (size_t i = 0; i < indices.size(); ++i) {
      indices[i] = i;
    }
    std::stable_sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
      const auto &l = (*stage_infos)[lhs];
      const auto &r = (*stage_infos)[rhs];
      if (l.stage != r.stage) {
        return l.stage < r.stage;
      }
      return l.original_order < r.original_order;
    });
    for (size_t order = 0; order < indices.size(); ++order) {
      (*stage_infos)[indices[order]].order = static_cast<int>(order);
    }
    *optimizer_decision = "tpu_stage_optimizer_v0";
    return true;
  }

  std::vector<int> AllowedTpuStagesForRole(const std::string &role,
                                           int num_stages) const {
    int max_stage = std::max(0, num_stages - 1);
    if (role == "load") {
      return {0};
    }
    int compute_stage = std::min(1, max_stage);
    int store_stage = std::min(2, max_stage);
    std::vector<int> stages;
    int start = role == "store" ? store_stage : compute_stage;
    for (int stage = start; stage <= max_stage; ++stage) {
      stages.push_back(stage);
    }
    if (stages.empty()) {
      stages.push_back(max_stage);
    }
    return stages;
  }

  struct TpuStageSearchScore {
    bool valid = false;
    int overlap_score = 0;
    int64_t s2l_bytes = 0;
    int64_t l2s_bytes = 0;
    int64_t local_copy_bytes = 0;
    int64_t bdc_ops = 0;
    int64_t dma_cost_units = 0;
    int64_t bdc_cost_units = 0;
    int64_t legacy_dma_cost_units = 0;
    int64_t legacy_bdc_cost_units = 0;
    int64_t legacy_pipeline_balance_penalty = 0;
    int64_t pipeline_balance_penalty = 0;
    int max_raw_stage_distance = 0;
    int expanded_prefetch_edges = 0;
    int prefetch_distance_penalty = 0;
    int64_t versioned_lmem_bytes = 0;
    int64_t lmem_bank_pressure_units = 0;
    int max_required_versions = 0;
    int version_sum = 0;
    int preferred_penalty = 0;
    int max_stage = 0;
    bool unknown_static_cost = false;
    std::vector<int> stages;
  };

  bool IsBetterTpuStageCandidate(const TpuStageSearchScore &candidate,
                                 const TpuStageSearchScore &best) const {
    if (!best.valid) {
      return true;
    }
    if (candidate.overlap_score != best.overlap_score) {
      return candidate.overlap_score > best.overlap_score;
    }
    if (candidate.pipeline_balance_penalty !=
        best.pipeline_balance_penalty) {
      return candidate.pipeline_balance_penalty <
             best.pipeline_balance_penalty;
    }
    if (candidate.prefetch_distance_penalty !=
        best.prefetch_distance_penalty) {
      return candidate.prefetch_distance_penalty <
             best.prefetch_distance_penalty;
    }
    if (candidate.expanded_prefetch_edges != best.expanded_prefetch_edges) {
      return candidate.expanded_prefetch_edges <
             best.expanded_prefetch_edges;
    }
    if (candidate.lmem_bank_pressure_units !=
        best.lmem_bank_pressure_units) {
      return candidate.lmem_bank_pressure_units <
             best.lmem_bank_pressure_units;
    }
    if (candidate.versioned_lmem_bytes != best.versioned_lmem_bytes) {
      return candidate.versioned_lmem_bytes < best.versioned_lmem_bytes;
    }
    if (candidate.max_required_versions != best.max_required_versions) {
      return candidate.max_required_versions < best.max_required_versions;
    }
    if (candidate.version_sum != best.version_sum) {
      return candidate.version_sum < best.version_sum;
    }
    if (candidate.preferred_penalty != best.preferred_penalty) {
      return candidate.preferred_penalty < best.preferred_penalty;
    }
    if (candidate.max_stage != best.max_stage) {
      return candidate.max_stage < best.max_stage;
    }
    return candidate.stages < best.stages;
  }

  std::unordered_map<std::string, int64_t> BuildLocalBufferFootprintBytes(
      const std::vector<PipelineStageInfo> &stage_infos) const {
    std::unordered_map<std::string, int64_t> footprints;
    auto add_buffer = [&](const Buffer &buffer) {
      std::string name(buffer->name);
      if (footprints.count(name)) {
        return;
      }
      int64_t bytes = 0;
      if (TryEstimateLocalBufferFootprintBytes(buffer, &bytes)) {
        footprints[name] = bytes;
      }
    };
    for (const auto &pinfo : stage_infos) {
      for (const BufferRegion &region : pinfo.reads) {
        add_buffer(region->buffer);
      }
      for (const BufferRegion &region : pinfo.writes) {
        add_buffer(region->buffer);
      }
    }
    return footprints;
  }

  TpuStageSearchScore ScoreTpuStageCandidate(
      int num_stages, const std::vector<PipelineStageInfo> &stage_infos,
      const std::vector<PipelineDependency> &dependencies,
      const std::vector<int> &candidate_stages) const {
    std::vector<PipelineStageInfo> candidate_infos = stage_infos;
    for (size_t i = 0; i < candidate_infos.size(); ++i) {
      candidate_infos[i].stage = candidate_stages[i];
    }

    TpuStageSearchScore score;
    score.valid = true;
    score.stages = candidate_stages;
    std::vector<int64_t> stage_dma_cost_units(num_stages, 0);
    std::vector<int64_t> stage_bdc_cost_units(num_stages, 0);
    std::vector<int64_t> legacy_stage_dma_cost_units(num_stages, 0);
    std::vector<int64_t> legacy_stage_bdc_cost_units(num_stages, 0);
    for (const auto &dep : dependencies) {
      int src_stage = candidate_stages[dep.src];
      int dst_stage = candidate_stages[dep.dst];
      if (src_stage > dst_stage) {
        score.valid = false;
        return score;
      }
      if (dep.kind == "RAW" && dst_stage > src_stage) {
        int stage_distance = dst_stage - src_stage;
        score.max_raw_stage_distance =
            std::max(score.max_raw_stage_distance, stage_distance);
        const std::string &src_role =
            stage_infos[dep.src].tpu_pipeline_role;
        const std::string &dst_role =
            stage_infos[dep.dst].tpu_pipeline_role;
        bool load_to_compute =
            src_role == "load" &&
            (dst_role == "compute" || dst_role == "local_copy");
        bool compute_to_store =
            (src_role == "compute" || src_role == "local_copy") &&
            dst_role == "store";
        if ((load_to_compute || compute_to_store) && stage_distance > 1) {
          score.expanded_prefetch_edges++;
          score.prefetch_distance_penalty += stage_distance - 1;
        }
        if (load_to_compute) {
          score.overlap_score += 4;
        } else if (compute_to_store) {
          score.overlap_score += 2;
        } else {
          score.overlap_score += 1;
        }
      }
    }

    int max_allowed_stage = std::max(0, num_stages - 1);
    int preferred_compute = std::min(1, max_allowed_stage);
    int preferred_store = std::min(2, max_allowed_stage);
    for (size_t i = 0; i < stage_infos.size(); ++i) {
      const std::string &role = stage_infos[i].tpu_pipeline_role;
      int stage = candidate_stages[i];
      if (stage < 0 || stage >= num_stages) {
        score.valid = false;
        return score;
      }
      score.max_stage = std::max(score.max_stage, stage);
      if (role == "compute" || role == "local_copy") {
        score.preferred_penalty += std::abs(stage - preferred_compute);
      } else if (role == "store") {
        score.preferred_penalty += std::abs(stage - preferred_store);
      }

      const TpuStageResourceEstimate &resource =
          stage_infos[i].tpu_resource;
      score.s2l_bytes += resource.s2l_bytes;
      score.l2s_bytes += resource.l2s_bytes;
      score.local_copy_bytes += resource.local_copy_bytes;
      score.bdc_ops += resource.bdc_ops;
      score.unknown_static_cost =
          score.unknown_static_cost || resource.unknown_static_cost;

      int64_t dma_bytes = resource.s2l_bytes + resource.l2s_bytes +
                          resource.local_copy_bytes;
      int64_t legacy_dma_units = DivUpInt64(dma_bytes, bm1690::kEuBytes);
      int64_t legacy_bdc_units = DivUpInt64(resource.bdc_ops, 1024);
      int64_t dma_units = resource.phase_gdma_cost_units +
                          resource.phase_local_copy_cost_units;
      int64_t bdc_units = resource.phase_bdc_cost_units;
      if (dma_units == 0 && bdc_units == 0 && dma_bytes + resource.bdc_ops > 0) {
        dma_units = legacy_dma_units;
        bdc_units = legacy_bdc_units;
      }
      stage_dma_cost_units[stage] += dma_units;
      stage_bdc_cost_units[stage] += bdc_units;
      legacy_stage_dma_cost_units[stage] += legacy_dma_units;
      legacy_stage_bdc_cost_units[stage] += legacy_bdc_units;
      score.dma_cost_units += dma_units;
      score.bdc_cost_units += bdc_units;
      score.legacy_dma_cost_units += legacy_dma_units;
      score.legacy_bdc_cost_units += legacy_bdc_units;
    }

    for (int stage = 0; stage < num_stages - 1; ++stage) {
      if (stage_dma_cost_units[stage] != 0 ||
          stage_bdc_cost_units[stage + 1] != 0) {
        score.pipeline_balance_penalty +=
            AbsDiffInt64(stage_dma_cost_units[stage],
                         stage_bdc_cost_units[stage + 1]);
      }
      if (legacy_stage_dma_cost_units[stage] != 0 ||
          legacy_stage_bdc_cost_units[stage + 1] != 0) {
        score.legacy_pipeline_balance_penalty +=
            AbsDiffInt64(legacy_stage_dma_cost_units[stage],
                         legacy_stage_bdc_cost_units[stage + 1]);
      }
    }

    std::unordered_map<std::string, int64_t> local_footprints =
        BuildLocalBufferFootprintBytes(candidate_infos);
    for (const auto &kv : BuildBufferVersionNeeds(candidate_infos, dependencies)) {
      score.version_sum += kv.second;
      score.max_required_versions =
          std::max(score.max_required_versions, kv.second);
      auto footprint_it = local_footprints.find(kv.first);
      if (footprint_it != local_footprints.end()) {
        score.versioned_lmem_bytes +=
            static_cast<int64_t>(kv.second) * footprint_it->second;
      }
    }
    score.lmem_bank_pressure_units =
        DivUpInt64(score.versioned_lmem_bytes, bm1690::kBankSize);
    return score;
  }

  bool TryApplyTpuStageOptimizerSearch(
      int num_stages, std::vector<PipelineStageInfo> *stage_infos,
      const std::vector<PipelineDependency> &dependencies,
      std::string *optimizer_decision) const {
    if (!EnvFlagEnabled("TL_TPU_STAGE_SEARCH", true)) {
      *optimizer_decision = "legacy_heuristic:stage_search_disabled";
      return false;
    }
    if (!IsTpuTarget() || num_stages < 2) {
      return false;
    }

    bool has_load = false;
    bool has_compute = false;
    std::vector<std::vector<int>> allowed_stages;
    allowed_stages.reserve(stage_infos->size());
    int64_t candidate_count = 1;
    for (auto &pinfo : *stage_infos) {
      pinfo.tpu_pipeline_role = ClassifyTpuPipelineRole(&pinfo);
      if (pinfo.tpu_pipeline_role == "unknown" ||
          pinfo.tpu_pipeline_role == "mixed_or_unknown") {
        *optimizer_decision = "legacy_heuristic:unsupported_tpu_role";
        return false;
      }
      has_load = has_load || pinfo.tpu_pipeline_role == "load";
      has_compute = has_compute || pinfo.tpu_pipeline_role == "compute" ||
                    pinfo.tpu_pipeline_role == "local_copy";
      allowed_stages.push_back(
          AllowedTpuStagesForRole(pinfo.tpu_pipeline_role, num_stages));
      candidate_count *= static_cast<int64_t>(allowed_stages.back().size());
      if (candidate_count > 8192) {
        *optimizer_decision = "legacy_heuristic:stage_search_too_large";
        return false;
      }
    }
    if (!has_load || !has_compute) {
      *optimizer_decision = "legacy_heuristic:missing_lc_pattern";
      return false;
    }

    TpuStageSearchScore best;
    std::vector<int> candidate(stage_infos->size(), 0);
    std::function<void(size_t)> enumerate = [&](size_t idx) {
      if (idx == allowed_stages.size()) {
        TpuStageSearchScore score = ScoreTpuStageCandidate(
            num_stages, *stage_infos, dependencies, candidate);
        if (score.valid && IsBetterTpuStageCandidate(score, best)) {
          best = std::move(score);
        }
        return;
      }
      for (int stage : allowed_stages[idx]) {
        candidate[idx] = stage;
        enumerate(idx + 1);
      }
    };
    enumerate(0);
    if (!best.valid) {
      *optimizer_decision = "legacy_heuristic:no_valid_stage_candidate";
      return false;
    }

    for (size_t i = 0; i < stage_infos->size(); ++i) {
      (*stage_infos)[i].stage = best.stages[i];
    }
    std::vector<size_t> indices(stage_infos->size());
    for (size_t i = 0; i < indices.size(); ++i) {
      indices[i] = i;
    }
    std::stable_sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
      const auto &l = (*stage_infos)[lhs];
      const auto &r = (*stage_infos)[rhs];
      if (l.stage != r.stage) {
        return l.stage < r.stage;
      }
      return l.original_order < r.original_order;
    });
    for (size_t order = 0; order < indices.size(); ++order) {
      (*stage_infos)[indices[order]].order = static_cast<int>(order);
    }
    *optimizer_decision = "tpu_stage_search_exact";
    return true;
  }

  void AddDependencyIfConflict(
      std::vector<PipelineDependency> *dependencies, int src, int dst,
      const char *kind, const Array<BufferRegion> &lhs,
      const Array<BufferRegion> &rhs) const {
    for (const BufferRegion &l : lhs) {
      for (const BufferRegion &r : rhs) {
        if (l->buffer == r->buffer && MayConflict(l->region, r->region)) {
          dependencies->push_back(
              {src, dst, kind, std::string(l->buffer->name)});
        }
      }
    }
  }

  bool DependencyExists(const std::vector<PipelineDependency> &dependencies,
                        int src, int dst, const char *kind,
                        const std::string &buffer) const {
    return std::any_of(dependencies.begin(), dependencies.end(),
                       [&](const PipelineDependency &dep) {
                         return dep.src == src && dep.dst == dst &&
                                dep.kind == kind && dep.buffer == buffer;
                       });
  }

  void AddDependencyIfNameConflict(
      std::vector<PipelineDependency> *dependencies, int src, int dst,
      const char *kind, const std::unordered_set<std::string> &lhs,
      const std::unordered_set<std::string> &rhs) const {
    for (const std::string &buffer : lhs) {
      if (rhs.count(buffer) &&
          !DependencyExists(*dependencies, src, dst, kind, buffer)) {
        dependencies->push_back({src, dst, kind, buffer});
      }
    }
  }

  std::vector<PipelineDependency> BuildDependencies(
      const std::vector<PipelineStageInfo> &stage_infos) const {
    std::vector<PipelineDependency> dependencies;
    for (size_t i = 0; i < stage_infos.size(); ++i) {
      for (size_t j = i + 1; j < stage_infos.size(); ++j) {
        AddDependencyIfConflict(&dependencies, stage_infos[i].original_order,
                                stage_infos[j].original_order, "RAW",
                                stage_infos[i].writes, stage_infos[j].reads);
        AddDependencyIfConflict(&dependencies, stage_infos[i].original_order,
                                stage_infos[j].original_order, "WAR",
                                stage_infos[i].reads, stage_infos[j].writes);
        AddDependencyIfConflict(&dependencies, stage_infos[i].original_order,
                                stage_infos[j].original_order, "WAW",
                                stage_infos[i].writes, stage_infos[j].writes);
        AddDependencyIfNameConflict(&dependencies,
                                    stage_infos[i].original_order,
                                    stage_infos[j].original_order, "RAW",
                                    stage_infos[i].tpu_write_buffers,
                                    stage_infos[j].tpu_read_buffers);
        AddDependencyIfNameConflict(&dependencies,
                                    stage_infos[i].original_order,
                                    stage_infos[j].original_order, "WAR",
                                    stage_infos[i].tpu_read_buffers,
                                    stage_infos[j].tpu_write_buffers);
        AddDependencyIfNameConflict(&dependencies,
                                    stage_infos[i].original_order,
                                    stage_infos[j].original_order, "WAW",
                                    stage_infos[i].tpu_write_buffers,
                                    stage_infos[j].tpu_write_buffers);
      }
    }
    return dependencies;
  }

  std::unordered_map<std::string, int> BuildBufferVersionNeeds(
      const std::vector<PipelineStageInfo> &stage_infos,
      const std::vector<PipelineDependency> &dependencies) const {
    std::unordered_map<std::string, int> versions;
    std::unordered_map<int, int> order_to_stage;
    for (const auto &pinfo : stage_infos) {
      order_to_stage[pinfo.original_order] = pinfo.stage;
      for (const BufferRegion &read : pinfo.reads) {
        versions.emplace(std::string(read->buffer->name), 1);
      }
      for (const BufferRegion &write : pinfo.writes) {
        versions.emplace(std::string(write->buffer->name), 1);
      }
      for (const std::string &buffer : pinfo.tpu_read_buffers) {
        versions.emplace(buffer, 1);
      }
      for (const std::string &buffer : pinfo.tpu_write_buffers) {
        versions.emplace(buffer, 1);
      }
    }

    for (const auto &dep : dependencies) {
      if (dep.kind != "RAW") {
        continue;
      }
      auto src_it = order_to_stage.find(dep.src);
      auto dst_it = order_to_stage.find(dep.dst);
      if (src_it == order_to_stage.end() || dst_it == order_to_stage.end()) {
        continue;
      }
      int stage_distance = std::abs(dst_it->second - src_it->second);
      if (stage_distance > 0) {
        int needed = stage_distance + 1;
        auto it = versions.find(dep.buffer);
        if (it == versions.end()) {
          versions[dep.buffer] = needed;
        } else {
          it->second = std::max(it->second, needed);
        }
      }
    }
    return versions;
  }

  bool HasTpuOp(const PipelineStageInfo &pinfo,
                const std::string &op_name) const {
    for (const auto &op_info : pinfo.tpu_ops) {
      if (op_info.op_name == op_name) {
        return true;
      }
    }
    return false;
  }

  bool HasTpuPhaseEngine(const PipelineStageInfo &pinfo,
                         const std::string &engine) const {
    for (const auto &op_info : pinfo.tpu_ops) {
      for (const auto &phase : op_info.phases) {
        if (phase.engine == engine) {
          return true;
        }
      }
    }
    return false;
  }

  bool HasConservativeTpuPhase(const PipelineStageInfo &pinfo) const {
    for (const auto &op_info : pinfo.tpu_ops) {
      for (const auto &phase : op_info.phases) {
        if (phase.conservative) {
          return true;
        }
      }
    }
    return false;
  }

  bool IsTpuS2LGdmaLoad(const PipelineStageInfo &pinfo) const {
    if (pinfo.tpu_pipeline_role != "load") {
      return false;
    }
    for (const auto &op_info : pinfo.tpu_ops) {
      if (op_info.op_name == "ppl.copy" && op_info.transaction == "S2L") {
        for (const auto &phase : op_info.phases) {
          if (phase.engine == "gdma" && phase.role == "load") {
            return true;
          }
        }
      }
    }
    return false;
  }

  int FindFirstStatementAfter(
      const std::vector<PipelineStageInfo> &stage_infos, int after,
      const std::function<bool(const PipelineStageInfo &)> &pred) const {
    for (const auto &pinfo : stage_infos) {
      if (pinfo.original_order > after && pred(pinfo)) {
        return pinfo.original_order;
      }
    }
    return -1;
  }

  std::vector<int> StatementRange(int begin, int end) const {
    std::vector<int> result;
    for (int stmt = begin; stmt <= end; ++stmt) {
      result.push_back(stmt);
    }
    return result;
  }

  std::vector<std::string> DependenciesCrossedByMovingEarlier(
      int moved_statement, int first_crossed_statement,
      int last_crossed_statement,
      const std::vector<PipelineDependency> &dependencies) const {
    std::vector<std::string> blockers;
    for (const auto &dep : dependencies) {
      if (dep.dst != moved_statement) {
        continue;
      }
      if (dep.src < first_crossed_statement ||
          dep.src > last_crossed_statement) {
        continue;
      }
      blockers.push_back("stmt" + std::to_string(dep.src) + ":" + dep.kind +
                         ":" + dep.buffer);
    }
    std::sort(blockers.begin(), blockers.end());
    return blockers;
  }

  bool SegmentHasEngine(const std::vector<PipelineStageInfo> &stage_infos,
                        const std::vector<int> &statements,
                        const std::string &engine) const {
    for (int stmt : statements) {
      if (stmt < 0 || stmt >= static_cast<int>(stage_infos.size())) {
        continue;
      }
      if (HasTpuPhaseEngine(stage_infos[stmt], engine)) {
        return true;
      }
    }
    return false;
  }

  bool SegmentHasConservativePhase(
      const std::vector<PipelineStageInfo> &stage_infos,
      const std::vector<int> &statements) const {
    for (int stmt : statements) {
      if (stmt < 0 || stmt >= static_cast<int>(stage_infos.size())) {
        continue;
      }
      if (HasConservativeTpuPhase(stage_infos[stmt]) ||
          stage_infos[stmt].tpu_resource.unknown_static_cost) {
        return true;
      }
    }
    return false;
  }

  bool StatementWritesFloat32LocalBuffer(
      const PipelineStageInfo &pinfo) const {
    for (const BufferRegion &region : pinfo.writes) {
      const Buffer &buffer = region->buffer;
      if (IsTpuLocalScope(buffer.scope()) &&
          buffer->dtype == DataType::Float(32)) {
        return true;
      }
    }
    return false;
  }

  bool PlannedStagesAreAllZero(
      const std::vector<PipelineStageInfo> &stage_infos) const {
    for (const auto &pinfo : stage_infos) {
      if (pinfo.stage != 0) {
        return false;
      }
    }
    return true;
  }

  bool LoopHasTpuPhaseEngine(
      const std::vector<PipelineStageInfo> &stage_infos,
      const std::string &engine) const {
    for (const auto &pinfo : stage_infos) {
      if (HasTpuPhaseEngine(pinfo, engine)) {
        return true;
      }
    }
    return false;
  }

  bool ShouldUseTpuParallelGuardRegion(
      const std::vector<PipelineStageInfo> &stage_infos,
      const TpuPipelinePolicyDecision &policy) const {
    if (!IsTpuTarget() ||
        !EnvFlagEnabled("TL_TPU_PARALLEL_GUARD_REGION", true)) {
      return false;
    }
    if (policy.kind == "unsupported_tpu_role" ||
        policy.allow_stage_optimization ||
        policy.enable_buffer_version_hints ||
        !policy.phase_model_available) {
      return false;
    }
    if (!PlannedStagesAreAllZero(stage_infos)) {
      return false;
    }
    return LoopHasTpuPhaseEngine(stage_infos, "gdma") &&
           LoopHasTpuPhaseEngine(stage_infos, "bdc");
  }

  std::vector<TpuOverlapWindowInfo> AnalyzeAttentionMixedOverlapWindows(
      const std::vector<PipelineStageInfo> &stage_infos,
      const std::vector<PipelineDependency> &dependencies,
      const TpuPipelinePolicyDecision &policy) const {
    std::vector<TpuOverlapWindowInfo> windows;
    if (!IsTpuTarget() || policy.policy_family != "attention_mixed") {
      return windows;
    }

    std::vector<int> s2l_loads;
    std::vector<int> gemms;
    for (const auto &pinfo : stage_infos) {
      if (IsTpuS2LGdmaLoad(pinfo)) {
        s2l_loads.push_back(pinfo.original_order);
      }
      if (HasTpuOp(pinfo, "ppl.gemm")) {
        gemms.push_back(pinfo.original_order);
      }
    }

    TpuOverlapWindowInfo k_window;
    k_window.name = "WindowA_K_prefetch_next_iter";
    k_window.kind = "inter_iteration_late_prefetch";
    k_window.materialization_kind =
        "stage_split_or_late_parallel_marker";
    if (s2l_loads.empty() || gemms.size() < 2) {
      k_window.status = "not_applicable";
      k_window.blocked_reason = "need_at_least_one_s2l_load_and_two_gemms";
    } else {
      int first_load = s2l_loads[0];
      int first_gemm = FindFirstStatementAfter(
          stage_infos, first_load,
          [&](const PipelineStageInfo &pinfo) { return HasTpuOp(pinfo, "ppl.gemm"); });
      int output_gemm =
          gemms.size() >= 2 ? gemms[1] : FindFirstStatementAfter(
                                            stage_infos, first_gemm,
                                            [&](const PipelineStageInfo &pinfo) {
                                              return HasTpuOp(pinfo, "ppl.gemm");
                                            });
      k_window.load_statement = first_load;
      k_window.consumer_statement = output_gemm;
      if (output_gemm < 0 || first_gemm < 0 || output_gemm <= first_gemm) {
        k_window.status = "not_applicable";
        k_window.blocked_reason = "missing_late_output_gemm_window";
      } else {
        k_window.status = "candidate_needs_split_or_versioning";
        k_window.overlap_statements.push_back(output_gemm);
        for (const auto &buffer : stage_infos[first_load].tpu_write_buffers) {
          k_window.required_versions.push_back({buffer, 2});
        }
        std::sort(k_window.required_versions.begin(),
                  k_window.required_versions.end());
        k_window.blocked_reason =
            "generic_stage_pipeline_can_start_next_k_load_before_current_score_gemm";
        k_window.notes =
            "late_prefetch_after_score_gemm_may_avoid_pingpong_but_needs_family_materializer";
      }
    }
    windows.push_back(std::move(k_window));

    TpuOverlapWindowInfo v_window;
    v_window.name = "WindowB_V_prefetch_same_iter";
    v_window.kind = "intra_iteration_prefetch";
    v_window.materialization_kind = "parallel_marker_inside_iteration";
    if (s2l_loads.size() < 2 || gemms.size() < 2) {
      v_window.status = "not_applicable";
      v_window.blocked_reason = "need_two_s2l_loads_and_two_gemms";
    } else {
      int first_load = s2l_loads[0];
      int first_gemm = FindFirstStatementAfter(
          stage_infos, first_load,
          [&](const PipelineStageInfo &pinfo) { return HasTpuOp(pinfo, "ppl.gemm"); });
      int second_load = s2l_loads[1];
      int second_gemm = FindFirstStatementAfter(
          stage_infos, second_load,
          [&](const PipelineStageInfo &pinfo) { return HasTpuOp(pinfo, "ppl.gemm"); });
      v_window.load_statement = second_load;
      v_window.consumer_statement = second_gemm;
      if (first_gemm < 0 || second_load <= first_gemm + 1 ||
          second_gemm < 0) {
        v_window.status = "not_applicable";
        v_window.blocked_reason = "missing_softmax_chain_between_gemms";
      } else {
        v_window.overlap_statements =
            StatementRange(first_gemm + 1, second_load - 1);
        std::vector<std::string> blockers =
            DependenciesCrossedByMovingEarlier(second_load, first_gemm + 1,
                                               second_load - 1, dependencies);
        if (!blockers.empty()) {
          v_window.status = "blocked_dependency";
          v_window.blocked_reason = StringVectorToString(blockers);
        } else if (SegmentHasEngine(stage_infos, v_window.overlap_statements,
                                    "gdma")) {
          v_window.status = "blocked_gdma_conflict";
          v_window.blocked_reason =
              "overlap_segment_contains_gdma_phase";
        } else if (!SegmentHasEngine(stage_infos, v_window.overlap_statements,
                                     "bdc")) {
          v_window.status = "blocked_no_bdc_work";
          v_window.blocked_reason =
              "overlap_segment_has_no_bdc_phase_to_hide_load";
        } else {
          v_window.status = "candidate_legal_certificate_only";
          v_window.blocked_reason =
              "materializer_not_implemented";
          v_window.notes =
              "no_crossed_dependency_or_gdma_phase_in_softmax_segment";
        }
        if (SegmentHasConservativePhase(stage_infos,
                                        v_window.overlap_statements)) {
          if (!v_window.notes.empty()) {
            v_window.notes += ";";
          }
          v_window.notes += "score_has_conservative_or_unknown_static_cost";
        }
      }
    }
    windows.push_back(std::move(v_window));
    return windows;
  }

  bool TrySelectAttentionWindowBMaterializer(
      const std::vector<PipelineStageInfo> &stage_infos,
      const std::vector<PipelineDependency> &dependencies,
      const TpuPipelinePolicyDecision &policy,
      TpuOverlapWindowInfo *selected_window) const {
    if (!EnvFlagEnabled("TL_TPU_ATTENTION_WINDOW_B_MATERIALIZE", true)) {
      return false;
    }
    std::vector<TpuOverlapWindowInfo> windows =
        AnalyzeAttentionMixedOverlapWindows(stage_infos, dependencies, policy);
    for (const auto &window : windows) {
      if (window.name == "WindowB_V_prefetch_same_iter" &&
          window.status == "candidate_legal_certificate_only" &&
          !window.overlap_statements.empty() && window.load_statement >= 0 &&
          window.consumer_statement >= 0) {
        if (StatementWritesFloat32LocalBuffer(stage_infos[window.load_statement])) {
          continue;
        }
        *selected_window = window;
        selected_window->status = "selected_for_materialization";
        selected_window->blocked_reason = "[]";
        if (!selected_window->notes.empty()) {
          selected_window->notes += ";";
        }
        selected_window->notes += "guarded_window_b_materializer_enabled";
        return true;
      }
    }
    return false;
  }

  std::string ProfitabilityGuardStatus(
      const TpuPipelinePolicyDecision &policy) const {
    if (!IsTpuTarget()) {
      return "disabled_non_tpu";
    }
    if (policy.policy_family == "pure_gemm") {
      return "model_bound_non_regression";
    }
    if (policy.policy_family == "reduction_norm") {
      return "legality_and_materialization_only";
    }
    if (policy.policy_family == "attention_mixed") {
      return "legality_or_opt_in_materializer_only";
    }
    return "not_modeled";
  }

  std::string ProfitabilityGuardRecommendation(
      const TpuPipelinePolicyDecision &policy) const {
    if (!IsTpuTarget()) {
      return "not_applicable";
    }
    if (policy.policy_family == "pure_gemm") {
      return "model_local_claim_only";
    }
    if (policy.policy_family == "reduction_norm") {
      return "require_serialized_runtime_evidence_before_speedup_claim";
    }
    if (policy.policy_family == "attention_mixed") {
      return "keep_materializers_opt_in_until_positive_timing_region";
    }
    return "do_not_materialize_for_performance_without_family_model";
  }

  void DumpCertificate(const ForNode *loop,
                       const std::vector<PipelineStageInfo> &stage_infos,
                       int num_stages, bool has_async_copy,
                       const std::string &optimizer_decision,
                       const TpuPipelinePolicyDecision &policy,
                       bool has_tpu_parallel_guard_region) {
    if (GetPipelineCertificateSink().empty()) {
      return;
    }
    std::ostringstream os;
    os << "PIPELINE_CERTIFICATE_BEGIN\n";
    os << "function: " << global_symbol_ << "\n";
    os << "loop_id: " << loop_counter_++ << "\n";
    os << "loop_var: " << loop->loop_var << "\n";
    os << "min: " << loop->min << "\n";
    os << "extent: " << loop->extent << "\n";
    os << "target: " << target_->str() << "\n";
    os << "num_stages: " << num_stages << "\n";
    os << "optimizer_decision: " << optimizer_decision << "\n";
    os << "pipeline_policy: " << policy.kind << "\n";
    os << "policy_family: " << policy.policy_family << "\n";
    os << "evidence_tier: " << policy.evidence_tier << "\n";
    os << "policy_reason: " << policy.reason << "\n";
    os << "policy_stage_optimization: "
       << (policy.allow_stage_optimization ? "enabled" : "disabled") << "\n";
    os << "policy_buffer_version_hints: "
       << (policy.enable_buffer_version_hints ? "enabled" : "disabled")
       << "\n";
    os << "policy_stage_aware_lmem: "
       << (policy.enable_stage_aware_lmem ? "enabled" : "disabled") << "\n";
    os << "policy_phase_model: "
       << (policy.phase_model_available ? "available" : "unavailable")
       << "\n";
    os << "known_unknowns: "
       << (policy.known_unknowns.empty() ? "[]" : policy.known_unknowns)
       << "\n";
    int64_t trip_count = 0;
    bool trip_count_known = TryGetPositiveIntImm(loop->extent, &trip_count);
    os << "profitability_guard:\n";
    os << "  status: " << ProfitabilityGuardStatus(policy) << "\n";
    os << "  trip_count_static: "
       << (trip_count_known ? std::to_string(trip_count) : "unknown") << "\n";
    os << "  calibrated_runtime_model: unavailable\n";
    os << "  marker_overhead_model: unavailable\n";
    os << "  recommendation: " << ProfitabilityGuardRecommendation(policy)
       << "\n";
    os << "has_async_copy: " << (has_async_copy ? "true" : "false") << "\n";
    os << "async_stages: " << (has_async_copy ? "[0]" : "[]") << "\n";
    os << "parallel_guard_region:\n";
    os << "  enabled: "
       << (has_tpu_parallel_guard_region ? "true" : "false") << "\n";
    os << "  annotation: "
       << (has_tpu_parallel_guard_region
               ? "tl_tpu_parallel_guard_region"
               : "none")
       << "\n";
    os << "  role: bm1690_mixed_engine_liveness_guard\n";
    os << "manual_constraints:\n";
    for (const char *key :
         {"tl_pipeline_stage", "tl_pipeline_order", "tl_pipeline_sync",
          "tl_pipeline_group"}) {
      if (auto value = loop->annotations.Get(key)) {
        os << "  " << key << ": " << ObjectRefToString(value.value()) << "\n";
      }
    }
    os << "annotations:\n";
    for (const auto &[key, value] : loop->annotations) {
      os << "  " << key << ": " << ObjectRefToString(value) << "\n";
    }
    std::vector<PipelineDependency> dependencies = BuildDependencies(stage_infos);
    os << "dependencies:\n";
    for (const PipelineDependency &dep : dependencies) {
      os << "  - src: " << dep.src << "\n";
      os << "    dst: " << dep.dst << "\n";
      os << "    kind: " << dep.kind << "\n";
      os << "    buffer: " << dep.buffer << "\n";
    }
    os << "buffer_versions:\n";
    for (const auto &kv : BuildBufferVersionNeeds(stage_infos, dependencies)) {
      os << "  - buffer: " << kv.first << "\n";
      os << "    required_versions: " << kv.second << "\n";
    }
    os << "tpu_cost_model:\n";
    if (!IsTpuTarget()) {
      os << "  status: disabled_non_tpu\n";
    } else if (!policy.allow_stage_optimization) {
      os << "  status: inactive_policy_stage_optimization_disabled\n";
      os << "  model: bm1690_relative_v2_phase_cost\n";
    } else {
      std::vector<int> planned_stages;
      planned_stages.reserve(stage_infos.size());
      bool stages_in_range = true;
      for (const auto &pinfo : stage_infos) {
        planned_stages.push_back(pinfo.stage);
        if (pinfo.stage < 0 || pinfo.stage >= num_stages) {
          stages_in_range = false;
        }
      }
      if (!stages_in_range) {
        os << "  status: stages_out_of_range\n";
      } else {
        TpuStageSearchScore final_score = ScoreTpuStageCandidate(
            num_stages, stage_infos, dependencies, planned_stages);
        os << "  status: "
           << (policy.allow_stage_optimization ? "active" : "observed")
           << "\n";
        os << "  model: bm1690_relative_v2_phase_cost\n";
        os << "  valid: " << (final_score.valid ? "true" : "false") << "\n";
        os << "  overlap_score: " << final_score.overlap_score << "\n";
        os << "  s2l_bytes: " << final_score.s2l_bytes << "\n";
        os << "  l2s_bytes: " << final_score.l2s_bytes << "\n";
        os << "  local_copy_bytes: " << final_score.local_copy_bytes << "\n";
        os << "  bdc_ops: " << final_score.bdc_ops << "\n";
        os << "  dma_cost_units: " << final_score.dma_cost_units << "\n";
        os << "  bdc_cost_units: " << final_score.bdc_cost_units << "\n";
        os << "  legacy_dma_cost_units: "
           << final_score.legacy_dma_cost_units << "\n";
        os << "  legacy_bdc_cost_units: "
           << final_score.legacy_bdc_cost_units << "\n";
        os << "  pipeline_balance_penalty: "
           << final_score.pipeline_balance_penalty << "\n";
        os << "  legacy_pipeline_balance_penalty: "
           << final_score.legacy_pipeline_balance_penalty << "\n";
        os << "  max_raw_stage_distance: "
           << final_score.max_raw_stage_distance << "\n";
        os << "  expanded_prefetch_edges: "
           << final_score.expanded_prefetch_edges << "\n";
        os << "  prefetch_distance_penalty: "
           << final_score.prefetch_distance_penalty << "\n";
        os << "  versioned_lmem_bytes: "
           << final_score.versioned_lmem_bytes << "\n";
        os << "  lmem_bank_pressure_units: "
           << final_score.lmem_bank_pressure_units << "\n";
        os << "  max_required_versions: "
           << final_score.max_required_versions << "\n";
        os << "  version_sum: " << final_score.version_sum << "\n";
        os << "  unknown_static_cost: "
           << (final_score.unknown_static_cost ? "true" : "false") << "\n";
      }
    }
    std::unordered_map<std::string, int64_t> local_footprints =
        BuildLocalBufferFootprintBytes(stage_infos);
    std::vector<std::pair<std::string, int64_t>> ordered_footprints(
        local_footprints.begin(), local_footprints.end());
    std::sort(ordered_footprints.begin(), ordered_footprints.end());
    os << "local_buffer_footprints:\n";
    for (const auto &kv : ordered_footprints) {
      os << "  - buffer: " << kv.first << "\n";
      os << "    bm1690_tpu_align_bytes: " << kv.second << "\n";
    }
    int64_t hidden_workspace_lmem = 0;
    std::unordered_set<std::string> emitted_workspace_buffers;
    os << "hidden_workspace_lmem:\n";
    os << "  buffers:\n";
    for (const auto &pinfo : stage_infos) {
      for (const std::string &buffer : pinfo.tpu_workspace_buffers) {
        if (emitted_workspace_buffers.count(buffer)) {
          continue;
        }
        emitted_workspace_buffers.insert(buffer);
        int64_t bytes = 0;
        auto workspace_footprint_it =
            pinfo.tpu_workspace_footprints.find(buffer);
        if (workspace_footprint_it != pinfo.tpu_workspace_footprints.end()) {
          bytes = workspace_footprint_it->second;
        } else {
          auto footprint_it = local_footprints.find(buffer);
          if (footprint_it != local_footprints.end()) {
            bytes = footprint_it->second;
          }
        }
        hidden_workspace_lmem += bytes;
        os << "    - buffer: " << buffer << "\n";
        os << "      bm1690_tpu_align_bytes: " << bytes << "\n";
      }
    }
    os << "  total_bm1690_tpu_align_bytes: " << hidden_workspace_lmem
       << "\n";
    os << "loop_carried_dependencies:\n";
    for (const auto &pinfo : stage_infos) {
      for (const std::string &buffer :
           SortedIntersection(pinfo.tpu_read_buffers, pinfo.tpu_write_buffers)) {
        os << "  - statement: " << pinfo.original_order << "\n";
        os << "    buffer: " << buffer << "\n";
        os << "    reason: readwrite_same_statement_candidate\n";
      }
    }
    if (policy.policy_family == "attention_mixed") {
      std::vector<TpuOverlapWindowInfo> overlap_windows =
          AnalyzeAttentionMixedOverlapWindows(stage_infos, dependencies, policy);
      TpuOverlapWindowInfo selected_window;
      bool has_selected_window = TrySelectAttentionWindowBMaterializer(
          stage_infos, dependencies, policy, &selected_window);
      if (has_selected_window) {
        for (auto &window : overlap_windows) {
          if (window.name == selected_window.name) {
            window = selected_window;
          }
        }
      }
      os << "overlap_windows:\n";
      for (const auto &window : overlap_windows) {
        os << "  - name: " << window.name << "\n";
        os << "    kind: " << window.kind << "\n";
        os << "    status: " << window.status << "\n";
        os << "    materialization_kind: " << window.materialization_kind
           << "\n";
        os << "    load_statement: " << window.load_statement << "\n";
        os << "    consumer_statement: " << window.consumer_statement << "\n";
        os << "    overlap_statements: "
           << IntVectorToString(window.overlap_statements) << "\n";
        os << "    required_versions:\n";
        for (const auto &version : window.required_versions) {
          os << "      - buffer: " << version.first << "\n";
          os << "        versions: " << version.second << "\n";
        }
        os << "    blocked_reason: " << window.blocked_reason << "\n";
        os << "    notes: " << (window.notes.empty() ? "[]" : window.notes)
           << "\n";
      }
      os << "attention_materialization_plan:\n";
      if (has_selected_window) {
        os << "  status: selected\n";
        os << "  window: " << selected_window.name << "\n";
        os << "  materialization_kind: "
           << selected_window.materialization_kind << "\n";
        os << "  load_statement: " << selected_window.load_statement << "\n";
        os << "  overlap_statements: "
           << IntVectorToString(selected_window.overlap_statements) << "\n";
        os << "  consumer_statement: " << selected_window.consumer_statement
           << "\n";
      } else {
        os << "  status: not_selected\n";
        os << "  reason: no_candidate_legal_window_b_or_disabled\n";
      }
    }
    os << "engine_timeline:\n";
    for (const auto &pinfo : stage_infos) {
      os << "  - statement: " << pinfo.original_order << "\n";
      os << "    planned_stage: " << pinfo.stage << "\n";
      os << "    phases:\n";
      for (const auto &op_info : pinfo.tpu_ops) {
        for (const auto &phase : op_info.phases) {
          os << "      - op: " << op_info.op_name << "\n";
          os << "        phase: " << phase.name << "\n";
          os << "        engine: " << phase.engine << "\n";
          os << "        role: " << phase.role << "\n";
          os << "        conservative: "
             << (phase.conservative ? "true" : "false") << "\n";
          os << "        hidden_workspace: "
             << (phase.hidden_workspace ? "true" : "false") << "\n";
        }
      }
    }
    os << "statements:\n";
    for (const auto &pinfo : stage_infos) {
      os << "  - original_order: " << pinfo.original_order << "\n";
      os << "    planned_order: " << pinfo.order << "\n";
      os << "    planned_stage: " << pinfo.stage << "\n";
      os << "    copy_stage: " << (pinfo.copy_stage ? "true" : "false")
         << "\n";
      os << "    last_use_stage: " << pinfo.last_use_stage << "\n";
      os << "    tpu_pipeline_role: " << pinfo.tpu_pipeline_role << "\n";
      os << "    tpu_ops: " << TpuOpInfosToString(pinfo.tpu_ops) << "\n";
      os << "    tpu_resource: "
         << TpuStageResourceEstimateToString(pinfo.tpu_resource) << "\n";
      os << "    tpu_read_buffers: "
         << StringSetToString(pinfo.tpu_read_buffers) << "\n";
      os << "    tpu_write_buffers: "
         << StringSetToString(pinfo.tpu_write_buffers) << "\n";
      os << "    tpu_workspace_buffers: "
         << StringSetToString(pinfo.tpu_workspace_buffers) << "\n";
      os << "    reads: " << BufferRegionsToString(pinfo.reads) << "\n";
      os << "    writes: " << BufferRegionsToString(pinfo.writes) << "\n";
    }
    os << "PIPELINE_CERTIFICATE_END\n";
    EmitPipelineCertificate(os.str());
  }

  void AccumulateLocalFootprintFromRegions(
      const Array<BufferRegion> &regions,
      std::unordered_set<std::string> *seen_buffers,
      TpuStageResourceEstimate *estimate) const {
    for (const BufferRegion &region : regions) {
      const Buffer &buffer = region->buffer;
      std::string name(buffer->name);
      if (seen_buffers->count(name)) {
        continue;
      }
      int64_t bytes = 0;
      if (TryEstimateLocalBufferFootprintBytes(buffer, &bytes)) {
        estimate->local_footprint_bytes += bytes;
        seen_buffers->insert(name);
      } else if (IsTpuLocalScope(buffer.scope())) {
        estimate->unknown_static_cost = true;
      }
    }
  }

  PipelineStageInfo MakePipelineStageInfo(Stmt stmt, int idx) {
    Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
                /*body*/ stmt);
    Array<Array<BufferRegion>> access =
        GetBlockReadWriteRegion(block, buffer_data_to_buffer_);

    PipelineStageInfo pinfo;
    pinfo.reads = std::move(access[0]);
    pinfo.writes = std::move(access[1]);
    pinfo.original_order = idx;
    pinfo.tpu_ops = TpuOpSummaryCollector(buffer_data_to_buffer_).Collect(stmt);
    TpuBufferAccessNameCollector access_name_collector(buffer_data_to_buffer_);
    access_name_collector.Collect(stmt);
    pinfo.tpu_read_buffers = access_name_collector.reads();
    pinfo.tpu_write_buffers = access_name_collector.writes();
    TpuWorkspaceNameCollector workspace_collector(buffer_data_to_buffer_);
    workspace_collector.Collect(stmt);
    pinfo.tpu_workspace_buffers = workspace_collector.workspaces();
    pinfo.tpu_workspace_footprints = workspace_collector.workspace_footprints();
    pinfo.tpu_resource =
        TpuStageResourceEstimateCollector(buffer_data_to_buffer_).Collect(stmt);
    std::unordered_set<std::string> footprint_buffers;
    AccumulateLocalFootprintFromRegions(pinfo.reads, &footprint_buffers,
                                        &pinfo.tpu_resource);
    AccumulateLocalFootprintFromRegions(pinfo.writes, &footprint_buffers,
                                        &pinfo.tpu_resource);

    // copy stage should only have one reads and one writes
    bool write_to_shared = false;
    bool read_from_global = false;
    for (auto region : pinfo.reads)
      if (region->buffer.scope() == "global")
        read_from_global = true;
    for (auto region : pinfo.writes)
      if (region->buffer.scope() == "shared" ||
          region->buffer.scope() == "shared.dyn")
        write_to_shared = true;

    pinfo.copy_stage = write_to_shared && read_from_global;

    return std::move(pinfo);
  }

  Stmt VisitStmt_(const ForNode *loop) final {
    auto num_stages_anno = loop->annotations.Get("num_stages");
    if (!num_stages_anno.defined())
      return StmtExprMutator::VisitStmt_(loop);
    int num_stages = num_stages_anno.as<IntImmNode>()->value;
    Stmt pipeline_body{nullptr};
    if (const auto *realize = loop->body.as<BlockRealizeNode>()) {
      const auto &block = realize->block;
      for (const auto &buffer : block->alloc_buffers) {
        ICHECK(buffer->IsInstance<BufferNode>());
        buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
      if (const auto *seq_stmt = block->body.as<SeqStmtNode>()) {
        pipeline_body = block->body;
      } else if (const auto *if_then_else = block->body.as<IfThenElseNode>()) {
        // should assert else case is nullptr
        ICHECK(!if_then_else->else_case.defined())
            << "Pipeline_Planning: Can't handle the body of the loop because "
               "it is not a SeqStmt";
        pipeline_body = if_then_else->then_case;
      } else {
        LOG(FATAL) << "Pipeline_Planning: Can't handle the body of the loop "
                      "because it is not a SeqStmt or IfThenElse";
      }
    } else {
      pipeline_body = loop->body;
    }
    const SeqStmtNode *pipeline_body_seq = pipeline_body.as<SeqStmtNode>();
    CHECK(pipeline_body_seq)
        << "ValueError: The body of the software pipeline "
           "should be SeqStmt, got "
        << pipeline_body->GetTypeKey() << " " << pipeline_body;
    CHECK(num_stages >= 1);
    CHECK(loop->kind == ForKind::kSerial);

    std::vector<PipelineStageInfo> pipeline_stage_infos;
    for (size_t i = 0; i < pipeline_body_seq->size(); i++) {
      auto pinfo = MakePipelineStageInfo(pipeline_body_seq->seq[i], i);
      pipeline_stage_infos.push_back(std::move(pinfo));
    }

    // analysis use-def chain
    for (auto &pinfo : pipeline_stage_infos) {
      for (int i = pinfo.original_order + 1;
           i < static_cast<int>(pipeline_body_seq->size()); i++) {
        if (!pinfo.copy_stage)
          continue;
        for (const BufferRegion &read : pipeline_stage_infos[i].reads) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const BufferRegion &r) {
                             return r->buffer == read->buffer &&
                                    MayConflict(r->region, read->region);
                           }) != pinfo.writes.end()) {
            pinfo.last_use_stage = std::max(pinfo.last_use_stage, i);
          }
        }
        for (const BufferRegion &write : pipeline_stage_infos[i].writes) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const BufferRegion &r) {
                             return r->buffer == write->buffer &&
                                    MayConflict(r->region, write->region);
                           }) != pinfo.writes.end()) {
            LOG(FATAL) << "Pipeline planning error: Multiple writes to "
                          "overlapping buffer regions detected. "
                       << "Stage " << pinfo.original_order << " and stage " << i
                       << " are both writing to buffer '" << write->buffer->name
                       << "' with overlapping regions. This is not supported "
                          "in pipeline planning.";
          }
        }
      }
    }

    // Making stages and orders
    int order_idx = 0;
    for (auto &pinfo : pipeline_stage_infos) {
      if (pinfo.copy_stage && pinfo.last_use_stage != -1)
        continue;
      pinfo.order = order_idx++;
      pinfo.stage = num_stages;
      for (auto &pinfo_1 : pipeline_stage_infos) {
        if (pinfo_1.copy_stage &&
            pinfo_1.last_use_stage == pinfo.original_order) {
          pinfo_1.order = order_idx++;
          pinfo_1.stage = 0;
        }
      }
    }
    ICHECK(size_t(order_idx) == pipeline_stage_infos.size())
        << "The number of stages should be equal to the number of pipeline "
           "stages. "
        << "Got " << order_idx << " stages and " << pipeline_stage_infos.size()
        << " pipeline stages.";

    // if all the copy is at the end of the order, we can move these copy to the
    // beginning of the order and shrink the stage offset by 1.
    int copy_stage_at_end = [&]() {
      int copy_stage_cnt = 0;
      int copy_order_min = pipeline_stage_infos.size();
      int non_copy_order_max = 0;
      for (auto &pinfo : pipeline_stage_infos) {
        if (pinfo.copy_stage) {
          copy_stage_cnt++;
          copy_order_min = std::min(copy_order_min, pinfo.order);
        } else {
          non_copy_order_max = std::max(non_copy_order_max, pinfo.order);
        }
      }
      if (copy_order_min > non_copy_order_max)
        return copy_stage_cnt;
      return -1;
    }();
    if (copy_stage_at_end > 0 && num_stages >= 2) {
      for (auto &pinfo : pipeline_stage_infos) { // move copy to the beginning
        pinfo.order =
            (pinfo.order + copy_stage_at_end) % pipeline_stage_infos.size();
        if (!pinfo.copy_stage)
          pinfo.stage--;
      }
    }

    std::string optimizer_decision = "legacy_heuristic";
    std::vector<PipelineDependency> dependencies =
        BuildDependencies(pipeline_stage_infos);
    for (auto &pinfo : pipeline_stage_infos) {
      pinfo.tpu_pipeline_role = ClassifyTpuPipelineRole(&pinfo);
    }
    TpuPipelinePolicyDecision policy =
        DecideTpuPipelinePolicy(pipeline_stage_infos, num_stages);
    if (HasManualStageOrOrder(loop)) {
      optimizer_decision = "manual_constraints";
    } else if (policy.allow_stage_optimization) {
      if (!TryApplyTpuStageOptimizerSearch(num_stages, &pipeline_stage_infos,
                                           dependencies,
                                           &optimizer_decision)) {
        TryApplyTpuStageOptimizerV0(num_stages, &pipeline_stage_infos,
                                    &optimizer_decision);
      }
    } else if (IsTpuTarget()) {
      optimizer_decision =
          "legacy_heuristic:policy_stage_optimization_disabled:" +
          policy.kind;
      if (policy.kind == "mixed_gemm") {
        for (auto &pinfo : pipeline_stage_infos) {
          pinfo.order = pinfo.original_order;
          pinfo.stage = 0;
        }
      }
    }

    ApplyManualPipelineAnnotations(loop, &pipeline_stage_infos);
    dependencies = BuildDependencies(pipeline_stage_infos);
    std::unordered_map<std::string, int> buffer_version_needs =
        BuildBufferVersionNeeds(pipeline_stage_infos, dependencies);
    TpuOverlapWindowInfo selected_attention_window;
    bool has_selected_attention_window = TrySelectAttentionWindowBMaterializer(
        pipeline_stage_infos, dependencies, policy, &selected_attention_window);

    // Finally, make the pipeline annotation
    Map<String, ObjectRef> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages") {
        annotations.Set(key, value);
      }
    }

    std::vector<Integer> orders, stages;
    orders.reserve(pipeline_stage_infos.size());
    stages.reserve(pipeline_stage_infos.size());
    for (auto &pinfo : pipeline_stage_infos) {
      orders.push_back(pinfo.order);
      stages.push_back(pinfo.stage);
    }

    annotations.Set(tir::attr::software_pipeline_stage, Array<Integer>(stages));
    annotations.Set(tir::attr::software_pipeline_order, Array<Integer>(orders));
    if (policy.enable_buffer_version_hints) {
      Map<String, Integer> buffer_version_hints;
      for (const auto &kv : buffer_version_needs) {
        buffer_version_hints.Set(kv.first, Integer(kv.second));
      }
      annotations.Set("tl_pipeline_buffer_versions", buffer_version_hints);
    }
    bool has_async_copy = TargetHasAsyncCopy(target_);
    if (has_async_copy)
      annotations.Set(tir::attr::software_pipeline_async_stages,
                      Array<Integer>{0});
    if (has_selected_attention_window) {
      Map<String, Integer> window_annotation;
      window_annotation.Set("load_statement",
                            Integer(selected_attention_window.load_statement));
      window_annotation.Set(
          "overlap_start",
          Integer(selected_attention_window.overlap_statements.front()));
      window_annotation.Set(
          "overlap_end",
          Integer(selected_attention_window.overlap_statements.back()));
      window_annotation.Set(
          "consumer_statement",
          Integer(selected_attention_window.consumer_statement));
      annotations.Set("tl_tpu_attention_window_b", window_annotation);
    }
    bool has_tpu_parallel_guard_region =
        !has_selected_attention_window &&
        ShouldUseTpuParallelGuardRegion(pipeline_stage_infos, policy);
    if (has_tpu_parallel_guard_region) {
      annotations.Set("tl_tpu_parallel_guard_region", Integer(1));
    } else if (!has_selected_attention_window &&
               policy.policy_family == "attention_mixed" &&
               policy.evidence_tier == "tier_1_5_phase_model" &&
               EnvFlagEnabled("TL_TPU_ATTENTION_GUARD_REGION", false)) {
      annotations.Set("tl_tpu_attention_guard_region", Integer(1));
      has_tpu_parallel_guard_region = true;
    }

    DumpCertificate(loop, pipeline_stage_infos, num_stages, has_async_copy,
                    optimizer_decision, policy,
                    has_tpu_parallel_guard_region);

    return For(loop->loop_var, loop->min, loop->extent, loop->kind, loop->body,
               loop->thread_binding, annotations);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return std::move(block);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
  String global_symbol_ = "<unknown>";
  int loop_counter_ = 0;
};

tvm::transform::Pass PipelinePlanning() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = PipelinePlanner::Substitute(f);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PipelinePlanning", {});
}

TVM_REGISTER_GLOBAL("tl.transform.PipelinePlanning")
    .set_body_typed(PipelinePlanning);

} // namespace tl
} // namespace tvm
