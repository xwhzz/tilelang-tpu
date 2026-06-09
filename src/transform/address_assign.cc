/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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
 * \file storage_rewrite.cc
 * \brief Memory access pattern analysis and optimization.
 *  Re-write data access to enable memory sharing when possible.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ir/type.h>
#include <tvm/node/repr_printer.h>
#include <tvm/relay/expr.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

// #include <map>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "../op/bulk_copy.h"
#include "../op/gemm.h"
#include "../target/bm1690_lmem.h"
#include "tpu_op_access_model.h"

namespace tvm {
namespace tl {
using namespace tir;

namespace {

int64_t AlignUp(int64_t value, int64_t align) {
  return bm1690::AlignUp(value, align);
}

std::string GetAddressAssignDumpSink() {
  const char *env = std::getenv("TL_TPU_ADDRESS_ASSIGN_DUMP");
  if (env == nullptr) {
    return "";
  }
  std::string sink(env);
  if (sink.empty() || sink == "0" || sink == "false" || sink == "False") {
    return "";
  }
  return sink;
}

void EmitAddressAssignDump(const std::string &text) {
  std::string sink = GetAddressAssignDumpSink();
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

std::string ShapeToString(const Array<PrimExpr> &shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << AsLegacyRepr(shape[i]);
  }
  os << "]";
  return os.str();
}

bool StageAwareLmemEnabled() {
  const char *env = std::getenv("TL_TPU_STAGE_AWARE_LMEM");
  if (env == nullptr) {
    return true;
  }
  std::string value(env);
  return !(value.empty() || value == "0" || value == "false" ||
           value == "False");
}

bool IsTpuPrimFunc(const PrimFunc &f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  return target.defined() && target.value()->kind.defined() &&
         target.value()->kind->name == "tpu";
}

bool IsSimpleGemmCompanionOp(const std::string &op_name) {
  return op_name == "ppl.copy" || op_name == "ppl.fill" ||
         op_name == "ppl.gemm";
}

struct StageAwareLmemPolicy {
  std::string kind = "non_tpu";
  std::string reason = "not_tpu_target";
  bool enabled = false;
};

class TpuFunctionOpPolicyCollector : public StmtExprVisitor {
public:
  void Analyze(const Stmt &stmt) { VisitStmt(stmt); }

  void VisitExpr_(const CallNode *op) final {
    TpuCallExternAccessInfo info = GetTpuCallExternAccessInfo(op);
    if (info.valid && info.op_name.rfind("ppl.", 0) == 0) {
      if (info.op_name == "ppl.gemm") {
        has_gemm = true;
      }
      if (!IsSimpleGemmCompanionOp(info.op_name)) {
        has_mixed_compute = true;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "tpu_parallel_start") {
      has_tpu_parallel_region = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool has_gemm = false;
  bool has_mixed_compute = false;
  bool has_tpu_parallel_region = false;
};

StageAwareLmemPolicy DecideStageAwareLmemPolicy(const PrimFunc &f) {
  StageAwareLmemPolicy policy;
  if (!IsTpuPrimFunc(f)) {
    return policy;
  }
  if (!StageAwareLmemEnabled()) {
    policy.kind = "disabled_by_env";
    policy.reason = "TL_TPU_STAGE_AWARE_LMEM=0";
    policy.enabled = false;
    return policy;
  }

  TpuFunctionOpPolicyCollector collector;
  collector.Analyze(f->body);
  if (!collector.has_tpu_parallel_region) {
    policy.kind = "not_pipeline_materialized";
    policy.reason = "no_tpu_parallel_region";
    policy.enabled = false;
    return policy;
  }
  if (collector.has_gemm && !collector.has_mixed_compute) {
    policy.kind = "gemm_heavy";
    policy.reason = "pure_gemm_pipeline_function";
    policy.enabled = true;
    return policy;
  }
  if (collector.has_gemm && collector.has_mixed_compute) {
    policy.kind = "mixed_gemm";
    policy.reason = "gemm_function_with_elementwise_or_reduction_ops";
    policy.enabled = false;
    return policy;
  }
  policy.kind = "elementwise_or_generic";
  policy.reason = "no_pure_gemm_pipeline_function";
  policy.enabled = false;
  return policy;
}

struct PipelineVersionName {
  bool valid = false;
  std::string base;
  int version = -1;
};

PipelineVersionName ParsePipelineVersionName(const std::string &name) {
  size_t underscore = name.rfind('_');
  if (underscore == std::string::npos || underscore + 1 >= name.size()) {
    return {};
  }
  for (size_t i = underscore + 1; i < name.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
      return {};
    }
  }
  PipelineVersionName result;
  result.valid = true;
  result.base = name.substr(0, underscore);
  result.version = std::stoi(name.substr(underscore + 1));
  return result;
}

std::string PipelineVersionGroupKey(const BufferNode *buffer,
                                    const PipelineVersionName &version) {
  std::ostringstream os;
  os << version.base << "|" << buffer->dtype << "|"
     << ShapeToString(buffer->shape);
  return os.str();
}

std::unordered_map<const BufferNode *, int> AddPipelineVersionSiblingConflicts(
    const std::vector<const BufferNode *> &alloc_ops,
    std::unordered_map<const BufferNode *, std::unordered_set<const BufferNode *>>
        *conflict_map) {
  std::unordered_map<std::string, std::vector<const BufferNode *>> groups;
  std::unordered_map<const BufferNode *, int> sibling_degree;
  for (const BufferNode *op : alloc_ops) {
    PipelineVersionName version =
        ParsePipelineVersionName(std::string(op->name));
    if (!version.valid) {
      sibling_degree[op] = 0;
      continue;
    }
    groups[PipelineVersionGroupKey(op, version)].push_back(op);
  }

  for (const auto &kv : groups) {
    const std::vector<const BufferNode *> &siblings = kv.second;
    if (siblings.size() < 2) {
      continue;
    }
    for (const BufferNode *lhs : siblings) {
      sibling_degree[lhs] = static_cast<int>(siblings.size()) - 1;
      for (const BufferNode *rhs : siblings) {
        if (lhs != rhs) {
          (*conflict_map)[lhs].insert(rhs);
        }
      }
    }
  }
  return sibling_degree;
}

} // namespace

class AddressAllocator : public StmtExprVisitor {
public:
  explicit AddressAllocator() {}

  std::vector<const BufferNode *> collectAllocOp(tir::Stmt body) {
    this->VisitStmt(body);
    return alloc_ops_;
  }

  void VisitStmt_(const DeclBufferNode *op) {
    alloc_ops_.emplace_back(op->buffer.get());
    this->VisitStmt(op->body);
  }

private:
  std::vector<const BufferNode *> alloc_ops_;
};

struct TensorLive {
  uint32_t start = 0;
  uint32_t end = 0;
  int64_t tensor_size = 0;
};

struct OpAddr {
  const BufferNode *op;
  int64_t start = 0;
  int64_t end = 0;
  int64_t size = 0;
  uint32_t first_pos = 0;
  uint32_t end_pos = 0;

  OpAddr(const BufferNode *_op, int64_t _size, uint32_t _first_pos,
         uint32_t _end_pos) {
    op = _op;
    size = _size;
    first_pos = _first_pos;
    end_pos = _end_pos;
  }
};

bool LiveRangesOverlap(const TensorLive &lhs, const TensorLive &rhs) {
  return std::max(lhs.start, rhs.start) < std::min(lhs.end, rhs.end);
}

bool LiveRangesOverlap(const OpAddr &lhs, const OpAddr &rhs) {
  return std::max(lhs.first_pos, rhs.first_pos) <
         std::min(lhs.end_pos, rhs.end_pos);
}

class MemAllocBankConflictAware {
public:
  MemAllocBankConflictAware(int64_t bank_num, int64_t bank_size)
      : bank_num_(bank_num), bank_size_(bank_size) {
    total_consumption_ = 0;
    mem_size_ = bank_num * bank_size;
    bank_ops.resize(bank_num);
  }

  bool assignAddr(
      std::vector<const BufferNode *> &ops,
      std::unordered_map<const BufferNode *, TensorLive> &liveRange,
      std::unordered_map<const BufferNode *,
                         std::unordered_set<const BufferNode *>> &conflictMap,
      std::unordered_map<const BufferNode *, int64_t> &addrMap,
      int64_t &totalSize) {
    std::list<const BufferNode *> op_list;
    std::copy(ops.begin(), ops.end(), std::back_inserter(op_list));

    op_list.sort([&liveRange, &conflictMap](const BufferNode *a,
                                             const BufferNode *b) {
      auto &lhs = liveRange[a];
      auto &rhs = liveRange[b];
      size_t lhs_degree = conflictMap[a].size();
      size_t rhs_degree = conflictMap[b].size();
      if (lhs_degree != rhs_degree) {
        return lhs_degree > rhs_degree;
      }
      if (lhs.tensor_size != rhs.tensor_size) {
        return lhs.tensor_size > rhs.tensor_size;
      }
      return lhs.start < rhs.start;
    });
    for (auto &op : op_list) {
      std::shared_ptr<OpAddr> best_addr;
      int64_t min_conflict_count = std::numeric_limits<int64_t>::max();
      if (liveRange[op].tensor_size > mem_size_) {
        return false;
      }
      int64_t bytes = liveRange[op].tensor_size;
      int64_t mem_cross_bank_num =
          static_cast<int64_t>(bm1690::DivUp(bytes, bank_size_));
      mem_cross_bank_num = std::max<int64_t>(mem_cross_bank_num, 1);
      for (int i = 0; i < bank_num_; ++i) {
        int64_t offset = i * bank_size_;

        if (i + mem_cross_bank_num > bank_num_) {
          break;
        }
        int64_t end_offset = std::min(
            offset + (mem_cross_bank_num + 1) * bank_size_, mem_size_);
        auto op_addr = searchAddr(op, liveRange, offset, end_offset);

        // op can insert
        if (op_addr->start + op_addr->size <= end_offset) {
          int64_t conf_count =
              getConflictCount(op_addr, liveRange, conflictMap);
          bool better_addr =
              !best_addr || conf_count < min_conflict_count ||
              (conf_count == min_conflict_count &&
               (op_addr->end < best_addr->end ||
                (op_addr->end == best_addr->end &&
                 op_addr->start < best_addr->start)));
          if (better_addr) {
            min_conflict_count = conf_count;
            best_addr = op_addr;
          }
        }
      }
      if (!best_addr) {
        return false;
      }
      insertAddr(best_addr);
      // addrMap.Set(op, best_addr->start);
      addrMap[op] = best_addr->start;
    }
    totalSize = total_consumption_;
    return true;
  }

protected:
  void insertAddr(std::shared_ptr<OpAddr> &opAddr) {
    total_consumption_ = std::max(total_consumption_, opAddr->end);
    auto iter =
        std::find_if(allocated_op_list_.begin(), allocated_op_list_.end(),
                     [&opAddr](std::shared_ptr<OpAddr> &p) {
                       return p->start >= opAddr->start;
                     });
    allocated_op_list_.emplace(iter, opAddr);
    int64_t bank_start = opAddr->start / bank_size_;
    int64_t bank_end = (opAddr->end - 1) / bank_size_;
    for (int i = bank_start; i <= bank_end; i++) {
      bank_ops[i].push_back(opAddr->op);
    }
  }

  int64_t getConflictCount(
      std::shared_ptr<OpAddr> &opAddr,
      std::unordered_map<const BufferNode *, TensorLive> &liveRange,
      std::unordered_map<const BufferNode *,
                         std::unordered_set<const BufferNode *>> &conflictMap) {
    int64_t bank_start = opAddr->start / bank_size_;
    int64_t bank_end = (opAddr->end - 1) / bank_size_;
    int count = 0;
    for (int i = bank_start; i <= bank_end; i++) {
      for (auto op : bank_ops[i]) {
        if (conflictMap[opAddr->op].count(op) &&
            LiveRangesOverlap(liveRange[opAddr->op], liveRange[op])) {
          ++count;
        }
      }
    }
    return count;
  }

  std::shared_ptr<OpAddr>
  searchAddr(const BufferNode *op,
             std::unordered_map<const BufferNode *, TensorLive> &liveRange,
             int64_t offset, int64_t end_offset) {

    std::shared_ptr<OpAddr> op_addr = std::make_shared<OpAddr>(
        op, liveRange[op].tensor_size, liveRange[op].start, liveRange[op].end);
    int64_t prev_offset = AlignUp(offset, bm1690::kTensorAlignBytes);
    int64_t best_offset = -1;
    int64_t smallest_gap = std::numeric_limits<int64_t>::max();

    for (auto &allocated_op_addr : allocated_op_list_) {
      if (allocated_op_addr->end <= offset) {
        continue;
      }
      if (allocated_op_addr->start >= end_offset) {
        break;
      }
      if (LiveRangesOverlap(*op_addr, *allocated_op_addr)) {
        int64_t candidate =
            AlignUp(prev_offset, bm1690::kTensorAlignBytes);
        int64_t gap = allocated_op_addr->start - candidate;
        if (gap >= op_addr->size && gap < smallest_gap) {
          smallest_gap = gap;
          best_offset = candidate;
        }
        prev_offset = std::max(prev_offset, allocated_op_addr->end);
      }
    }
    int64_t trailing_candidate =
        AlignUp(prev_offset, bm1690::kTensorAlignBytes);
    int64_t trailing_gap = end_offset - trailing_candidate;
    if (trailing_gap >= op_addr->size && trailing_gap < smallest_gap) {
      best_offset = trailing_candidate;
    } else if (best_offset == -1) {
      best_offset = trailing_candidate;
    }
    op_addr->start = best_offset;
    op_addr->end = op_addr->start + op_addr->size;
    return op_addr;
  }

protected:
  std::list<std::shared_ptr<OpAddr>> allocated_op_list_;
  std::vector<std::vector<const BufferNode *>> bank_ops;
  int64_t total_consumption_;
  int64_t bank_num_;
  int64_t bank_size_;
  int64_t mem_size_;
};

using BufferAccessKind = TpuBufferAccessKind;

class BufferUseCollector : public StmtExprVisitor {
public:
  explicit BufferUseCollector(
      const std::vector<const BufferNode *> &alloc_ops,
      std::unordered_map<const BufferNode *, TensorLive> *live_ranges,
      std::unordered_map<const BufferNode *,
                         std::unordered_set<const BufferNode *>> *conflict_map)
      : live_ranges_(live_ranges), conflict_map_(conflict_map) {
    for (const BufferNode *buffer : alloc_ops) {
      buffer_var_to_buffer_[buffer->data.get()] = buffer;
      (*conflict_map_)[buffer];
    }
  }

  void Analyze(const Stmt &body) {
    VisitStmt(body);
    for (auto &kv : *live_ranges_) {
      if (!seen_buffers_.count(kv.first)) {
        kv.second.start = 0;
        kv.second.end = 0;
      }
    }
  }

private:
  struct LoopUseContext {
    uint32_t start_loc = 0;
    std::unordered_set<const BufferNode *> buffers;
  };

  class OpScope {
  public:
    explicit OpScope(BufferUseCollector *collector) : collector_(collector) {
      if (!collector_->inside_op_) {
        started_ = true;
        collector_->inside_op_ = true;
        collector_->current_loc_ = collector_->NextLoc();
      }
    }

    ~OpScope() {
      if (started_) {
        collector_->FinishCurrentOp();
        collector_->inside_op_ = false;
      }
    }

  private:
    BufferUseCollector *collector_;
    bool started_ = false;
  };

  uint32_t NextLoc() { return ++loc_; }

  void FinishCurrentOp() {
    if (!conservative_buffers_.empty()) {
      for (auto buffer : read_buffers_) {
        conservative_buffers_.insert(buffer);
      }
      for (auto buffer : write_buffers_) {
        conservative_buffers_.insert(buffer);
      }
      for (auto lhs : conservative_buffers_) {
        for (auto rhs : conservative_buffers_) {
          if (lhs != rhs) {
            (*conflict_map_)[lhs].insert(rhs);
          }
        }
      }
    } else {
      for (auto lhs : read_buffers_) {
        for (auto rhs : read_buffers_) {
          if (lhs != rhs) {
            (*conflict_map_)[lhs].insert(rhs);
          }
        }
      }
    }
    read_buffers_.clear();
    write_buffers_.clear();
    conservative_buffers_.clear();
  }

  void MarkUse(const BufferNode *buffer, BufferAccessKind access_kind) {
    auto it = live_ranges_->find(buffer);
    if (it == live_ranges_->end()) {
      return;
    }
    for (auto &loop_ctx : loop_use_stack_) {
      loop_ctx.buffers.insert(buffer);
    }
    if (inside_tpu_parallel_region_) {
      parallel_region_buffers_.insert(buffer);
    }
    if (!inside_op_) {
      current_loc_ = NextLoc();
    } else {
      switch (access_kind) {
      case BufferAccessKind::kRead:
        read_buffers_.insert(buffer);
        break;
      case BufferAccessKind::kWrite:
        write_buffers_.insert(buffer);
        break;
      case BufferAccessKind::kReadWrite:
        read_buffers_.insert(buffer);
        write_buffers_.insert(buffer);
        break;
      case BufferAccessKind::kConservative:
        conservative_buffers_.insert(buffer);
        break;
      }
    }
    auto &range = it->second;
    if (!seen_buffers_.count(buffer)) {
      range.start = current_loc_;
      range.end = current_loc_ + 1;
      seen_buffers_.insert(buffer);
    } else {
      range.start = std::min<uint32_t>(range.start, current_loc_);
      range.end = std::max<uint32_t>(range.end, current_loc_ + 1);
    }
  }

  void VisitStmt_(const ForNode *op) final {
    LoopUseContext loop_ctx;
    loop_ctx.start_loc = loc_ + 1;
    loop_use_stack_.push_back(std::move(loop_ctx));
    StmtExprVisitor::VisitStmt_(op);
    LoopUseContext finished_loop = std::move(loop_use_stack_.back());
    loop_use_stack_.pop_back();

    uint32_t loop_end = loc_ + 1;
    for (const BufferNode *buffer : finished_loop.buffers) {
      auto it = live_ranges_->find(buffer);
      if (it == live_ranges_->end()) {
        continue;
      }
      if (it->second.start < finished_loop.start_loc) {
        it->second.end = std::max<uint32_t>(it->second.end, loop_end);
      }
    }
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "tpu_parallel_start") {
      ICHECK(!inside_tpu_parallel_region_)
          << "Nested tpu_parallel_start is not supported by AddressAssign";
      inside_tpu_parallel_region_ = true;
      parallel_region_start_loc_ = NextLoc();
      parallel_region_buffers_.clear();
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    if (op->attr_key == "tpu_parallel_end") {
      uint32_t end_loc = NextLoc() + 1;
      for (const BufferNode *buffer : parallel_region_buffers_) {
        auto it = live_ranges_->find(buffer);
        if (it == live_ranges_->end()) {
          continue;
        }
        it->second.start =
            std::min<uint32_t>(it->second.start, parallel_region_start_loc_);
        it->second.end = std::max<uint32_t>(it->second.end, end_loc);
      }
      inside_tpu_parallel_region_ = false;
      parallel_region_buffers_.clear();
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void MarkExprAs(const PrimExpr &expr, BufferAccessKind access_kind) {
    bool previous_collecting = collecting_operand_;
    BufferAccessKind previous_access = operand_access_kind_;
    collecting_operand_ = true;
    operand_access_kind_ = access_kind;
    VisitExpr(expr);
    collecting_operand_ = previous_collecting;
    operand_access_kind_ = previous_access;
  }

  bool IsAccessPtrCall(const CallNode *op) const {
    return op->op.same_as(builtin::tvm_access_ptr());
  }

  const BufferNode *BufferFromAccessPtr(const CallNode *op) const {
    if (!IsAccessPtrCall(op) || op->args.size() < 2) {
      return nullptr;
    }
    auto *var = op->args[1].as<VarNode>();
    if (!var) {
      return nullptr;
    }
    auto it = buffer_var_to_buffer_.find(var);
    if (it == buffer_var_to_buffer_.end()) {
      return nullptr;
    }
    return it->second;
  }

  BufferAccessKind AccessKindFromAccessPtr(const CallNode *op) const {
    if (op->args.size() < 5) {
      return BufferAccessKind::kConservative;
    }
    auto *mask = op->args[4].as<IntImmNode>();
    if (!mask) {
      return BufferAccessKind::kConservative;
    }
    bool read = (mask->value & 1) != 0;
    bool write = (mask->value & 2) != 0;
    if (read && write) {
      return BufferAccessKind::kReadWrite;
    }
    if (read) {
      return BufferAccessKind::kRead;
    }
    if (write) {
      return BufferAccessKind::kWrite;
    }
    return BufferAccessKind::kConservative;
  }

  void VisitAccessPtrCall(const CallNode *op, BufferAccessKind access_kind) {
    const BufferNode *buffer = BufferFromAccessPtr(op);
    if (buffer) {
      MarkUse(buffer, access_kind);
    }
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (i == 1 || i == 4) {
        continue;
      }
      VisitExpr(op->args[i]);
    }
  }

  bool VisitKnownPPLExtern(const CallNode *op) {
    TpuCallExternAccessInfo info = GetTpuCallExternAccessInfo(op);
    if (!info.valid) {
      return false;
    }
    auto mark_arg = [&](size_t index, BufferAccessKind access_kind) {
      if (index < op->args.size()) {
        MarkExprAs(op->args[index], access_kind);
      }
    };

    OpScope scope(this);
    for (const auto &operand_access : info.operand_accesses) {
      mark_arg(operand_access.arg_index, operand_access.access_kind);
    }
    return true;
  }

  void VisitExpr_(const VarNode *op) {
    auto it = buffer_var_to_buffer_.find(op);
    if (it != buffer_var_to_buffer_.end()) {
      MarkUse(it->second, collecting_operand_ ? operand_access_kind_
                                               : BufferAccessKind::kConservative);
    }
  }

  void VisitExpr_(const CallNode *op) {
    if (collecting_operand_) {
      if (IsAccessPtrCall(op)) {
        VisitAccessPtrCall(op, operand_access_kind_);
        return;
      }
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    if (VisitKnownPPLExtern(op)) {
      return;
    }
    if (IsAccessPtrCall(op)) {
      VisitAccessPtrCall(op, AccessKindFromAccessPtr(op));
      return;
    }
    OpScope scope(this);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) {
    OpScope scope(this);
    MarkUse(op->buffer.get(), BufferAccessKind::kRead);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) {
    OpScope scope(this);
    MarkUse(op->buffer.get(), BufferAccessKind::kWrite);
    StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_map<const VarNode *, const BufferNode *> buffer_var_to_buffer_;
  std::unordered_set<const BufferNode *> seen_buffers_;
  std::unordered_set<const BufferNode *> read_buffers_;
  std::unordered_set<const BufferNode *> write_buffers_;
  std::unordered_set<const BufferNode *> conservative_buffers_;
  std::unordered_map<const BufferNode *, TensorLive> *live_ranges_;
  std::unordered_map<const BufferNode *,
                     std::unordered_set<const BufferNode *>> *conflict_map_;
  uint32_t loc_ = 0;
  uint32_t current_loc_ = 0;
  bool inside_op_ = false;
  bool collecting_operand_ = false;
  BufferAccessKind operand_access_kind_ = BufferAccessKind::kConservative;
  bool inside_tpu_parallel_region_ = false;
  uint32_t parallel_region_start_loc_ = 0;
  std::unordered_set<const BufferNode *> parallel_region_buffers_;
  std::vector<LoopUseContext> loop_use_stack_;
};

PrimFunc InferAddress(PrimFunc f) {
  int bank_num = bm1690::kBankNum;
  int bank_size = bm1690::kBankSize;
  std::unordered_map<const BufferNode *, std::unordered_set<const BufferNode *>>
      bank_conflict_map;
  std::unordered_map<const BufferNode *, TensorLive> live_ranges;
  std::vector<const BufferNode *> alloc_ops =
      AddressAllocator().collectAllocOp(f->body);

  for (auto &op : alloc_ops) {
    TensorLive live;
    live.tensor_size =
        bm1690::TpuAlignSizeBytes(op->shape, op->dtype, "AddressAssign");
    live_ranges[op] = live;
  }
  BufferUseCollector(alloc_ops, &live_ranges, &bank_conflict_map)
      .Analyze(f->body);
  StageAwareLmemPolicy stage_aware_lmem_policy =
      DecideStageAwareLmemPolicy(f);
  bool stage_aware_lmem = stage_aware_lmem_policy.enabled;
  std::unordered_map<const BufferNode *, int> stage_sibling_degree;
  if (stage_aware_lmem) {
    stage_sibling_degree =
        AddPipelineVersionSiblingConflicts(alloc_ops, &bank_conflict_map);
  }

  std::unordered_map<const BufferNode *, int64_t> addrMapWithBC;
  int64_t memUsedWithBC = 0;
  MemAllocBankConflictAware allocatorBC(bank_num, bank_size);
  auto success = allocatorBC.assignAddr(
      alloc_ops, live_ranges, bank_conflict_map, addrMapWithBC, memUsedWithBC);
  ICHECK(success) << "BM1690 local memory allocation failed. buffers="
                  << alloc_ops.size() << ", lmem=" << bank_num * bank_size
                  << " bytes";

  if (success) {
    // std::unordered_map<String, PrimExpr> result;
    auto fn = f.CopyOnWrite();
    auto fn_attr = fn->attrs.CopyOnWrite();
    for (auto op : alloc_ops) {
      int64_t address = addrMapWithBC[op];
      fn_attr->dict.Set(op->name, IntImm(DataType::Int(64), address));
    }
  }
  if (!GetAddressAssignDumpSink().empty()) {
    std::ostringstream os;
    os << "ADDRESS_ASSIGN_DUMP_BEGIN\n";
    os << "bank_num: " << bank_num << "\n";
    os << "bank_size: " << bank_size << "\n";
    os << "total: " << memUsedWithBC << "\n";
    os << "stage_aware_lmem: "
       << (stage_aware_lmem ? "enabled" : "disabled") << "\n";
    os << "stage_aware_lmem_policy: " << stage_aware_lmem_policy.kind
       << "\n";
    os << "stage_aware_lmem_reason: " << stage_aware_lmem_policy.reason
       << "\n";
    os << "buffers:\n";
    for (auto op : alloc_ops) {
      const TensorLive &live = live_ranges[op];
      int64_t address = addrMapWithBC[op];
      int64_t size = live.tensor_size;
      int64_t bank_start = address / bank_size;
      int64_t bank_end = size > 0 ? (address + size - 1) / bank_size
                                  : bank_start;
      os << "  - name: " << op->name << "\n";
      os << "    dtype: " << op->dtype << "\n";
      os << "    shape: " << ShapeToString(op->shape) << "\n";
      os << "    size: " << size << "\n";
      os << "    live_range: [" << live.start << ", " << live.end << ")\n";
      os << "    addr: " << address << "\n";
      os << "    bank_range: [" << bank_start << ", " << bank_end << "]\n";
      os << "    conflict_degree: " << bank_conflict_map[op].size() << "\n";
      PipelineVersionName version =
          ParsePipelineVersionName(std::string(op->name));
      if (version.valid) {
        os << "    pipeline_base: " << version.base << "\n";
        os << "    pipeline_version: " << version.version << "\n";
        os << "    stage_sibling_conflict_degree: "
           << stage_sibling_degree[op] << "\n";
      }
    }
    os << "ADDRESS_ASSIGN_DUMP_END\n";
    EmitAddressAssignDump(os.str());
  }
  return f;
}

tvm::transform::Pass AddressAssign() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return InferAddress(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AddressAssign", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AddressAssign").set_body_typed(AddressAssign);

} // namespace tl
} // namespace tvm
