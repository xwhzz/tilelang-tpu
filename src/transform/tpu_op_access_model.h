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
 * \file tpu_op_access_model.h
 * \brief TPU call_extern access roles shared by pipeline planning and address
 * assignment.
 */

#ifndef TVM_TL_TRANSFORM_TPU_OP_ACCESS_MODEL_H_
#define TVM_TL_TRANSFORM_TPU_OP_ACCESS_MODEL_H_

#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

enum class TpuBufferAccessKind {
  kRead,
  kWrite,
  kReadWrite,
  kConservative,
};

struct TpuOperandAccess {
  size_t arg_index;
  TpuBufferAccessKind access_kind;
};

struct TpuOpPhaseInfo {
  std::string name;
  std::string engine = "unknown";
  std::string role = "unknown";
  bool conservative = false;
  bool hidden_workspace = false;
  std::vector<size_t> read_args;
  std::vector<size_t> write_args;
  std::vector<size_t> scratch_args;
};

struct TpuCallExternAccessInfo {
  bool valid = false;
  std::string op_name;
  std::string engine = "unknown";
  std::string transaction = "";
  bool conservative = false;
  std::vector<TpuOperandAccess> operand_accesses;
  std::vector<TpuOpPhaseInfo> phases;
};

inline const char *TpuBufferAccessKindName(TpuBufferAccessKind kind) {
  switch (kind) {
  case TpuBufferAccessKind::kRead:
    return "read";
  case TpuBufferAccessKind::kWrite:
    return "write";
  case TpuBufferAccessKind::kReadWrite:
    return "readwrite";
  case TpuBufferAccessKind::kConservative:
    return "conservative";
  }
  return "conservative";
}

inline void AddOperandAccess(TpuCallExternAccessInfo *info, size_t index,
                             TpuBufferAccessKind access_kind) {
  info->operand_accesses.push_back({index, access_kind});
}

inline void AddTpuOpPhase(TpuCallExternAccessInfo *info, std::string name,
                          std::string engine, std::string role,
                          std::vector<size_t> read_args,
                          std::vector<size_t> write_args,
                          std::vector<size_t> scratch_args = {},
                          bool conservative = false,
                          bool hidden_workspace = false) {
  TpuOpPhaseInfo phase;
  phase.name = std::move(name);
  phase.engine = std::move(engine);
  phase.role = std::move(role);
  phase.read_args = std::move(read_args);
  phase.write_args = std::move(write_args);
  phase.scratch_args = std::move(scratch_args);
  phase.conservative = conservative;
  phase.hidden_workspace = hidden_workspace;
  info->phases.push_back(std::move(phase));
}

inline bool IsTpuElementwisePplOp(const std::string &op_name) {
  return op_name == "ppl.sub" || op_name == "ppl.mul" ||
         op_name == "ppl.add" || op_name == "ppl.div";
}

inline bool IsTpuElementwiseConstPplOp(const std::string &op_name) {
  return op_name == "ppl.mul_C" || op_name == "ppl.add_C" ||
         op_name == "ppl.rsqrt";
}

inline bool IsTpuReducePplOp(const std::string &op_name) {
  return op_name == "ppl.reduce_sum" || op_name == "ppl.reduce_max";
}

inline std::string InferTpuPplEngine(const std::string &op_name) {
  if (op_name == "ppl.copy") {
    return "dma_or_bdc";
  }
  if (op_name.rfind("ppl.", 0) == 0) {
    return "bdc";
  }
  return "unknown";
}

inline bool IsGlobalScope(const std::string &scope) { return scope == "global"; }

inline std::string ClassifyTpuPplCopyTransaction(const std::string &src_scope,
                                                 const std::string &dst_scope) {
  if (src_scope.empty() || dst_scope.empty()) {
    return "unknown";
  }
  bool src_global = IsGlobalScope(src_scope);
  bool dst_global = IsGlobalScope(dst_scope);
  if (src_global && !dst_global) {
    return "S2L";
  }
  if (!src_global && dst_global) {
    return "L2S";
  }
  if (!src_global && !dst_global) {
    return "local_to_local";
  }
  return "unsupported_global_to_global";
}

inline std::string TpuCopyPhaseEngine(const std::string &transaction) {
  if (transaction == "S2L" || transaction == "L2S") {
    return "gdma";
  }
  if (transaction == "local_to_local") {
    return "bdc_or_dma";
  }
  return "unknown";
}

inline std::string TpuCopyPhaseRole(const std::string &transaction) {
  if (transaction == "S2L") {
    return "load";
  }
  if (transaction == "L2S") {
    return "store";
  }
  if (transaction == "local_to_local") {
    return "local_copy";
  }
  return "copy_unknown";
}

inline void PopulateTpuCallExternPhases(TpuCallExternAccessInfo *info) {
  info->phases.clear();
  const std::string &op_name = info->op_name;
  if (op_name == "ppl.copy") {
    AddTpuOpPhase(info, "copy", TpuCopyPhaseEngine(info->transaction),
                  TpuCopyPhaseRole(info->transaction), {1}, {2}, {},
                  info->transaction.empty() || info->transaction == "unknown" ||
                      info->transaction == "unsupported_global_to_global");
  } else if (op_name == "ppl.fill") {
    AddTpuOpPhase(info, "fill", "bdc", "fill", {}, {1});
  } else if (op_name == "ppl.gemm") {
    AddTpuOpPhase(info, "gemm", "bdc", "gemm", {1, 2}, {3});
  } else if (IsTpuElementwisePplOp(op_name)) {
    AddTpuOpPhase(info, op_name.substr(4), "bdc", "elementwise", {2, 3},
                  {1});
  } else if (IsTpuElementwiseConstPplOp(op_name)) {
    AddTpuOpPhase(info, op_name.substr(4), "bdc", "elementwise", {2}, {1});
  } else if (op_name == "ppl.reduce_sum" || op_name == "ppl.reduce_max") {
    AddTpuOpPhase(info, "reduce_pad_optional", "bdc", "workspace_init", {1},
                  {3}, {3}, false, true);
    AddTpuOpPhase(info, "reduce_h", "bdc", "reduce", {1}, {3}, {3}, false,
                  true);
    AddTpuOpPhase(info, "reduce_w", "bdc", "reduce", {3}, {2}, {3}, false,
                  true);
  } else if (op_name == "ppl.exp") {
    AddTpuOpPhase(info, "exp_table_load", "bdc", "workspace_init", {},
                  {4, 5}, {4, 5}, false, true);
    AddTpuOpPhase(info, "exp_compute", "bdc", "elementwise", {1, 2, 3, 4, 5},
                  {1}, {2, 3, 4, 5}, true, true);
  } else if (op_name == "ppl.sigmoid") {
    AddTpuOpPhase(info, "sigmoid_table_load", "bdc", "workspace_init", {},
                  {5, 6}, {5, 6}, false, true);
    AddTpuOpPhase(info, "sigmoid_compute", "bdc", "elementwise",
                  {2, 3, 4, 5, 6}, {1}, {3, 4, 5, 6}, true, true);
  } else if (op_name == "ppl.gather") {
    AddTpuOpPhase(info, "gather", "gdma", "gather", {2, 3}, {1}, {}, true);
  } else if (op_name == "ppl.topk") {
    AddTpuOpPhase(info, "topk", "bdc", "sort", {3}, {1, 2}, {}, true);
  } else if (op_name == "ppl.rope_add") {
    AddTpuOpPhase(info, "rope_add", "bdc", "elementwise", {2, 3, 4, 5}, {1});
  } else {
    AddTpuOpPhase(info, "unknown", "unknown", "unknown", {}, {}, {}, true);
  }
}

inline TpuCallExternAccessInfo GetTpuCallExternAccessInfo(const CallNode *op) {
  TpuCallExternAccessInfo info;
  if (!op->op.same_as(builtin::call_extern()) || op->args.empty()) {
    return info;
  }
  auto *op_name_node = op->args[0].as<StringImmNode>();
  if (!op_name_node) {
    return info;
  }
  info.valid = true;
  info.op_name = op_name_node->value;
  info.engine = InferTpuPplEngine(info.op_name);

  if (info.op_name == "ppl.copy") {
    AddOperandAccess(&info, 1, TpuBufferAccessKind::kRead);
    AddOperandAccess(&info, 2, TpuBufferAccessKind::kWrite);
  } else if (info.op_name == "ppl.fill") {
    AddOperandAccess(&info, 1, TpuBufferAccessKind::kWrite);
  } else if (info.op_name == "ppl.gemm") {
    AddOperandAccess(&info, 1, TpuBufferAccessKind::kRead);
    AddOperandAccess(&info, 2, TpuBufferAccessKind::kRead);
    AddOperandAccess(&info, 3, TpuBufferAccessKind::kWrite);
  } else if (IsTpuElementwisePplOp(info.op_name)) {
    AddOperandAccess(&info, 1, TpuBufferAccessKind::kWrite);
    AddOperandAccess(&info, 2, TpuBufferAccessKind::kRead);
    AddOperandAccess(&info, 3, TpuBufferAccessKind::kRead);
  } else if (IsTpuElementwiseConstPplOp(info.op_name)) {
    AddOperandAccess(&info, 1, TpuBufferAccessKind::kWrite);
    AddOperandAccess(&info, 2, TpuBufferAccessKind::kRead);
  } else if (IsTpuReducePplOp(info.op_name)) {
    AddOperandAccess(&info, 1, TpuBufferAccessKind::kRead);
    AddOperandAccess(&info, 2, TpuBufferAccessKind::kWrite);
    AddOperandAccess(&info, 3, TpuBufferAccessKind::kReadWrite);
  } else if (info.op_name == "ppl.exp") {
    info.conservative = true;
    for (size_t i = 1; i <= 5; ++i) {
      AddOperandAccess(&info, i, TpuBufferAccessKind::kConservative);
    }
  } else if (info.op_name == "ppl.sigmoid") {
    info.conservative = true;
    for (size_t i = 1; i <= 6; ++i) {
      AddOperandAccess(&info, i, TpuBufferAccessKind::kConservative);
    }
  } else if (info.op_name == "ppl.gather") {
    AddOperandAccess(&info, 1, TpuBufferAccessKind::kWrite);
    AddOperandAccess(&info, 2, TpuBufferAccessKind::kRead);
    AddOperandAccess(&info, 3, TpuBufferAccessKind::kRead);
  } else if (info.op_name == "ppl.topk") {
    AddOperandAccess(&info, 1, TpuBufferAccessKind::kWrite);
    AddOperandAccess(&info, 2, TpuBufferAccessKind::kWrite);
    AddOperandAccess(&info, 3, TpuBufferAccessKind::kRead);
  } else if (info.op_name == "ppl.rope_add") {
    AddOperandAccess(&info, 1, TpuBufferAccessKind::kWrite);
    AddOperandAccess(&info, 2, TpuBufferAccessKind::kRead);
    AddOperandAccess(&info, 3, TpuBufferAccessKind::kRead);
    AddOperandAccess(&info, 4, TpuBufferAccessKind::kRead);
    AddOperandAccess(&info, 5, TpuBufferAccessKind::kRead);
  } else {
    info.conservative = true;
    for (size_t i = 1; i < op->args.size(); ++i) {
      AddOperandAccess(&info, i, TpuBufferAccessKind::kConservative);
    }
  }
  PopulateTpuCallExternPhases(&info);
  return info;
}

inline std::string SizeTVectorToString(const std::vector<size_t> &values) {
  std::string result = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      result += ",";
    }
    result += std::to_string(values[i]);
  }
  result += "]";
  return result;
}

inline std::string TpuOpPhaseInfoToString(const TpuOpPhaseInfo &phase) {
  std::string result = phase.name + "{engine=" + phase.engine +
                       ",role=" + phase.role;
  if (phase.conservative) {
    result += ",conservative=true";
  }
  if (phase.hidden_workspace) {
    result += ",hidden_workspace=true";
  }
  result += ",reads=" + SizeTVectorToString(phase.read_args);
  result += ",writes=" + SizeTVectorToString(phase.write_args);
  result += ",scratch=" + SizeTVectorToString(phase.scratch_args);
  result += "}";
  return result;
}

inline std::string TpuOpPhasesToString(
    const std::vector<TpuOpPhaseInfo> &phases) {
  std::string result = "[";
  for (size_t i = 0; i < phases.size(); ++i) {
    if (i != 0) {
      result += ",";
    }
    result += TpuOpPhaseInfoToString(phases[i]);
  }
  result += "]";
  return result;
}

inline std::string TpuCallExternAccessInfoToString(
    const TpuCallExternAccessInfo &info) {
  if (!info.valid) {
    return "unknown";
  }
  std::string result = info.op_name + "{engine=" + info.engine;
  if (!info.transaction.empty()) {
    result += ",transaction=" + info.transaction;
  }
  if (info.conservative) {
    result += ",conservative=true";
  }
  result += ",args=[";
  for (size_t i = 0; i < info.operand_accesses.size(); ++i) {
    if (i != 0) {
      result += ",";
    }
    const auto &access = info.operand_accesses[i];
    result += std::to_string(access.arg_index);
    result += ":";
    result += TpuBufferAccessKindName(access.access_kind);
  }
  result += "]}";
  result += ",phases=" + TpuOpPhasesToString(info.phases);
  return result;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_TPU_OP_ACCESS_MODEL_H_
