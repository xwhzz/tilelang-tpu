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
 * \file target/codegen.cc
 */

#include "codegen_ppl.h"
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "bm1690_lmem.h"
#include "../op/builtin.h"
#include "../op/bulk_copy.h"
#include "../op/gemm.h"
// #include "../../target/source/ptx.h"

namespace tvm {
namespace codegen {

CodeGenTileLangPPL::CodeGenTileLangPPL() {
  restrict_keyword_ = "global_addr_t";
}

void CodeGenTileLangPPL::PrintFuncPrefix(
    std::ostream &os) { /*os << "extern \"C\" __global__ "; */
}

class LaunchConfigExtractor : public tir::StmtVisitor {
private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x" ||
          iv->thread_tag == "threadIdx.x") {
        threadIdx_x_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.y" ||
                 iv->thread_tag == "threadIdx.y") {
        threadIdx_y_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.z" ||
                 iv->thread_tag == "threadIdx.z") {
        threadIdx_z_ext = op->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

public:
  PrimExpr threadIdx_x_ext = Integer(1);
  PrimExpr threadIdx_y_ext = Integer(1);
  PrimExpr threadIdx_z_ext = Integer(1);
};

void CodeGenTileLangPPL::PrintExtraAttrs(const PrimFunc &f, std::ostream &os) {}

std::string CodeGenTileLangPPL::Finish() {
  decl_stream << "#include \"ppl_helper.h\"\n";
  decl_stream << "#ifndef __PPL_HELPER__\n";
  decl_stream << "static data_type_t __ppl_get_dtype(int type) {\n"
              << "  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  "
                 "DT_BFP16,\n"
              << "    DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,\n"
              << "    DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,\n"
              << "    DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};\n"
              << "  return __dtype[type];\n"
              << "}\n"
              << "#endif  // __PPL_HELPER__\n\n";
  decl_stream << "typedef struct {\n"
              << "    dim4 shape;\n"
              << "    dim4 stride;\n"
              << "    global_addr_t addr;\n"
              << "    data_type_t dtype;\n"
              << "    int mode;\n"
              << "    int align_mode;\n"
              << "    int size;\n"
              << "    int offset;\n"
              << "    bool unsigned_flag;\n"
              << "    bool default_stride;\n"
              << "} __ppl_tensor_info;\n\n"
              << "static inline int __ppl_div_up_i32(int x, int y) { return (x + y - 1) / y; }\n"
              << "static inline int __ppl_align_up_i32(int x, int y) { return __ppl_div_up_i32(x, y) * y; }\n"
              << "static inline int __ppl_dtype_bytes_for_stride(data_type_t dtype) {\n"
              << "  if (dtype == DT_FP32 || dtype == DT_TF32 || dtype == DT_INT32 || dtype == DT_UINT32) return 4;\n"
              << "  if (dtype == DT_FP16 || dtype == DT_BFP16 || dtype == DT_INT16 || dtype == DT_UINT16) return 2;\n"
              << "  return 1;\n"
              << "}\n"
              << "static inline void __ppl_static_aligned_stride(dim4 *stride, const dim4 *shape, data_type_t dtype) {\n"
              << "  int dtype_bytes = __ppl_dtype_bytes_for_stride(dtype);\n"
              << "  int align_elems = 64 / dtype_bytes;\n"
              << "  int hw_stride = shape->w;\n"
              << "  int c_stride = __ppl_align_up_i32(shape->h * shape->w, align_elems);\n"
              << "  stride->w = 1;\n"
              << "  stride->h = hw_stride;\n"
              << "  stride->c = c_stride;\n"
              << "  stride->n = __ppl_div_up_i32(shape->c, 64) * c_stride;\n"
              << "}\n\n";
  return CodeGenC::Finish();
  }

/* no need to change */
void CodeGenTileLangPPL::VisitStmt_(const tir::ForNode *op) {

  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  std::string extent =
      PrintExpr(arith::Analyzer().Simplify(op->extent + op->min));
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  std::string start = PrintExpr(op->min);
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = " << start << "; " << vid << " < " << extent
         << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  loop_var_ranges_.push_back(
      {op->loop_var, Range::FromMinExtent(op->min, op->extent)});
  PrintStmt(op->body);
  loop_var_ranges_.pop_back();
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangPPL::BindThreadIndex(const IterVar &iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}

void CodeGenTileLangPPL::PrintType(DataType t, std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  if (t == tl::cuTensorMapType()) {
    os << "CUtensorMap";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      if (t.is_scalar()) {
        os << "half_t";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp16 vector elements.
        //
        // half4 is stored as uint2
        //
        // h4.x is emitted as *(half2*)(&(u2.x)).x
        // h4.y is emitted as *(half2*)(&(u2.x)).y
        // h4.z is emitted as *(half2*)(&(u2.y)).x
        // h4.w is emitted as *(half2*)(&(u2.y)).y
        //
        ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
        os << "uint" << lanes / 2;
      } else {
        fail = true;
      }
      break;
    case 32:
      if (lanes <= 4) {
        os << "float";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp32 vector elements for 4 < lanes <= 8.
        //
        // float8 is stored as ulonglong4
        //
        // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
        // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for float type with lanes > 4";
        os << "ulonglong" << lanes / 2;
      } else {
        fail = true;
      }
      break;
    case 64:
      os << "double";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32))
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    if (t.is_scalar()) {
      os << "bfloat16_t";
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
      os << "uint" << lanes / 2;
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t.is_float8()) {
    if (t.is_scalar()) {
      os << "unsigned char"; // __nv_fp8_storage_t is an alias of unsigned char
    } else if (lanes == 2) {
      os << "unsigned short int"; // __nv_fp8x2_storage_t is an alias of
                                  // unsigned short
    } else if (lanes == 4) {
      os << "unsigned int"; // __nv_fp8x4_storage_t is an alias of unsigned int
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t == DataType::Bool()) {
    os << "bool";
    return;
  } else if (t.is_vector_bool()) {
    // CUDA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.is_scalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        os << "int8_t";
        return;
      } else if (t.lanes() == 16) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 32) {
        os << "int";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 4: {
      if (t.is_scalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 4) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 8) {
        // directly 8 4-bit int in integer.
        os << "int";
        return;
      } else if (t.lanes() == 16) {
        os << "int2";
        return;
      } else if (t.lanes() == 32) {
        os << "int4";
        return;
      } else if (t.lanes() == 64) {
        os << "int8";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 8: {
      if (t.lanes() == 4) {
        // directly 4 8 bit int in integer.

        // We use int for int8x4 instead of char4 because using char4 is
        // likely to produce extra instructions to pack four int8 elements
        // into 32-bit data.
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        os << "int2";
        return;
      } else if (t.lanes() == 16) {
        os << "int4";
        return;
      } else if (!t.is_uint() && t.is_scalar()) {
        os << "signed char";
        break;
      } else {
        os << "char";
        break;
      }
    }
    case 16: {
      if (t.is_scalar()) {
        os << "short";
      } else if (t.lanes() <= 4) {
        os << "short" << lanes;
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int16 vector elements.
        //
        // short4 is stored as int2
        //
        // s4.x is emitted as *(short2*)(&(i2.x)).x
        // s4.y is emitted as *(short2*)(&(i2.x)).y
        // s4.z is emitted as *(short2*)(&(i2.y)).x
        // s4.w is emitted as *(short2*)(&(i2.y)).y
        //
        ICHECK_EQ(t.lanes() % 2, 0)
            << "only support even lane for shorT type with lanes > 4";
        os << "int" << t.lanes() / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 32: {
      if (t.is_scalar()) {
        os << "int";
      } else if (t.lanes() <= 4) {
        os << "int" << t.lanes();
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int32 vector elements for 4 < lanes <= 8.
        //
        // int8 is stored as longlong4
        //
        // i8.v1 is emitted as *(int2*)(&(l4.x)).x
        // i8.v2 is emitted as *(int2*)(&(l4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for int32 type with lanes > 4";
        os << "longlong" << lanes / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 64: {
      if (t.is_scalar()) {
        os << "int64_t";
      } else if (t.lanes() == 2) {
        os << "longlong2";
      } else if (t.lanes() == 3) {
        os << "longlong3";
      } else if (t.lanes() == 4) {
        os << "longlong4";
      }
      return;
    }
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenTileLangPPL::PrintVecBinaryOp(const std::string &op, DataType t,
                                          PrimExpr lhs, PrimExpr rhs,
                                          std::ostream &os) { // NOLINT(*)
  // Delcare the result.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    // Unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());

    for (int i = 0, lanes = t.lanes(); i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenTileLangPPL::PrintVecElemLoad(const std::string &vec, DataType t,
                                          int i,
                                          std::ostream &os) { // NOLINT(*)
  if (t.is_scalar()) {
    os << vec;
    return;
  }

  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8                        ? 16
                        : (t.bits() == 16 || t.bits() == 32) ? 8
                                                             : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    std::string type_name = t.is_int() ? "char" : "unsigned char";
    if (t.lanes() == 2 || t.lanes() == 3) {
      os << vec << "." << access[i % t.lanes()];
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      os << "((" << type_name << ")(" << ac << " >> " << i % 4 * 8 << "))";
    }
  } else if (t.is_float16()) {
    os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
       << access[i % 2];
  } else if (t.is_bfloat16()) {
    os << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->"
       << access[i % 2];
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    os << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
       << ")))->" << access[i % 2];
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenTileLangPPL::PrintVecElemStore(const std::string &vec, DataType t,
                                           int i, const std::string &value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8                        ? 16
                        : (t.bits() == 16 || t.bits() == 32) ? 8
                                                             : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (t.lanes() == 2 || t.lanes() == 3) {
      stream << vec << '.' << access[i % t.lanes()] << "="
             << "(" << value << ");\n";
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i != 0) {
        stream << ac << " & ~(0x000000ff << " << i % 4 * 8 << ") |";
      }
      stream << "(" << value << " << " << i % 4 * 8 << ");\n";
    }
  } else if (t.is_float16()) {
    stream << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
           << access[i % 2] << " = " << value << ";\n";
  } else if (t.is_bfloat16()) {
    stream << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->"
           << access[i % 2] << " = " << value << ";\n";
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    stream << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
           << ")))->" << access[i % 2] << " = " << value << ";\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenTileLangPPL::PrintStorageSync(const CallNode *op) {
  const std::string &sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  }
}

void CodeGenTileLangPPL::PrintStorageScope(const std::string &scope,
                                           std::ostream &os) { // NOLINT(*)
  ICHECK_NE(scope, "global")
      << "Cannot allocate global memory when targeting CUDA. You must pass "
         "all global arrays as input instead";
  if (scope == "shared") {
    os << "__shared__ ";
  } else if (scope == "shared.dyn") {
    os << "extern __shared__ __align__(1024) ";
  }
}

std::string CodeGenTileLangPPL::CastFromTo(std::string value, DataType from,
                                           DataType target) {
  if (from == target)
    return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")";
  if (from.is_float16() && (target.is_int() || target.is_uint()) &&
      target.bits() == 8) {
    os << "(";
    if (target.is_uint()) {
      os << "u";
    }
    os << "int)";
  }
  os << value << ")";
  return os.str();
}

void CodeGenTileLangPPL::VisitExpr_(const CastNode *op, std::ostream &os) {
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  // Emit simple C-style type conversion.
  if (from_ty.is_scalar())
    return CodeGenC::VisitExpr_(op, os);

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  {
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
      std::ostringstream val;
      val << "(";
      PrintType(target_ty.element_of(), val);
      val << ")(";
      PrintVecElemLoad(src, from_ty, i, val);
      val << ")";
      PrintVecElemStore(sret, target_ty, i, val.str());
    }
  }
  os << sret;
}

void CodeGenTileLangPPL::PrintCallExtern(Type ret_type, String global_symbol,
                                         const Array<PrimExpr> &args,
                                         bool skip_first_arg,
                                         std::ostream &os) { // NOLINT(*)
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_vector()) {
    //
    // Emit an unsupported vector call
    //
    // v = intrin_f((float4*)A[0], (float4*)B[0])
    //
    // as
    //
    // float4 __ret;
    // {
    //   float4 __arg0 = ((float4*)A)[0];
    //   float4 __arg1 = ((float4*)B)[0];
    //   __ret.x = intrin_f(__arg0.x, __arg1.x);
    //   __ret.y = intrin_f(__arg0.y, __arg1.y);
    //   __ret.z = intrin_f(__arg0.z, __arg1.z);
    //   __ret.w = intrin_f(__arg0.w, __arg1.w);
    // }
    // v = __ret;
    //
    // Declare the result vector.
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(ret_dtype, stream);
    stream << ' ' << sret << ";\n";
    {
      // Load arguments.
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        std::string val = SSAGetID(PrintExpr(args[i]), args[i].dtype());
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0)
            scall << ", ";
          PrintVecElemLoad(sargs[j], args[arg_begin + j].dtype(), i, scall);
        }
        scall << ")";
        PrintVecElemStore(sret, ret_dtype, i, scall.str());
      }
    }
    os << sret;
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg,
                              os);
  }
}

// Print a reference expression to a buffer.
std::string CodeGenTileLangPPL::GetBufferRef(DataType t,
                                             const BufferNode *buffer,
                                             PrimExpr index) {
  const VarNode *buffer_var = buffer->data.get();
  std::ostringstream os;
  std::string vid = GetVarID(buffer_var);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  // bool is_vol = IsVolatile(buffer_var);
  // always false for tl cutlass backend.
  bool is_vol = false;

  auto ptr_cast = [this, is_vol, scope](DataType pointed_to) {
    std::ostringstream ptr_os;
    ptr_os << "(";
    if (is_vol) {
      ptr_os << "volatile ";
    }
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(scope, ptr_os);
    }
    PrintType(pointed_to, ptr_os);
    ptr_os << "*)";
    return ptr_os.str();
  };

  DataType buffer_element_dtype = buffer->dtype;

  std::string buffer_str = vid;
  if (!HandleTypeMatch(buffer_var, buffer_element_dtype) || is_vol) {
    std::stringstream temp;
    temp << "(" << ptr_cast(buffer_element_dtype) << vid << ")";
    buffer_str = temp.str();
  }

  std::string index_str = PrintExpr(index);
  if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
    // This is a special case, because CodegenCUDA::PrintType()
    // returns "int" for bool and for 4-bit integers. In most cases,
    // we divide by the number of lanes to determine the index.
    // However, the backing type for scalar int4 and scalar bool is
    // int32.  Therefore, we need to divide by the ratio of their
    // sizes in that case.
    int div_factor = (t.lanes() == 1) ? (32 / t.bits()) : t.lanes();

    os << "*("
       << "(" << ptr_cast(t) << vid << ")"
       << " + " << index_str << " / " << div_factor << ")";
  } else if (t == buffer_element_dtype) {
    os << buffer_str << "[" << index_str << "]";
  } else {
    os << "*" << ptr_cast(t) << "(" << buffer_str << " + " << index_str << ")";
  }

  return os.str();
}

inline std::string vector2string(const std::vector<int> &vec) {
  std::string ret = "{";
  for (auto &v : vec) {
    ret += std::to_string(v) + ", ";
  }
  ret[ret.size() - 2] = '}';
  return ret;
}

static inline std::string TargetDTypeName(DataType dtype) {
  if (dtype == DataType::Float(32)) {
    return "DT_FP32";
  } else if (dtype == DataType::Float(16)) {
    return "DT_FP16";
  } else if (dtype == DataType::BFloat(16)) {
    return "DT_BFP16";
  } else if (dtype.is_e5m2_float8()) {
    return "DT_FP8E5M2";
  } else if (dtype.is_e4m3_float8()) {
    return "DT_FP8E4M3";
  } else if (dtype == DataType::UInt(32)) {
    return "DT_UINT32";
  } else if (dtype == DataType::Int(32)) {
    return "DT_INT32";
  } else if (dtype == DataType::UInt(16)) {
    return "DT_UINT16";
  } else if (dtype == DataType::Int(16)) {
    return "DT_INT16";
  } else if (dtype == DataType::UInt(8)) {
    return "DT_UINT8";
  } else if (dtype == DataType::Int(8)) {
    return "DT_INT8";
  }
  LOG(FATAL) << "Unsupported dtype " << dtype;
  return "DT_FP32";
}

static inline int TargetDTypeBytes(DataType dtype) {
  if (dtype == DataType::Float(32) || dtype == DataType::UInt(32) ||
      dtype == DataType::Int(32)) {
    return 4;
  } else if (dtype == DataType::Float(16) ||
             dtype == DataType::BFloat(16) ||
             dtype == DataType::UInt(16) || dtype == DataType::Int(16)) {
    return 2;
  } else if (dtype.is_e5m2_float8() || dtype.is_e4m3_float8() ||
             dtype == DataType::UInt(8) || dtype == DataType::Int(8)) {
    return 1;
  }
  LOG(FATAL) << "Unsupported dtype " << dtype;
  return 0;
}

static inline const char* AsBDTypeStr(const DataType& dtype_) {
  if (dtype_ == DataType::Float(32)) {
    return "DT_FP32";
  } else if (dtype_ == DataType::Float(16)) {
    return "DT_FP16";
  } else if (dtype_ == DataType::BFloat(16)) {
    return "DT_BFP16";
  } else if (dtype_.is_e5m2_float8()) {
    return "DT_FP8E5M2";
  } else if (dtype_.is_e4m3_float8()) {
    return "DT_FP8E4M3";
  }

  // 其它类型回退为FP32
  return "DT_FP32";
}

inline int GetIntImmValueForDim4(const PrimExpr &expr, const char *context) {
  auto *imm = expr.as<IntImmNode>();
  ICHECK(imm) << context << " expects IntImm shape or region extent";
  ICHECK_GT(imm->value, 0)
      << context << " expects positive shape or region extent";
  ICHECK_LE(imm->value, std::numeric_limits<int>::max())
      << context << " shape or region extent exceeds int range";
  return static_cast<int>(imm->value);
}

inline std::vector<int> LowerDimValuesToDim4(const std::vector<int> &dims,
                                             bool local_layout) {
  int rank = dims.size();
  ICHECK(rank >= 1 && rank <= 4) << "Only support rank 1 to 4, but got "
                                 << rank;
  std::vector<int> dim4 = {1, 1, 1, 1};
  if (rank == 1) {
    dim4[3] = dims[0];
  } else if (rank == 2) {
    dim4[1] = dims[0];
    dim4[3] = dims[1];
  } else if (rank == 3) {
    if (local_layout) {
      dim4[0] = dims[0];
      dim4[1] = dims[1];
      dim4[3] = dims[2];
    } else {
      dim4[1] = dims[0];
      dim4[2] = dims[1];
      dim4[3] = dims[2];
    }
  } else {
    for (int i = 0; i < 4; i++) {
      dim4[i] = dims[i];
    }
  }
  return dim4;
}

inline std::vector<int> LowerRegionToDim4(const Array<Range> &ranges,
                                          bool local_layout) {
  int rank = ranges.size();
  std::vector<int> dims;
  dims.reserve(rank);
  for (const auto &range : ranges) {
    dims.push_back(GetIntImmValueForDim4(range->extent, "PPL copy region"));
  }
  return LowerDimValuesToDim4(dims, local_layout);
}

inline std::vector<int> LowerGlobalShapeToDim4(const Array<PrimExpr> &shape) {
  int rank = shape.size();
  std::vector<int> dims;
  dims.reserve(rank);
  for (const auto &dim : shape) {
    dims.push_back(GetIntImmValueForDim4(dim, "PPL global tensor"));
  }
  return LowerDimValuesToDim4(dims, true);
}

inline std::vector<int> StrideIndicesForRank(int rank, bool local_layout) {
  if (rank == 4) {
    return {0, 1, 2, 3};
  } else if (rank == 3) {
    return local_layout ? std::vector<int>{0, 1, 3}
                        : std::vector<int>{1, 2, 3};
  } else if (rank == 2) {
    return {1, 3};
  } else if (rank == 1) {
    return {3};
  }
  LOG(FATAL) << "Unsupported region dims: " << rank;
  return {};
}

void CodeGenTileLangPPL::VisitExpr_(const CallNode *op, std::ostream &os) {
  auto process_stride = [&,
                         this](const std::vector<int> &src0_shape,
                               const std::vector<int> &src1_shape,
                               const std::string &src0, const std::string &src1,
                               const std::string &dtype) -> std::stringstream {
    std::stringstream src1_stride;
    bool need_broadcast_stride = src1_shape.size() != src0_shape.size();
    size_t common_rank = std::min(src0_shape.size(), src1_shape.size());
    for (size_t i = 0; i < common_rank; ++i) {
      if (src1_shape[i] == 1 && src0_shape[i] != 1) {
        need_broadcast_stride = true;
      } else if (src1_shape[i] != src0_shape[i]) {
        need_broadcast_stride = true;
      }
    }
    if (need_broadcast_stride) {
      std::string stride_var = name_supply_->FreshName(src1 + "_stride");
      this->PrintIndent();
      this->stream << "dim4 " << stride_var << ";\n";
      this->PrintIndent();
      this->stream << "__ppl_static_aligned_stride(&" << stride_var << ", &" << src1
                   << ".shape, " << dtype << ");\n";
      bool src1_is_scalar = true;
      for (auto dim : src1_shape) {
        if (dim != 1) {
          src1_is_scalar = false;
          break;
        }
      }
      if (src1_is_scalar) {
        this->PrintIndent();
        this->stream << stride_var << ".n = 0;\n";
        this->PrintIndent();
        this->stream << stride_var << ".c = 0;\n";
        this->PrintIndent();
        this->stream << stride_var << ".h = 0;\n";
        this->PrintIndent();
        this->stream << stride_var << ".w = 0;\n";
      }
      if (src1_shape.size() >= 1 && src0_shape.size() >= 1 &&
          src1_shape[0] == 1 && src0_shape[0] != 1) {
        this->PrintIndent();
        this->stream << stride_var << ".c = 0;\n";
      }
      if (src1_shape.size() >= 2 && src0_shape.size() >= 2 &&
          src1_shape[1] == 1 && src0_shape[1] != 1) {
        this->PrintIndent();
        this->stream << stride_var << ".w = 0;\n";
      }
      src1_stride << "&" << stride_var << ", ";
    } else {
      src1_stride << "(" << src1 << ".default_stride ? NULL : &" << src1
                  << ".stride), ";
    }
    return src1_stride;
  };

  auto handle_elementwise = [&, this](std::string op_name, bool has_dtype) {
    auto dst = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
    auto src0 = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
    auto src1 = var_idmap_[op->args[3].as<CallNode>()->args[1].as<VarNode>()];
    auto dst_shape = buffer_shape[dst];
    auto src0_shape = buffer_shape[src0];
    auto src1_shape = buffer_shape[src1];
    auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
    std::string dtype = TargetDTypeName(dtype_);
    if (!has_dtype) {
      dtype = "";
    }
    std::stringstream src1_stride =
        process_stride(src0_shape, src1_shape, src0, src1, dtype);
    this->PrintIndent();
    this->stream << op_name << "( " << dst << ".addr, " << src0 << ".addr, "
                 << src1 << ".addr, "
                 << "&" << dst << ".shape, "
                 << "(" << dst << ".default_stride ? NULL : &" << dst
                 << ".stride), "
                 << "(" << src0 << ".default_stride ? NULL : &" << src0
                 << ".stride), " << src1_stride.str() << dtype << ");\n";
  };
  // void tpu_bdc_fp_mul_C(local_addr_t dst_addr, local_addr_t src_addr,
  // scalar_t C, const dim4 *shape, const dim4 *dst_stride, const dim4
  // *src_stride, data_type_t dtype)
  auto handle_elementwise_const = [&, this](std::string op_name) {
    auto dst = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
    auto src0 = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
    float value = Downcast<FloatImm>(op->args[3])->value;
    auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
    std::string dtype;
    std::string scalar_type;
    if (dtype_ == DataType::Float(16)) {
      dtype = "DT_FP16";
      scalar_type = "f16";
    } else if (dtype_ == DataType::Float(32)) {
      dtype = "DT_FP32";
      scalar_type = "f32";
    } else if (dtype_ == DataType::BFloat(16)){
      dtype = "DT_BFP16";
      scalar_type = "bf16";
    }
    if (dtype == "DT_FP32"){
      this->PrintIndent();
      this->stream << op_name << "( " << dst << ".addr, " << src0 << ".addr, "
                  << "(scalar_t){." << scalar_type << " = " << value << "}, &"
                  << dst << ".shape, "
                  << "(" << dst << ".default_stride ? NULL : &" << dst
                  << ".stride), "
                  << "(" << src0 << ".default_stride ? NULL : &" << src0
                  << ".stride), " << dtype << ");\n";
    }
    else{
      this->PrintIndent();
      this->stream << "{\n";
      this->PrintIndent();
      this->stream << "scalar_t " << dst << "_scalar_" << dtype << " = {.f32 = " << "(float)" << value << "};\n";
      this->PrintIndent();
      // x_neg_scalar_DT_BFP16 = tpu_cast(x_neg_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
      this->stream << dst << "_scalar_" << dtype << " = tpu_cast(" << dst << "_scalar_" << dtype << ", " << dtype << ", DT_FP32, RM_HALF_TO_EVEN);\n";
      this->PrintIndent();
      this->stream << op_name << "( " << dst << ".addr, " << src0 << ".addr, " <<  dst << "_scalar_" << dtype << ", " << "&" << dst << ".shape, " << "(" << dst << ".default_stride ? NULL : &" << dst << ".stride), " << "(" << src0 << ".default_stride ? NULL : &" << src0 << ".stride), " <<  dtype << ");\n";
      this->PrintIndent();
      this->stream << "}\n";
    }
  };
  std::vector<std::string> inst;
  std::function<std::string(const PrimExpr&)> print_index_expr = [&](const PrimExpr &expr) -> std::string {
    if (const FloorDivNode *div = expr.as<FloorDivNode>()) {
      return "(" + print_index_expr(div->a) + " / " + print_index_expr(div->b) + ")";
    }
    if (const FloorModNode *mod = expr.as<FloorModNode>()) {
      return "(" + print_index_expr(mod->a) + " % " + print_index_expr(mod->b) + ")";
    }
    return PrintExpr(expr);
  };
  if (op->op.same_as(builtin::call_extern())) {
    std::string op_name = Downcast<StringImm>(op->args[0])->value;
    if (op_name == "ppl.workitem_index") {
      os << "tpu_workitem_index()";
    } else if (op_name == "ppl.sync_all") {
      this->PrintIndent();
      this->stream << "tpu_sync_all();\n";
    } else if (op_name == "ppl.npu_bcast" || op_name == "ppl.dequant_fp8_block_scale") {
      tl::BufferMap buffer_map;
      struct SimpleRegionInfo {
        std::string var;
        std::string scope;
        std::string dtype;
        int elem_bytes;
        int rows;
        int cols;
      };
      auto dtype_info = [](const DataType &dtype_) -> std::pair<std::string, int> {
        if (dtype_ == DataType::Float(16)) {
          return {"DT_FP16", 2};
        } else if (dtype_ == DataType::BFloat(16)) {
          return {"DT_BFP16", 2};
        } else if (dtype_ == DataType::Float(32)) {
          return {"DT_FP32", 4};
        } else if (dtype_.is_e4m3_float8()) {
          return {"DT_FP8E4M3", 1};
        } else if (dtype_.is_e5m2_float8()) {
          return {"DT_FP8E5M2", 1};
        }
        LOG(FATAL) << "Unsupported dtype in " << dtype_;
        return {"", 0};
      };
      auto process_simple_region = [&, this](const tl::RegionOp &region) -> SimpleRegionInfo {
        auto buffer = region.GetBuffer();
        auto ranges = region.GetRanges();
        auto id = var_idmap_[buffer->data.get()];
        if (id.empty()) {
          id = this->parameter_map[buffer->name];
        }
        ICHECK(ranges.size() == 2 || ranges.size() == 4)
            << op_name << " only supports 2D/4D regions for now";
        auto range_extent = [&](const Range &sr) -> int {
          // TileLang encodes Python slices such as buf[64:128, 0] as a Ramp in
          // the BufferLoad index and may leave the explicit region extent as 1.
          // Prefer the Ramp lane count so local subviews keep their real tile
          // shape when lowered to PPL tensor descriptors.
          if (const RampNode *ramp = sr->min.as<RampNode>()) {
            if (const IntImmNode *lanes = ramp->lanes.as<IntImmNode>()) {
              return lanes->value;
            }
          }
          if (const IntImmNode *imm = sr->extent.as<IntImmNode>()) {
            return imm->value;
          }
          LOG(FATAL) << "Unsupported dynamic region extent in ppl region lowering";
          return 1;
        };
        int rows = 1;
        int cols = 1;
        if (ranges.size() == 2) {
          rows = range_extent(ranges[0]);
          cols = range_extent(ranges[1]);
        } else {
          rows = range_extent(ranges[1]);
          cols = range_extent(ranges[3]);
        }
        auto [dtype, bytes] = dtype_info(buffer->dtype);
        std::string new_var = name_supply_->FreshName(buffer->data->name_hint);
        if (buffer.scope() == "shared.dyn") {
          auto parent_var = var_idmap_[buffer->data.get()];
          std::vector<std::string> strides = {
              parent_var + ".stride.n", parent_var + ".stride.c",
              parent_var + ".stride.h", parent_var + ".stride.w"};
          std::vector<int> stride_idx = ranges.size() == 4 ? std::vector<int>{0, 1, 2, 3}
                                                           : std::vector<int>{1, 3};
          std::string min_expr;
          for (int i = 0; i < static_cast<int>(ranges.size()); ++i) {
            auto sr = ranges[i];
            const PrimExpr &e = sr->min;
            std::string idx_str;
            if (const RampNode *ramp = e.as<RampNode>()) {
              idx_str = print_index_expr(ramp->base);
            } else {
              idx_str = print_index_expr(e);
            }
            if (stride_idx[i] == 1) {
              // Align with Sophgo MLA PPL local C-dim view semantics: local C
              // addresses advance by NPU 64-lane groups, not by scalar C index.
              min_expr += "((" + idx_str + ") / 64) * " + strides[stride_idx[i]] + "+";
            } else {
              min_expr += "(" + idx_str + ") * " + strides[stride_idx[i]] + "+";
            }
          }
          min_expr[min_expr.size() - 1] = ' ';
          min_expr = "(" + min_expr + ") * " + std::to_string(bytes);
        inst.push_back("__ppl_tensor_info " + new_var + " = {.shape = {1, " +
                         std::to_string(rows) + ", 1, " + std::to_string(cols) +
                         "}, .stride = " + parent_var + ".stride, .addr = " +
                         parent_var + ".addr + " + min_expr + ", .dtype = " + dtype +
                         ", .mode = 0, .size = 1, .offset = " + min_expr + ", "
                         ".unsigned_flag = 0, .default_stride = false};\n");
        } else if (buffer.scope() == "global") {
          auto strides = buffer_stride[buffer->name];
          std::vector<int> stride_idx = ranges.size() == 4 ? std::vector<int>{0, 1, 2, 3}
                                                           : std::vector<int>{1, 3};
          std::string min_expr;
          for (int i = 0; i < static_cast<int>(ranges.size()); ++i) {
            auto sr = ranges[i];
            const PrimExpr &e = sr->min;
            std::string idx_str;
            if (const RampNode *ramp = e.as<RampNode>()) {
              idx_str = print_index_expr(ramp->base);
            } else {
              idx_str = print_index_expr(e);
            }
            min_expr += "(" + idx_str + ") * " + std::to_string(strides[stride_idx[i]]) + "+";
          }
          min_expr[min_expr.size() - 1] = ' ';
          min_expr = "(" + min_expr + ") * " + std::to_string(bytes);
        inst.push_back("__ppl_tensor_info " + new_var + " = {.shape = {1, " +
                         std::to_string(rows) + ", 1, " + std::to_string(cols) +
                         "}, .stride = {" + std::to_string(rows * cols) + ", " +
                         std::to_string(cols) + ", " + std::to_string(cols) +
                         ", 1}, .addr = " + id + ".addr + " + min_expr +
                         ", .dtype = " + dtype + ", .mode = 2, .size = 1, .offset = " +
                         min_expr + ", .unsigned_flag = 0, .default_stride = false};\n");
        } else {
          LOG(FATAL) << "Unsupported " << op_name << " scope: " << buffer.scope();
        }
        return {new_var, buffer.scope(), dtype, bytes, rows, cols};
      };

      if (op_name == "ppl.npu_bcast") {
        tl::RegionOp src = tl::RegionOp(op->args[1].as<CallNode>()->args, buffer_map);
        tl::RegionOp dst = tl::RegionOp(op->args[2].as<CallNode>()->args, buffer_map);
        auto src_info = process_simple_region(src);
        auto dst_info = process_simple_region(dst);
        ICHECK(src_info.scope == "shared.dyn" && dst_info.scope == "shared.dyn")
            << "ppl.npu_bcast supports local/shared tiles only";
        ICHECK(src_info.dtype == dst_info.dtype) << "ppl.npu_bcast requires matching dtype";
        if (dst_info.rows <= 64) {
        inst.push_back("tpu_bdc_npu_bcast(" + dst_info.var + ".addr, " + src_info.var +
                         ".addr, &" + dst_info.var + ".shape, " + dst_info.dtype + ");\n");
        } else {
          ICHECK(dst_info.rows % 64 == 0)
              << "ppl.npu_bcast rows > 64 must be 64-aligned for TPU NPU lanes";
          for (int row_off = 0; row_off < dst_info.rows; row_off += 64) {
            int chunk_rows = std::min(64, dst_info.rows - row_off);
            std::string chunk_var = name_supply_->FreshName(dst_info.var + "_chunk");
          inst.push_back("// Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.\n");
          inst.push_back("__ppl_tensor_info " + chunk_var + " = " + dst_info.var + ";\n");
          inst.push_back(chunk_var + ".shape.c = " + std::to_string(chunk_rows) + ";\n");
          inst.push_back(chunk_var + ".addr = " + dst_info.var + ".addr + ((" +
                           std::to_string(row_off / 64) + ") * " + dst_info.var +
                           ".stride.c) * " + std::to_string(dst_info.elem_bytes) + ";\n");
          inst.push_back(chunk_var + ".offset = " + dst_info.var + ".offset + ((" +
                           std::to_string(row_off / 64) + ") * " + dst_info.var +
                           ".stride.c) * " + std::to_string(dst_info.elem_bytes) + ";\n");
          inst.push_back("tpu_bdc_npu_bcast(" + chunk_var + ".addr, " + src_info.var +
                           ".addr, &" + chunk_var + ".shape, " + dst_info.dtype + ");\n");
          }
        }
      } else {
        tl::RegionOp src = tl::RegionOp(op->args[1].as<CallNode>()->args, buffer_map);
        tl::RegionOp scale = tl::RegionOp(op->args[2].as<CallNode>()->args, buffer_map);
        tl::RegionOp dst = tl::RegionOp(op->args[3].as<CallNode>()->args, buffer_map);
        auto src_info = process_simple_region(src);
        auto scale_info = process_simple_region(scale);
        auto dst_info = process_simple_region(dst);
        ICHECK(src_info.scope == "shared.dyn" && scale_info.scope == "shared.dyn" &&
               dst_info.scope == "shared.dyn")
            << op_name << " supports local/shared tiles only";
        ICHECK(src_info.dtype == "DT_FP8E4M3" || src_info.dtype == "DT_FP8E5M2")
            << op_name << " expects fp8 source";
        ICHECK(dst_info.dtype == "DT_BFP16" || dst_info.dtype == "DT_FP16")
            << op_name << " expects fp16/bf16 destination";
        ICHECK(scale_info.dtype == dst_info.dtype)
            << op_name << " scale dtype must match destination";
        ICHECK(src_info.rows == dst_info.rows && src_info.cols == dst_info.cols)
            << op_name << " source/destination shapes must match";
        std::string dst_stride = "(" + dst_info.var + ".default_stride ? NULL : &" +
                                 dst_info.var + ".stride)";
        std::string src_stride = "(" + src_info.var + ".default_stride ? NULL : &" +
                                 src_info.var + ".stride)";
        inst.push_back("tpu_bdc_cast(" + dst_info.var + ".addr, " + src_info.var +
                       ".addr, &" + dst_info.var + ".shape, " + dst_stride + ", " +
                       src_stride + ", " + dst_info.dtype + ", " + src_info.dtype +
                       ", RM_HALF_TO_EVEN);\n");
        ICHECK((scale_info.rows == 1 && scale_info.cols == 1) ||
               (scale_info.rows == dst_info.rows && scale_info.cols == 1))
            << "ppl.dequant_fp8_block_scale expects scalar scale or per-row compact scale";
        std::string scale_stride = name_supply_->FreshName(scale_info.var + "_block_stride");
        inst.push_back("dim4 " + scale_stride + ";\n");
        inst.push_back("__ppl_static_aligned_stride(&" + scale_stride + ", &" + scale_info.var +
                       ".shape, " + scale_info.dtype + ");\n");
        if (scale_info.rows == 1 && scale_info.cols == 1) {
          inst.push_back(scale_stride + ".c = 0;\n");
        } else {
          inst.push_back("// Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.\n");
        }
        inst.push_back(scale_stride + ".w = 0;\n");
        inst.push_back("tpu_bdc_fp_mul(" + dst_info.var + ".addr, " + dst_info.var +
                       ".addr, " + scale_info.var + ".addr, &" + dst_info.var +
                       ".shape, " + dst_stride + ", " + dst_stride + ", &" +
                       scale_stride + ", " + dst_info.dtype + ");\n");
      }
      for (auto &i : inst) {
        this->PrintIndent();
        this->stream << i;
      }
    } else if (op_name == "ppl.copy") {

      tl::BufferMap buffer_map;
      auto is_local_tensor_scope = [](const std::string &scope) {
        return scope == "shared.dyn" || scope == "local" ||
               scope == "local.fragment";
      };
      auto check_copy_bounds = [&, this](const tir::Buffer &buffer,
                                         const Array<Range> &ranges,
                                         const char *operand_name) {
        ICHECK_LE(ranges.size(), buffer->shape.size())
            << "ppl.copy " << operand_name << " rank mismatch: region rank "
            << ranges.size() << ", buffer rank " << buffer->shape.size();
        arith::Analyzer analyzer;
        for (const auto &loop_range : loop_var_ranges_) {
          analyzer.Bind(loop_range.first, loop_range.second);
        }
        size_t dim_offset = buffer->shape.size() - ranges.size();
        for (size_t i = 0; i < ranges.size(); ++i) {
          const Range &range = ranges[i];
          PrimExpr raw_min = range->min;
          PrimExpr min = raw_min.as<RampNode>()
                             ? raw_min.as<RampNode>()->base
                             : raw_min;
          min = analyzer.Simplify(min);
          PrimExpr upper = analyzer.Simplify(min + range->extent);
          PrimExpr shape_dim = buffer->shape[dim_offset + i];
          bool lower_bad = analyzer.CanProve(
              min < make_const(min.dtype(), 0),
              arith::ProofStrength::kSymbolicBound);
          bool upper_bad = analyzer.CanProve(
              upper > shape_dim, arith::ProofStrength::kSymbolicBound);
          // Keep rejecting regions that are provably invalid. Runtime scalar
          // offsets, such as Sophgo MLA seqlen-derived cache slots, cannot be
          // proven here and are validated by the host wrapper before launch.
          ICHECK(!lower_bad && !upper_bad)
              << "ppl.copy " << operand_name << " region is out of bounds "
              << "for buffer " << buffer->name << " at dim "
              << (dim_offset + i) << ": min=" << min
              << ", extent=" << range->extent << ", upper=" << upper
              << ", shape_dim=" << shape_dim
              << ". PPL copy does not support implicit tail masking; make "
              << "the tile divide the static shape or add explicit tail "
              << "handling in the frontend.";
        }
      };
      auto process_copy = [&, this](const tl::RegionOp &src,
                                    const Array<Range> &src_ranges,
                                    const char *operand_name)
          -> std::tuple<std::string, std::string, std::string> {
        auto src_buffer = src.GetBuffer();
        bool is_local = is_local_tensor_scope(src_buffer.scope());

        auto src_id = var_idmap_[src_buffer->data.get()];
        if (src_id.empty()) {
          src_id = this->parameter_map[src_buffer->name];
        }
        check_copy_bounds(src_buffer, src_ranges, operand_name);
        std::string new_src_var =
            name_supply_->FreshName(src_buffer->data->name_hint);
        bool use_local_rank3_layout = is_local || src_buffer.scope() == "global";
        std::string src_shape = vector2string(
            LowerRegionToDim4(src_ranges, use_local_rank3_layout));

        std::string dtype = TargetDTypeName(src_buffer->dtype);
        int bytes_size = TargetDTypeBytes(src_buffer->dtype);
        if (src_buffer.scope() == "global") {
          auto strides = buffer_stride[src_buffer->name];
          ICHECK_EQ(strides.size(), 4U)
              << "buffer_stride not initialized for global buffer: "
              << src_buffer->name;
          std::string src_strides = vector2string(strides);

          std::string min_expr;
          std::vector<int> stride_idx =
              StrideIndicesForRank(src_ranges.size(), true);

          for (size_t i = 0; i < src_ranges.size(); i++) {
            auto sr = src_ranges[i];
            const PrimExpr &e = sr->min;
            std::string idx_str;
            if (const RampNode *ramp = e.as<RampNode>()) {
              idx_str = print_index_expr(ramp->base);
            } else {
              idx_str = print_index_expr(e);
            }
            min_expr += "(" + idx_str + ") * " +
                        std::to_string(strides[stride_idx[i]]) + "+";
          }
          min_expr[min_expr.size() - 1] = ' ';
          min_expr = "(" + min_expr + ")" + " * " + std::to_string(bytes_size);
        inst.push_back("__ppl_tensor_info " + new_src_var +
                         " = {.shape = " + src_shape +
                         ", .stride = " + src_strides + ", .addr = " + src_id +
                         ".addr + " + min_expr + ", .dtype = " + dtype +
                         ", .mode = 2, .size = 1, .offset = " + min_expr +
                         ", .unsigned_flag = 0, .default_stride = false};\n");
        } else if (is_local) {
          auto parent_var = var_idmap_[src_buffer->data.get()];
          std::string min_expr;
          std::vector<std::string> strides = {
              parent_var + ".stride.n", parent_var + ".stride.c",
              parent_var + ".stride.h", parent_var + ".stride.w"};
          std::vector<int> stride_idx =
              StrideIndicesForRank(src_ranges.size(), true);
          for (size_t i = 0; i < src_ranges.size(); i++) {
            auto sr = src_ranges[i];
            const PrimExpr &e = sr->min;
            std::string idx_str;
            if (const RampNode *ramp = e.as<RampNode>()) {
              idx_str = PrintExpr(ramp->base);
            } else {
              idx_str = PrintExpr(e);
            }
            min_expr += "(" + idx_str + ") * " + strides[stride_idx[i]] + "+";
          }
          min_expr[min_expr.size() - 1] = ' ';
          min_expr = "(" + min_expr + ")" + " * " + std::to_string(bytes_size);
        inst.push_back("__ppl_tensor_info " + new_src_var + " = {.shape = " +
                         src_shape + ", .stride = " + parent_var +
                         ".stride, .addr = " + parent_var +
                         ".addr + " + min_expr + ", .dtype = " + dtype +
                         ", .mode = 0, .size = 1, .offset = " + min_expr + ", "
                         ".unsigned_flag = 0, .default_stride = " + parent_var +
                         ".default_stride};\n");
        } else {
          LOG(FATAL) << "Unsupported ppl.copy buffer scope: "
                     << src_buffer.scope();
        }
        return std::make_tuple(new_src_var,
                               src_buffer.scope() == "global" ? "global" : "local",
                               dtype);
      };
      tl::RegionOp src =
          tl::RegionOp(op->args[1].as<CallNode>()->args, buffer_map);
      tl::RegionOp dst =
          tl::RegionOp(op->args[2].as<CallNode>()->args, buffer_map);

      auto range_min_base = [](const Range &range) {
        if (const RampNode *ramp = range->min.as<RampNode>()) {
          return ramp->base;
        }
        return range->min;
      };
      auto split_rank3_c_range = [&](const Array<Range> &ranges, int c_idx) {
        Array<Range> split_ranges;
        for (size_t i = 0; i < ranges.size(); ++i) {
          if (i == 1) {
            PrimExpr min = range_min_base(ranges[i]) + c_idx;
            split_ranges.push_back(
                Range::FromMinExtent(min, make_const(min.dtype(), 1)));
          } else {
            split_ranges.push_back(ranges[i]);
          }
        }
        return split_ranges;
      };
      auto should_split_rank3_c = [](const Array<Range> &src_ranges,
                                     const Array<Range> &dst_ranges) {
        if (src_ranges.size() != 3 || dst_ranges.size() != 3) {
          return false;
        }
        int src_n = GetIntImmValueForDim4(src_ranges[0]->extent,
                                          "PPL copy region");
        int dst_n = GetIntImmValueForDim4(dst_ranges[0]->extent,
                                          "PPL copy region");
        int src_c = GetIntImmValueForDim4(src_ranges[1]->extent,
                                          "PPL copy region");
        int dst_c = GetIntImmValueForDim4(dst_ranges[1]->extent,
                                          "PPL copy region");
        return src_n == 1 && dst_n == 1 && src_c == dst_c && src_c > 1;
      };

      auto emit_copy = [&](const std::string &src_var_id,
                           const std::string &src_flag,
                           const std::string &src_dtype,
                           const std::string &dst_var_id,
                           const std::string &dst_flag,
                           const std::string &dst_dtype) {
        std::string ppl_inst;
        // dtype不同，copy同时做类型转换
        if (src_dtype != dst_dtype) {
          // void tpu_bdc_cast(local_addr_t dst_addr, local_addr_t src_addr,
          // const dim4 *shape, const dim4 *dst_stride, const dim4 *src_stride,
          // data_type_t dst_dtype, data_type_t src_dtype, rounding_mode_t mode)
          // 使用RM_HALF_TO_EVEN舍入模式，只有在浮点数据类型参与的转换时使用
          ppl_inst += "tpu_bdc_cast(" + dst_var_id + ".addr, " + src_var_id +
                      ".addr, " + "&" + dst_var_id + ".shape, " + "(" +
                      dst_var_id + ".default_stride ? NULL : &" + dst_var_id +
                      ".stride), " + "(" + src_var_id +
                      ".default_stride ? NULL : &" + src_var_id +
                      ".stride), " + dst_dtype + ", " + src_dtype + ", " +
                      "RM_HALF_TO_EVEN" + ");\n";
        inst.push_back(ppl_inst);
          return;
        }

        if (src_flag == "global" && dst_flag == "local") {
          ppl_inst += "tpu_gdma_cpy_S2L";
        } else if (src_flag == "local" && dst_flag == "global") {
          ppl_inst += "tpu_gdma_cpy_L2S";
        } else {
          // local mem -> local mem copy within the same NPU
          ppl_inst += "tpu_bdc_cpy";
        }

        // 生成真正 TPU 指令
        ppl_inst += "(" + dst_var_id + ".addr, " + src_var_id + ".addr, &" +
                    dst_var_id + ".shape, " + "(" + dst_var_id +
                    ".default_stride ? NULL : &" + dst_var_id + ".stride), " +
                    "(" + src_var_id + ".default_stride ? NULL : &" +
                    src_var_id + ".stride), " + src_dtype + ");\n";
        inst.push_back(ppl_inst);
      };

      auto emit_copy_for_ranges = [&](const Array<Range> &src_ranges,
                                      const Array<Range> &dst_ranges) {
        auto [src_var_id, src_flag, src_dtype] =
            process_copy(src, src_ranges, "src");
        auto [dst_var_id, dst_flag, dst_dtype] =
            process_copy(dst, dst_ranges, "dst");
        emit_copy(src_var_id, src_flag, src_dtype, dst_var_id, dst_flag,
                  dst_dtype);
      };

      auto src_ranges = src.GetRanges();
      auto dst_ranges = dst.GetRanges();
      if (should_split_rank3_c(src_ranges, dst_ranges)) {
        int c_extent =
            GetIntImmValueForDim4(src_ranges[1]->extent, "PPL copy region");
        for (int c_idx = 0; c_idx < c_extent; ++c_idx) {
          emit_copy_for_ranges(split_rank3_c_range(src_ranges, c_idx),
                               split_rank3_c_range(dst_ranges, c_idx));
        }
      } else {
        emit_copy_for_ranges(src_ranges, dst_ranges);
      }

      for (auto &i : inst) {
        this->PrintIndent();
        this->stream << i;
      }
    } else if (op_name == "ppl.fill") {
      auto var_ = op->args[1].as<CallNode>()->args[1].as<VarNode>();
      auto data_ = var_idmap_[var_];
      auto dtype = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string dtype_1, dtype_2;
      if (dtype == DataType::Float(16)) {
        dtype_1 = "f16";
        dtype_2 = "DT_FP16";
      } else if (dtype == DataType::Float(32)) {
        dtype_1 = "f32";
        dtype_2 = "DT_FP32";
      } else if (dtype == DataType::BFloat(16)) {
        dtype_1 = "bf16";
        dtype_2 = "DT_BFP16";
      } else {
        LOG(FATAL) << "Unsupported dtype in ppl.fill: " << dtype;
      }
      auto addr = buffer_addrs_[var_];
      double value = Downcast<FloatImm>(op->args[2])->value;
      std::string scalar_value;
      if (std::isnan(value)) {
        scalar_value = "(0.0f / 0.0f)";
      } else if (std::isinf(value)) {
        scalar_value = value > 0 ? "(1.0f / 0.0f)" : "(-1.0f / 0.0f)";
      } else {
        std::ostringstream literal;
        literal << std::scientific << value;
        if (dtype == DataType::Float(16) || dtype == DataType::Float(32) ||
            dtype == DataType::BFloat(16)) {
          literal << 'f';
        }
        scalar_value = literal.str();
      }
      this->PrintIndent();
      this->stream << "{\n";
      this->PrintIndent();
      this->stream << "scalar_t " << data_ << "_scalar_" << dtype_1 << " = {."
                   << dtype_1 << " = " << scalar_value << "};\n";
      this->PrintIndent();
      this->stream << "tpu_bdc_set_C(" << data_ << ".addr, " << data_
                   << "_scalar_" << dtype_1 << ", &" << data_ << ".shape, ("
                   << data_ << ".default_stride ? NULL : &" << data_
                   << ".stride), " << dtype_2 << ");\n";
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "ppl.gemm_region" || op_name == "ppl.gemm_acc_region") {
      tl::BufferMap buffer_map;
      struct GemmRegionInfo {
        std::string var;
        std::string dtype;
      };
      auto range_extent = [](const Range &sr) -> int {
        if (const RampNode *ramp = sr->min.as<RampNode>()) {
          if (const IntImmNode *lanes = ramp->lanes.as<IntImmNode>()) {
            return lanes->value;
          }
        }
        if (const IntImmNode *imm = sr->extent.as<IntImmNode>()) {
          return imm->value;
        }
        LOG(FATAL) << "Unsupported dynamic region extent in ppl.gemm_region";
        return 1;
      };
      auto process_gemm_region = [&, this](const tl::RegionOp &region) -> GemmRegionInfo {
        auto buffer = region.GetBuffer();
        auto ranges = region.GetRanges();
        ICHECK(ranges.size() == 2 || ranges.size() == 4)
            << "ppl.gemm_region only supports 2D/4D regions";
        ICHECK(buffer.scope() == "shared.dyn" || buffer.scope() == "local" ||
               buffer.scope() == "local.fragment")
            << "ppl.gemm_region supports local/shared tiles only";
        int rows = 1;
        int cols = 1;
        if (ranges.size() == 2) {
          rows = range_extent(ranges[0]);
          cols = range_extent(ranges[1]);
        } else {
          rows = range_extent(ranges[1]);
          cols = range_extent(ranges[3]);
        }
        std::string dtype = TargetDTypeName(buffer->dtype);
        int bytes = TargetDTypeBytes(buffer->dtype);
        std::string parent_var = var_idmap_[buffer->data.get()];
        if (parent_var.empty()) {
          parent_var = this->parameter_map[buffer->name];
        }
        std::vector<std::string> strides = {
            parent_var + ".stride.n", parent_var + ".stride.c",
            parent_var + ".stride.h", parent_var + ".stride.w"};
        std::vector<int> stride_idx = ranges.size() == 4 ? std::vector<int>{0, 1, 2, 3}
                                                         : std::vector<int>{1, 3};
        std::string min_expr;
        for (int i = 0; i < static_cast<int>(ranges.size()); ++i) {
          auto sr = ranges[i];
          const PrimExpr &e = sr->min;
          std::string idx_str;
          if (const RampNode *ramp = e.as<RampNode>()) {
            idx_str = print_index_expr(ramp->base);
          } else {
            idx_str = print_index_expr(e);
          }
          if (stride_idx[i] == 1) {
            // Align with Sophgo MLA PPL local view semantics: C-dim local
            // offsets advance by 64-lane groups in NPU layout.
            min_expr += "((" + idx_str + ") / 64) * " + strides[stride_idx[i]] + "+";
          } else {
            min_expr += "(" + idx_str + ") * " + strides[stride_idx[i]] + "+";
          }
        }
        min_expr[min_expr.size() - 1] = ' ';
        min_expr = "(" + min_expr + ") * " + std::to_string(bytes);
        std::string new_var = name_supply_->FreshName(buffer->data->name_hint);
        inst.push_back("__ppl_tensor_info " + new_var + " = {.shape = {1, " +
                       std::to_string(rows) + ", 1, " + std::to_string(cols) +
                       "}, .stride = " + parent_var + ".stride, .addr = " +
                       parent_var + ".addr + " + min_expr + ", .dtype = " + dtype +
                       ", .mode = 0, .size = 1, .offset = " + min_expr +
                       ", .unsigned_flag = 0, .default_stride = false};\n");
        return {new_var, dtype};
      };

      tl::RegionOp a_region = tl::RegionOp(op->args[1].as<CallNode>()->args, buffer_map);
      tl::RegionOp b_region = tl::RegionOp(op->args[2].as<CallNode>()->args, buffer_map);
      tl::RegionOp c_region = tl::RegionOp(op->args[3].as<CallNode>()->args, buffer_map);
      auto a_info = process_gemm_region(a_region);
      auto b_info = process_gemm_region(b_region);
      auto c_info = process_gemm_region(c_region);
      auto M = Downcast<IntImm>(op->args[6])->value;
      auto N = Downcast<IntImm>(op->args[7])->value;
      auto K = Downcast<IntImm>(op->args[8])->value;
      auto trans_B = Downcast<Bool>(op->args[5])->value;
      bool accumulate_result = (op_name == "ppl.gemm_acc_region");
      std::string output_dtype_accum = "DT_FP32";
      std::string output_dtype_trans = (c_info.dtype == a_info.dtype) ? a_info.dtype : "DT_FP32";
      std::string M_K_N = std::to_string(M) + ", " + std::to_string(K) + ", " +
                          std::to_string(N);
      for (auto &i : inst) {
        this->PrintIndent();
        this->stream << i;
      }
      this->PrintIndent();
      if (!trans_B) {
        this->stream << "tpu_bdc_fp_mm(" << c_info.var << ".addr, "
                     << a_info.var << ".addr, " << b_info.var << ".addr, "
                     << M_K_N << ", " << output_dtype_accum << ", "
                     << a_info.dtype << ", true);\n";
      } else if (accumulate_result) {
        // Align with Sophgo MLA PPL GEMM accumulation form: use the
        // R-transposed with_bias API so the GEMM result can be accumulated
        // into the destination tile without a separate tpu_bdc_fp_add.
        this->stream << "tpu_bdc_fp_mm_R_trans_with_bias(" << c_info.var << ".addr, "
                     << a_info.var << ".addr, " << b_info.var << ".addr, "
                     << M_K_N << ", " << output_dtype_trans << ", "
                     << a_info.dtype << ", DT_FP32, true, true, "
                     << "(var_context_t){0}, false);\n";
      } else {
        this->stream << "tpu_bdc_fp_mm_R_trans(" << c_info.var << ".addr, "
                     << a_info.var << ".addr, " << b_info.var << ".addr, "
                     << M_K_N << ", " << output_dtype_trans << ", "
                     << a_info.dtype << ");\n";
      }
    } else if (op_name == "ppl.gemm") {
      auto a_access_data =
          var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto b_access_data =
          var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto c_access_data =
          var_idmap_[op->args[3].as<CallNode>()->args[1].as<VarNode>()];

      auto M = Downcast<IntImm>(op->args[6])->value;
      auto N = Downcast<IntImm>(op->args[7])->value;
      auto K = Downcast<IntImm>(op->args[8])->value;
      auto trans_B = Downcast<Bool>(op->args[5])->value;

      auto a_dtype = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto b_dtype = op->args[2].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto c_dtype = op->args[3].as<CallNode>()->args[0].as<CallNode>()->dtype;
      const char* left_right_dtype = AsBDTypeStr(a_dtype);
      // Non-transpose path: accumulation requires FP32
      const char* output_dtype_accum = "DT_FP32"; 
      // Transpose path: use matching precision when C matches A/B, otherwise use FP32
      const char* output_dtype_trans =
          (std::string(AsBDTypeStr(c_dtype)) == left_right_dtype) ? left_right_dtype : "DT_FP32";

      std::string M_K_N = std::to_string(M) + ", " + std::to_string(K) + ", " +
                          std::to_string(N);
      this->PrintIndent();
      if (!trans_B)
        this->stream << "tpu_bdc_fp_mm(" << c_access_data << ".addr, "
                     << a_access_data << ".addr, " << b_access_data << ".addr, "
                     << M_K_N << ", " << output_dtype_accum << ", "  
                     << left_right_dtype << ", true);\n";
      else
        this->stream << "tpu_bdc_fp_mm_R_trans(" << c_access_data << ".addr, "
                     << a_access_data << ".addr, " << b_access_data << ".addr, "
                     << M_K_N << ", " << output_dtype_trans << ", "
                     << left_right_dtype << ");\n";
    } else if (op_name == "ppl.sub") {
      handle_elementwise("tpu_bdc_fp_sub", true);
    } else if (op_name == "ppl.mul") {
      handle_elementwise("tpu_bdc_fp_mul", true);
    } else if (op_name == "ppl.add") {
      handle_elementwise("tpu_bdc_fp_add", true);
    } else if (op_name == "ppl.div") {
      handle_elementwise("tpu_bdc_fp_div", true);
    } else if (op_name == "ppl.mul_C") {
      handle_elementwise_const("tpu_bdc_fp_mul_C");
    } else if (op_name == "ppl.add_C") {
      handle_elementwise_const("tpu_bdc_fp_add_C");
    }
    /** The following op needs to be handled specially. */
    else if (op_name == "ppl.exp") {
      auto work0 =
          var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto work1 =
          var_idmap_[op->args[3].as<CallNode>()->args[1].as<VarNode>()];
      auto coeff =
          var_idmap_[op->args[4].as<CallNode>()->args[1].as<VarNode>()];
      auto table =
          var_idmap_[op->args[5].as<CallNode>()->args[1].as<VarNode>()];
      this->PrintIndent();
      this->stream << "tpu_bdc_load_fp32_exp_coeff(" << coeff << ".addr"
                   << ");\n";
      this->PrintIndent();
      this->stream << "tpu_bdc_load_fp32_exp_table(" << table << ".addr"
                   << ");\n";
      auto dst = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto src0 = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto src0_shape = buffer_shape[src0];
      // void tpu_bdc_fp32_exp(local_addr_t dst_addr, local_addr_t src_addr,
      // local_addr_t work0_addr, local_addr_t work1_addr, local_addr_t
      // coeff_addr, local_addr_t table_addr, const dim4 *shape)
      this->PrintIndent();
      this->stream << "tpu_bdc_fp32_exp(" << dst << ".addr, " << src0
                   << ".addr, " << work0 << ".addr, " << work1 << ".addr, "
                   << coeff << ".addr, " << table << ".addr, "
                   << "&" << src0 << ".shape"
                   << ");\n";
    } else if (op_name == "ppl.sigmoid") {
      auto dst_dtype = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto src_dtype = op->args[2].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto work0_dtype = op->args[3].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto work1_dtype = op->args[4].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto coeff_dtype = op->args[5].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto table_dtype = op->args[6].as<CallNode>()->args[0].as<CallNode>()->dtype;
      ICHECK(dst_dtype == DataType::Float(32) && src_dtype == DataType::Float(32) &&
             work0_dtype == DataType::Float(32) && work1_dtype == DataType::Float(32) &&
             coeff_dtype == DataType::Float(32) && table_dtype == DataType::Float(32))
          << "ppl.sigmoid expects FP32 for dst/src/work0/work1/coeff/table";

      auto dst = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto src = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto work0 = var_idmap_[op->args[3].as<CallNode>()->args[1].as<VarNode>()];
      auto work1 = var_idmap_[op->args[4].as<CallNode>()->args[1].as<VarNode>()];
      auto coeff = var_idmap_[op->args[5].as<CallNode>()->args[1].as<VarNode>()];
      auto table = var_idmap_[op->args[6].as<CallNode>()->args[1].as<VarNode>()];

      this->PrintIndent();
      this->stream << "tpu_bdc_load_fp32_exp_coeff(" << coeff << ".addr"
                   << ");\n";
      this->PrintIndent();
      this->stream << "tpu_bdc_load_fp32_exp_table(" << table << ".addr"
                   << ");\n";
      this->PrintIndent();
      this->stream << "tpu_bdc_fp32_sigmoid(" << dst << ".addr, " << src
                   << ".addr, " << work0 << ".addr, " << work1 << ".addr, "
                   << coeff << ".addr, " << table << ".addr, "
                   << "&" << src << ".shape"
                   << ");\n";
    } else if (op_name == "ppl.reduce_max") {
      // 提取输入、输出和临时张量
      auto input_tensor =
          var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto output_tensor =
          var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto tmp_tensor =
          var_idmap_[op->args[3].as<CallNode>()->args[1].as<VarNode>()];
      auto eu_num = Downcast<IntImm>(op->args[4])->value;
      auto align_w = Downcast<IntImm>(op->args[5])->value;
      auto stride_n = Downcast<IntImm>(op->args[6])->value;
      // 获取数据类型
      auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string dtype;
      if (dtype_ == DataType::Float(16)) {
        dtype = "DT_FP16";
      } else if (dtype_ == DataType::Float(32)) {
        dtype = "DT_FP32";
      } else if (dtype_ == DataType::BFloat(16)) {
        dtype = "DT_BFP16";
      }

      this->PrintIndent();
      int sid = this->BeginScope();
      this->stream << "{\n";

      // 计算EU数和对齐尺寸
      this->PrintIndent();
      this->stream << "int eu_num = " << eu_num << ";\n";
      this->PrintIndent();
      this->stream << "int align_w = " << align_w << ";\n";

      // 创建pad_val; FP_NEG_MAX returns the integer bit-pattern of -MAX,
      // so we must populate the `u32` field to reinterpret the bits,
      // not the float field which would cast the integer to float.
      this->PrintIndent();
      this->stream << "scalar_t pad_val = {.u32 = FP_NEG_MAX(" << dtype
                   << ")};\n";

      // 创建池化所需的形状
      this->PrintIndent();
      this->stream << "dim4 in_reduce_h = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, align_w / eu_num, eu_num};\n";
      this->PrintIndent();
      this->stream << "dim4 out_reduce_h = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, eu_num};\n";
      this->PrintIndent();
      this->stream << "dim4 in_reduce_w = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, eu_num};\n";
      this->PrintIndent();
      this->stream << "dim4 out_reduce_w = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, 1};\n";

      // 设置池化参数
      this->PrintIndent();
      this->stream << "dim2 kernel = {align_w / eu_num, 1};\n";
      this->PrintIndent();
      this->stream << "padding_t pad = {0, 0, 0, 0};\n";
      this->PrintIndent();
      this->stream << "dim2 stride = {1, 1};\n";
      this->PrintIndent();
      this->stream << "dim2 dilation = {1, 1};\n";

      this->PrintIndent();
      this->stream << "if (align_w > " << input_tensor
                   << ".shape.w && align_w == eu_num) {\n";
      this->PrintIndent();
      this->stream << "  dim4 padded_stride = {" << stride_n
                   << ", align_w, align_w, 1};\n";
      this->PrintIndent();
      this->stream << "  dim4 padded_shape = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, align_w};\n";
      this->PrintIndent();
      this->stream << "  dim4 copy_shape = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, " << input_tensor
                   << ".shape.w};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info padded_input = {.shape = "
                   << "padded_shape, .stride = padded_stride, .addr = "
                   << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", .mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = false};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info input_copy = {.shape = copy_shape, "
                   << ".stride = {0}, .addr = " << input_tensor
                   << ".addr, .dtype = " << dtype
                   << ", .mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info padded_input_copy = {.shape = "
                   << "copy_shape, .stride = padded_stride, .addr = "
                   << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", .mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = false};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info output_view = {.shape = "
                   << "out_reduce_w, .stride = {0}, .addr = "
                   << output_tensor << ".addr, .dtype = " << dtype
                   << ", .mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";
      this->PrintIndent();
      this->stream << "  tpu_bdc_set_C(padded_input.addr, pad_val, "
                   << "&padded_shape, &padded_input.stride, " << dtype << ");\n";
      this->PrintIndent();
      this->stream << "  tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, "
                   << "&copy_shape, &padded_input_copy.stride, "
                   << "(input_copy.default_stride ? NULL : &input_copy.stride), "
                   << dtype << ");\n";
      this->PrintIndent();
      this->stream << "  dim2 kernel2 = {1, eu_num};\n";
      this->PrintIndent();
      this->stream << "  pad_val.u32 = FP_NEG_MAX(" << dtype << ");\n";
      this->PrintIndent();
      this->stream << "  tpu_bdc_fp_max_pool2d(output_view.addr, "
                   << "padded_input.addr, &padded_shape, &kernel2, &pad, "
                   << "&stride, &dilation, " << dtype << ", pad_val);\n";
      this->PrintIndent();
      this->stream << "} else {\n";

      this->PrintIndent();
      this->stream << "  if (align_w > " << input_tensor << ".shape.w) {\n";
      this->PrintIndent();
      this->stream << "    dim4 fill_shape = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, align_w - " << input_tensor
                   << ".shape.w};\n";
      this->PrintIndent();
      int elem_size =
          (dtype_ == DataType::Float(16) || dtype_ == DataType::BFloat(16)) ? 2 : 4;
      this->stream << "    int elem_size = " << elem_size << ";\n";
      this->PrintIndent();
      this->stream << "    int offset = " << input_tensor
                   << ".shape.w * elem_size;\n";
      this->PrintIndent();
      this->stream << "    dim4 fill_tensor_stride = {" << stride_n
                   << ", align_w, " << input_tensor << ".shape.w, 1};\n";
      this->PrintIndent();
      this->stream
          << "    __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride "
             "= fill_tensor_stride, "
          << ".addr = " << input_tensor << ".addr + offset, .dtype = " << dtype
          << ", "
          << ".mode = 0, .align_mode = 4, .size = 1, .offset = offset, "
          << ".unsigned_flag = 0, .default_stride = false};\n";
      this->PrintIndent();
      this->stream
          << "    tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, "
          << "(fill_tensor.default_stride ? NULL : &fill_tensor.stride), "
          << dtype << ");\n";
      this->PrintIndent();
      this->stream << "  }\n";

      this->PrintIndent();
      this->stream << "  __ppl_tensor_info input_view = {.shape = in_reduce_h, "
                      ".stride = {0}, "
                   << ".addr = " << input_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";

      this->PrintIndent();
      this->stream << "  __ppl_tensor_info tmp_view = {.shape = out_reduce_h, "
                      ".stride = {0}, "
                   << ".addr = " << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";

      this->PrintIndent();
      this->stream << "  tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, "
                      "&input_view.shape, "
                   << "&kernel, &pad, &stride, &dilation, " << dtype
                   << ", pad_val);\n";
      this->PrintIndent();
      this->stream << "  dim2 kernel2 = {1, eu_num};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info output_view = {.shape = out_reduce_w, "
                      ".stride = {0}, "
                   << ".addr = " << output_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, "
                      ".stride = {0}, "
                   << ".addr = " << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";
      this->PrintIndent();
      this->stream << "  pad_val.u32 = FP_NEG_MAX(" << dtype << ");\n";
      this->PrintIndent();
      this->stream << "  tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, "
                      "&tmp_view2.shape, "
                   << "&kernel2, &pad, &stride, &dilation, " << dtype
                   << ", pad_val);\n";
      this->PrintIndent();
      this->stream << "}\n";

      // 结束块作用域
      this->EndScope(sid);
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "ppl.reduce_sum") {
      // 提取输入、输出和临时张量
      auto input_tensor =
          var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto output_tensor =
          var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto tmp_tensor =
          var_idmap_[op->args[3].as<CallNode>()->args[1].as<VarNode>()];
      auto eu_num = Downcast<IntImm>(op->args[4])->value;
      auto align_w = Downcast<IntImm>(op->args[5])->value;
      auto stride_n = Downcast<IntImm>(op->args[6])->value;

      this->PrintIndent();
      int sid = this->BeginScope();
      this->stream << "{\n";

      auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string dtype, dtype_2;
      if (dtype_ == DataType::Float(16)) {
        dtype = "DT_FP16";
        dtype_2 = "f16";
      } else if (dtype_ == DataType::Float(32)) {
        dtype = "DT_FP32";
        dtype_2 = "f32";
      } else if (dtype_ == DataType::BFloat(16)) {
        dtype = "DT_BFP16";
        dtype_2 = "bf16";
      } else if (dtype_ == DataType::Int(32)) {
        dtype = "DT_INT32";
        dtype_2 = "s32";
      } else if (dtype_ == DataType::UInt(32)) {
        dtype = "DT_UINT32";
        dtype_2 = "u32";
      } else if (dtype_ == DataType::Int(16)) {
        dtype = "DT_INT16";
        dtype_2 = "s16";
      } else {
        LOG(FATAL) << "Unsupported dtype " << dtype_;
      }
      // 计算EU数和对齐尺寸
      this->PrintIndent();
      this->stream << "int eu_num = " << eu_num << ";\n";
      this->PrintIndent();
      this->stream << "int align_w = " << align_w << ";\n";

      // // 创建填充值（对于sum，使用0作为填充值）
      this->PrintIndent();
      this->stream << "scalar_t pad_val = {." << dtype_2 << " = 0};\n";

      // 创建池化所需的形状
      this->PrintIndent();
      this->stream << "dim4 in_reduce_h = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, align_w / eu_num, eu_num};\n";
      this->PrintIndent();
      this->stream << "dim4 out_reduce_h = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, eu_num};\n";
      this->PrintIndent();
      this->stream << "dim4 in_reduce_w = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, eu_num};\n";
      this->PrintIndent();
      this->stream << "dim4 out_reduce_w = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, 1};\n";

      // 设置池化参数
      this->PrintIndent();
      this->stream << "dim2 kernel = {align_w / eu_num, 1};\n";
      this->PrintIndent();
      this->stream << "padding_t pad = {0, 0, 0, 0};\n";
      this->PrintIndent();
      this->stream << "dim2 stride = {1, 1};\n";
      this->PrintIndent();
      this->stream << "dim2 dilation = {1, 1};\n";

      this->PrintIndent();
      this->stream << "scalar_t scale = {.f32 = (float)1.000000000e+00};\n";
      if (dtype_ == DataType::Float(16)) {
        this->PrintIndent();
        this->stream
            << "scale = tpu_cast(scale, DT_FP16, DT_FP32, RM_HALF_TO_EVEN);\n";
      } else if (dtype_ == DataType::BFloat(16)) {
        this->PrintIndent();
        this->stream
            << "scale = tpu_cast(scale, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);\n";
      } else if (dtype_ == DataType::Int(32)) {
        this->PrintIndent();
        this->stream << "scale = tpu_cast(scale, DT_INT32, DT_FP32, "
                        "RM_HALF_TO_EVEN);\n";
      } else if (dtype_ == DataType::UInt(32)) {
        this->PrintIndent();
        this->stream << "scale = tpu_cast(scale, DT_UINT32, DT_FP32, "
                        "RM_HALF_TO_EVEN);\n";
      }

      this->PrintIndent();
      this->stream << "if (align_w > " << input_tensor
                   << ".shape.w && align_w == eu_num) {\n";
      this->PrintIndent();
      this->stream << "  dim4 padded_stride = {" << stride_n
                   << ", align_w, align_w, 1};\n";
      this->PrintIndent();
      this->stream << "  dim4 padded_shape = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, align_w};\n";
      this->PrintIndent();
      this->stream << "  dim4 copy_shape = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, " << input_tensor
                   << ".shape.w};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info padded_input = {.shape = "
                   << "padded_shape, .stride = padded_stride, .addr = "
                   << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", .mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = false};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info input_copy = {.shape = copy_shape, "
                   << ".stride = {0}, .addr = " << input_tensor
                   << ".addr, .dtype = " << dtype
                   << ", .mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info padded_input_copy = {.shape = "
                   << "copy_shape, .stride = padded_stride, .addr = "
                   << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", .mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = false};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info output_view = {.shape = "
                   << "out_reduce_w, .stride = {0}, .addr = "
                   << output_tensor << ".addr, .dtype = " << dtype
                   << ", .mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";
      this->PrintIndent();
      this->stream << "  tpu_bdc_set_C(padded_input.addr, pad_val, "
                   << "&padded_shape, &padded_input.stride, " << dtype << ");\n";
      this->PrintIndent();
      this->stream << "  tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, "
                   << "&copy_shape, &padded_input_copy.stride, "
                   << "(input_copy.default_stride ? NULL : &input_copy.stride), "
                   << dtype << ");\n";
      this->PrintIndent();
      this->stream << "  dim2 kernel2 = {1, eu_num};\n";
      this->PrintIndent();
      this->stream << "  tpu_bdc_fp_avg_pool2d(output_view.addr, "
                   << "padded_input.addr, &padded_shape, &kernel2, &pad, "
                   << "&stride, &dilation, " << dtype << ", scale);\n";
      this->PrintIndent();
      this->stream << "} else {\n";
      this->PrintIndent();
      this->stream << "  if (align_w > " << input_tensor << ".shape.w) {\n";
      this->PrintIndent();
      this->stream << "    dim4 fill_shape = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, align_w - " << input_tensor
                   << ".shape.w};\n";
      this->PrintIndent();
      int elem_size = 1;
      if (dtype_ == DataType::Float(16)) {
        elem_size = 2;
      } else if (dtype_ == DataType::Float(32)) {
        elem_size = 4;
      } else if (dtype_ == DataType::BFloat(16)) {
        elem_size = 2;
      } else if (dtype_ == DataType::Int(32)) {
        elem_size = 4;
      } else if (dtype_ == DataType::UInt(32)) {
        elem_size = 4;
      } else if (dtype_ == DataType::Int(16)) {
        elem_size = 2;
      } else {
        LOG(FATAL) << "Unsupported dtype " << dtype_;
      }
      this->stream << "    int elem_size = " << elem_size << ";\n";
      this->PrintIndent();
      this->stream << "    int offset = " << input_tensor
                   << ".shape.w * elem_size;\n";
      this->PrintIndent();
      this->stream << "    dim4 fill_tensor_stride = {" << stride_n
                   << ", align_w, " << input_tensor << ".shape.w, 1};\n";
      this->PrintIndent();
      this->stream
          << "    __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride "
             "= fill_tensor_stride, "
          << ".addr = " << input_tensor << ".addr + offset, .dtype = " << dtype
          << ", "
          << ".mode = 0, .align_mode = 4, .size = 1, .offset = offset, "
          << ".unsigned_flag = 0, .default_stride = false};\n";
      this->PrintIndent();
      this->stream
          << "    tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, "
          << "(fill_tensor.default_stride ? NULL : &fill_tensor.stride), "
          << dtype << ");\n";
      this->PrintIndent();
      this->stream << "  }\n";

      this->PrintIndent();
      this->stream << "  __ppl_tensor_info input_view = {.shape = in_reduce_h, "
                      ".stride = {0}, "
                   << ".addr = " << input_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info tmp_view = {.shape = out_reduce_h, "
                      ".stride = {0}, "
                   << ".addr = " << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";
      this->PrintIndent();
      this->stream << "  tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, "
                      "&input_view.shape, "
                   << "&kernel, &pad, &stride, &dilation, " << dtype
                   << ", scale);\n";
      this->PrintIndent();
      this->stream << "  dim2 kernel2 = {1, eu_num};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info output_view = {.shape = out_reduce_w, "
                      ".stride = {0}, "
                   << ".addr = " << output_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";
      this->PrintIndent();
      this->stream << "  __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, "
                      ".stride = {0}, "
                   << ".addr = " << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";
      this->PrintIndent();
      this->stream << "  tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, "
                      "&tmp_view2.shape, "
                   << "&kernel2, &pad, &stride, &dilation, " << dtype
                   << ", scale);\n";
      this->PrintIndent();
      this->stream << "}\n";

      // 结束块作用域
      this->EndScope(sid);
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "ppl.rsqrt") {
      auto dst_dtype = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto src_dtype = op->args[2].as<CallNode>()->args[0].as<CallNode>()->dtype;
      ICHECK(dst_dtype == DataType::Float(32) && src_dtype == DataType::Float(32))
          << "ppl.rsqrt expects FP32 for dst/src0, got dst=" << dst_dtype
          << ", src0=" << src_dtype;
      auto dst = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto src0 = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto src0_shape = buffer_shape[src0];
      // void tpu_bdc_fp32_rsqrt(local_addr_t dst_addr, local_addr_t src_addr, const dim4 *shape)
      this->PrintIndent();
      this->stream << "tpu_bdc_fp32_rsqrt(" << dst << ".addr, " << src0
                   << ".addr, "
                   << "&" << src0 << ".shape"
                   << ");\n";

    } else if (op_name == "ppl.rope_add") {
      auto dst = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto even_src0 = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto even_src1 = var_idmap_[op->args[3].as<CallNode>()->args[1].as<VarNode>()];
      auto odd_src0 = var_idmap_[op->args[4].as<CallNode>()->args[1].as<VarNode>()];
      auto odd_src1 = var_idmap_[op->args[5].as<CallNode>()->args[1].as<VarNode>()];
      auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string dtype;
      int bytes_size = 0;
      if (dtype_ == DataType::Float(16)){
        dtype = "DT_FP16";
        bytes_size = 2;
      } else if (dtype_ == DataType::Float(32)) {
        dtype = "DT_FP32";
        bytes_size = 4;
      } else if (dtype_ == DataType::BFloat(16)){
        dtype = "DT_BFP16";
        bytes_size = 2;
      }
      this->PrintIndent();
      this->stream << "{\n";
      this->PrintIndent();
      this->stream << "dim4 half_stride;\n";
      this->PrintIndent();
      this->stream << "__ppl_static_aligned_stride(&half_stride, &" << dst << ".shape, " << dtype << ");\n";
      this->PrintIndent();
      this->stream << "half_stride.w *= 2;\n";
      this->PrintIndent();
      // this->stream << "dim4 half_shape = " << dst << ".shape" << ";\n";
      this->stream << "dim4 half_shape = {.n = " << dst << ".shape.n, .c = " << dst << ".shape.c, .h = " << dst << ".shape.h, .w = " << dst << ".shape.w};\n";
      this->PrintIndent();
      this->stream << "half_shape.w /= 2;\n";
      // tpu_bdc_fp_add( out.addr, x_cos.addr, x_neg_sin.addr + DtypeSize(DT_FP32), &half_shape, &half_stride, &half_stride, &half_stride, DT_FP32);
      // tpu_bdc_fp_add( out.addr + DtypeSize(DT_FP32), x_cos.addr + DtypeSize(DT_FP32), x_sin.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_FP32);
      this->PrintIndent();
      this->stream << "tpu_bdc_fp_add( " << dst << ".addr, " << even_src0 << ".addr, " << even_src1 << ".addr + " << bytes_size << ", " << "&half_shape, " << "&half_stride, " << "&half_stride, " << "&half_stride, " << dtype << ");\n";
      this->PrintIndent();
      this->stream << "tpu_bdc_fp_add( " << dst << ".addr + " << bytes_size << ", " << odd_src0 << ".addr + " << bytes_size << ", " << odd_src1 << ".addr, " << "&half_shape, " << "&half_stride, " << "&half_stride, " << "&half_stride, " << dtype << ");\n";
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "ppl.gather") {
      auto dst_var   = op->args[1].as<CallNode>()->args[1].as<VarNode>();
      auto param_var = op->args[2].as<CallNode>()->args[1].as<VarNode>();
      auto index_var = op->args[3].as<CallNode>()->args[1].as<VarNode>();

      auto dst   = var_idmap_[dst_var];
      if (dst.empty()) dst = this->parameter_map[dst_var->name_hint];
      auto param = var_idmap_[param_var];
      if (param.empty()) param = this->parameter_map[param_var->name_hint];
      auto index = var_idmap_[index_var];
      if (index.empty()) index = this->parameter_map[index_var->name_hint];

      auto param_h = Downcast<IntImm>(op->args[4])->value;
      auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string dtype;
      if (dtype_ == DataType::Float(16)) dtype = "DT_FP16";
      else if (dtype_ == DataType::Float(32)) dtype = "DT_FP32";
      else if (dtype_ == DataType::BFloat(16)) dtype = "DT_BFP16";
      this->PrintIndent();
      this->stream << "{\n";
      this->PrintIndent();
      this->stream << "dim4 __gather_shape = {1, 1, " << dst << ".shape.c, " << dst << ".shape.w};\n";
      this->PrintIndent();
      this->stream << "tpu_gdma_h_gather_S2S("
                   << dst << ".addr, " << param << ".addr, " << index << ".addr, "
                   << "false, (scalar_t){.u32 = 0}, &__gather_shape, " << param_h << ", "
                   << "NULL, NULL, NULL, " << dtype << ");\n";
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "ppl.topk") {
      auto dst_data_var = op->args[1].as<CallNode>()->args[1].as<VarNode>();
      auto dst_idx_var  = op->args[2].as<CallNode>()->args[1].as<VarNode>();
      auto src_var      = op->args[3].as<CallNode>()->args[1].as<VarNode>();

      auto dst_data = var_idmap_[dst_data_var];
      if (dst_data.empty()) dst_data = this->parameter_map[dst_data_var->name_hint];

      auto dst_idx = var_idmap_[dst_idx_var];
      if (dst_idx.empty()) dst_idx = this->parameter_map[dst_idx_var->name_hint];

      auto src = var_idmap_[src_var];
      if (src.empty()) src = this->parameter_map[src_var->name_hint];

      auto K_val = Downcast<IntImm>(op->args[4])->value;
      auto descended_val = Downcast<IntImm>(op->args[5])->value;
      auto length_val = Downcast<IntImm>(op->args[6])->value;

      auto dtype_ = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      std::string dtype;
      // tpu_hau_sort_natural_index 只支持 DT_FP32 / DT_INT32 / DT_UINT32
      if (dtype_ == DataType::Float(32)) dtype = "DT_FP32";
      else if (dtype_ == DataType::Int(32)) dtype = "DT_INT32";
      else if (dtype_ == DataType::UInt(32)) dtype = "DT_UINT32";
      else ICHECK(false) << "ppl.topk: unsupported dtype " << dtype_
                         << "; HAU sort only supports fp32/int32/uint32";

      this->PrintIndent();
      this->stream << "tpu_hau_sort_natural_index(" << dst_data << ".addr, " << dst_idx  << ".addr, " << src << ".addr, " << length_val << ", " << K_val << ", " << (descended_val ? "true" : "false") << ", " << dtype << ");\n";
    }

  } else if (op->op.same_as(builtin::if_then_else())) {
    // conditional that skips eval if cond evals to false
    std::string result = name_supply_->FreshName("condval");
    std::string cond = PrintExpr(op->args[0]);
    this->PrintIndent();

    // remove hard code "shared"
    if (auto var = op->args[1].as<VarNode>()) {
      auto search = buffer_addrs_.find(var);
      if (search != buffer_addrs_.end()) {
        this->stream << "__ppl_tensor_info ";
      }
    } else {
      PrintType(op->dtype, this->stream);
    }

    this->stream << " " << result << ";\n";
    this->PrintIndent();
    this->stream << "if (" << cond << ") {\n";
    {
      int then_scope = this->BeginScope();
      std::string true_val = PrintExpr(op->args[1]);
      this->PrintIndent();
      this->stream << result << " = " << true_val << ";\n";
      this->EndScope(then_scope);
      this->PrintIndent();
      this->stream << "} else {\n";
    }
    {
      int else_scope = this->BeginScope();
      std::string false_val = PrintExpr(op->args[2]);
      this->PrintIndent();
      this->stream << result << " = " << false_val << ";\n";
      this->EndScope(else_scope);
      this->PrintIndent();
      this->stream << "}\n";
    }
    os << result;
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangPPL::VisitStmt_(const LetStmtNode *op) {

  if (op->body.as<Evaluate>()) {
    Evaluate e = Downcast<Evaluate>(op->body);
  }

  std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    ICHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    if (op->var.dtype() == DataType::Handle() &&
        handle_data_type_.count(op->var.get())) {
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "* " << AllocVarID(op->var.get()) << " = (";
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "*)" << value << ";\n";
    } else {

      // TODO: (xwh) Temporary fix
      auto var_node_ = op->var.get();
      std::string var_name = var_node_->name_hint;
      if (var_name.find("shared") != std::string::npos) {
        this->stream << "__ppl_tensor_info " << AllocVarID(op->var.get())
                     << " = " << value << ";\n";
      } else {
        PrintType(op->var.dtype(), this->stream);
        this->stream << ' ' << AllocVarID(op->var.get()) << " = " << value
                     << ";\n";
      }
    }
  }
  PrintStmt(op->body);
}

void CodeGenTileLangPPL::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "tpu_parallel_start") {
    this->PrintIndent();
    this->stream << "tpu_parallel_start(); \n";

  } else if (op->attr_key == "tpu_parallel_end") {
    this->PrintIndent();
    this->stream << "tpu_parallel_end(); \n";
  }
  this->PrintStmt(op->body);
}

std::string CodeGenTileLangPPL::AllocLocalVarID(const tir::VarNode *v) {
  // ICHECK(!local_buffer_name_map.count(v)) << "Need input to be in SSA form
  // dup " << v->name_hint;
  std::string key = v->name_hint;
  std::string vid = name_supply_->FreshName(key);
  std::replace(vid.begin(), vid.end(), ':', '_');
  std::replace(vid.begin(), vid.end(), '-', '_');
  std::replace(vid.begin(), vid.end(), '.', '_');
  // var_idmap_[v].push_back()
  local_buffer_name_map[v].push_back(vid);
  return vid;
}

static inline std::string
Shape4ToDim4Literal(const std::vector<int64_t> &shape) {
  ICHECK_EQ(shape.size(), 4U);
  std::ostringstream os;
  os << "{ " << shape[0] << ", " << shape[1] << ", " << shape[2] << ", "
     << shape[3] << "}";
  return os.str();
}

void CodeGenTileLangPPL::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  const tir::VarNode *buffer_var = op->buffer_var.get();
  auto old_var_it = var_idmap_.find(buffer_var);
  bool had_old_var = old_var_it != var_idmap_.end();
  std::string old_var_id;
  if (had_old_var) {
    old_var_id = old_var_it->second;
  }
  std::string vid = AllocLocalVarID(buffer_var);
  var_idmap_[buffer_var] = vid;

  auto shape4 = tl::bm1690::NormalizeLocalShape(op->extents, "PPL codegen");
  std::string bv_shape = Shape4ToDim4Literal(shape4);
  std::vector<int> shapes;
  shapes.push_back(static_cast<int>(shape4[1]));
  shapes.push_back(static_cast<int>(shape4[3]));
  std::string op_dtype = TargetDTypeName(op->dtype);
  int64_t tensor_size =
      tl::bm1690::TpuAlignSizeBytesFromShape4(shape4, op->dtype);
  ICHECK_LE(tensor_size, std::numeric_limits<int>::max());
  this->PrintIndent();
  auto addr =
      f_attrs.GetAttr(buffer_var->name_hint, PrimExpr(0)).as<IntImmNode>()->value;
  buffer_addrs_[buffer_var] = addr;
  stream << "__ppl_tensor_info " << vid << " = {.shape = " << bv_shape
         << ", .stride = {0}"
         << ", .addr = " << addr << ", .dtype = " << op_dtype << ", .mode = 2"
         << ", .align_mode = 1"
         << ", .size = " << tensor_size
         << ", .unsigned_flag = 0, .default_stride = false};\n";
  this->PrintIndent();
  stream << "tpu_aligned_stride(&" << vid << ".stride, 0, &" << vid
         << ".shape, " << op_dtype << ");\n";
  this->buffer_shape[vid] = shapes;
  // store local tensor shape

  this->PrintStmt(op->body);

  if (had_old_var) {
    var_idmap_[buffer_var] = old_var_id;
  } else {
    var_idmap_.erase(buffer_var);
  }
}

void CodeGenTileLangPPL::VisitExpr_(const RampNode *op, std::ostream &os) {
  //  int lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
  //  CHECK_LE(lanes, 4) << "ValueError: Ramp of more than 4 lanes is not
  //  allowed."; os << "(make_"; PrintType(op->dtype, os); os << "("; for (int i
  //  = 0; i < lanes; i++) {
  //    os << "(" << PrintExpr(op->base) << ")"
  //       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
  //    if (i != lanes - 1) os << ", ";
  //  }
  //  os << "))";
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangPPL *p) { // NOLINT(*)
  // Type code is kBFloat
  if (op->dtype.is_bfloat16()) {
    os << "bfloat16_t";
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  }
  // Type code is kFloat
  switch (op->dtype.bits()) {
  case 64:
  case 32: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << ((op->dtype.bits() == 32) ? "CUDART_INF_F" : "CUDART_INF");
    } else if (std::isnan(op->value)) {
      temp << ((op->dtype.bits() == 32) ? "CUDART_NAN_F" : "CUDART_NAN");
    } else {
      temp << std::scientific << op->value;
      if (op->dtype.bits() == 32)
        temp << 'f';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    break;
  }
  case 16: {
    os << "half_t" << '(';
    FloatImm const_f32 = FloatImm(DataType::Float(32), op->value);
    PrintConst(const_f32.get(), os, p);
    os << ')';
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenTileLangPPL::VisitExpr_(const FloatImmNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

template <typename T>
inline void PrintBinaryExpr(const T *op, const char *opstr,
                            std::ostream &os, // NOLINT(*)
                            CodeGenC *p) {
  if (op->dtype.lanes() == 1) {
    if (isalpha(opstr[0])) {
      os << opstr << '(';
      p->PrintExpr(op->a, os);
      os << ", ";
      p->PrintExpr(op->b, os);
      os << ')';
    } else {
      os << '(';
      p->PrintExpr(op->a, os);
      os << ' ' << opstr << ' ';
      p->PrintExpr(op->b, os);
      os << ')';
    }
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

void CodeGenTileLangPPL::VisitExpr_(const FloorModNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}

void CodeGenTileLangPPL::VisitExpr_(const FloorDivNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr(op, "/", os, this);
}

void CodeGenTileLangPPL::PrintWmmaScope(const std::string &scope, DataType t,
                                        const VarNode *variable,
                                        std::ostream &os) {}

int32_t CodeGenTileLangPPL::GetWmmaFragmentSize(const std::string &scope,
                                                const VarNode *variable,
                                                int32_t size) {
  return 0;
}

void CodeGenTileLangPPL::HandleVolatileLoads(const std::string &value,
                                             const BufferLoadNode *op,
                                             std::ostream &os) {}

void CodeGenTileLangPPL::PrintVecElemLoadExpr(DataType t, int i,
                                              const std::string &value,
                                              std::ostream &os) {
  return;
}


void CodeGenTileLangPPL::AddFunction(const PrimFunc &f) {
  this->InitFuncState(f);
  ReserveKeywordsAsUnique();
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  f_attrs = f->attrs;
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  auto buffer_map = f->buffer_map;
  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  this->PrintExtraAttrs(f, stream);
  // 获取kernel名
  std::string global_name = static_cast<std::string>(global_symbol.value());
  if (global_name == "main") {
    throw std::runtime_error("Kernel name 'main' is not allowed. Please use 'main_kernel_inner' as the kernel name.");
  }
  this->stream << " " << global_name << "(";
  std::vector<std::string> params_name; // 生成参数列表
  std::vector<std::string> params_type;
  // auto bf_map = f->buffer_map;
  std::unordered_map<const tir::VarNode *, std::string> var_global_mem_map;

  //根据shape自动计算contiguous stride
  auto default_stride = [this](const std::string &node) {
    auto buf_shape = buffer_shape[node];
    buffer_stride[node] = {1, 1, 1, 1}; // 初始化为1
    // 倒推
    for (int i = 2; i >= 0; i--) {
      buffer_stride[node][i] = buf_shape[i + 1] * buffer_stride[node][i + 1];
    }
  };

  // Don't use name hint for tensor arguments, but can remove later.
  auto allocate_buffer_name = [&, this](const Var &v, int index, int length) {
    auto v_node = v.get();
    std::string vid = "v" + std::to_string(index + 1);
    std::string rid = "v" + std::to_string(index + 1 + length);

    auto buffer_node = buffer_map[v];
    auto shape = buffer_node->shape;
    auto dim4_shape = LowerGlobalShapeToDim4(shape);
    buffer_shape[buffer_node->name] = dim4_shape;
    default_stride(buffer_node->name);
    std::string shape_s = vector2string(dim4_shape);

    std::string dtype = TargetDTypeName(buffer_node->dtype);
    int bytes_size = TargetDTypeBytes(buffer_node->dtype);
    int64_t tensor_size = 1;
    for (int dim : dim4_shape) {
      tensor_size *= dim;
    }
    tensor_size *= bytes_size;
    std::string inst =
        "__ppl_tensor_info " + rid + " = {.shape = " + shape_s +
        ", .stride = {0}, .addr = " + vid + ", .dtype = " + dtype +
        ", .mode = 2, .align_mode = 0, .size = " + std::to_string(tensor_size) +
        ", .unsigned_flag = 0, .default_stride = true};\n";
    var_global_mem_map[v_node] = inst;
    std::string name_hint = v_node->name_hint;
    this->var_idmap_[v_node] = rid;

    // 参数名处理：remove "_handle"
    for (int i{0}; i < 7; i++) {
      name_hint.pop_back();
    }
    this->parameter_map[name_hint] = rid;
    return vid;
  };
  int param_len = f->params.size();
  for (size_t i = 0; i < param_len; ++i) {
    tir::Var v = f->params[i];
    std::string vid = "v" + std::to_string(i + 1);
    std::string param_type;
    if (buffer_map.count(v)) {
      vid = allocate_buffer_name(v, i, param_len);
      param_type = restrict_keyword_;
    } else {
      // Sophgo kernel APIs carry runtime values such as seqlen/slot inline in
      // the launch argument struct. Preserve scalar PrimFunc parameters as C
      // scalars instead of treating every argument as a global tensor address.
      std::ostringstream type_stream;
      this->PrintType(v.dtype(), type_stream);
      param_type = type_stream.str();
      this->var_idmap_[v.get()] = vid;
    }
    params_name.push_back(vid);
    params_type.push_back(param_type);
    if (i != 0)
      stream << ", ";
    stream << param_type << ' ' << vid;  // 输出函数参数
  }
  stream << ") {\n";

  this->PreFunctionBody(f); // none
  int func_scope = this->BeginScope();

  // 输出tensor_info
  for (auto [v, inst] : var_global_mem_map) {
    this->PrintIndent();
    this->stream << inst;
  }
  // 真正生成TIR body
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";

  //生成Runtime Wrapper
  // TPU runtime接口层
  this->stream << "typedef struct {\n";
  for (size_t i = 0; i < params_name.size(); ++i) {
    this->stream << "  " << params_type[i] << " " << params_name[i] << ";\n";
  }
  std::string api_name = "tpu_kernel_api_main_inner_args_t";
  this->stream << "} " << api_name << ";\n";
  // TPU runtime实际调用入口
  this->stream << "int "
               << "main_kernel(const void * args) {\n"
               << "  " << api_name << " *api = (" << api_name << "*)args;\n"
               << "  tpu_initialize();\n"
               << "  " << global_name << "(";
  int name_index = 0;
  int name_len = params_name.size();
  for (auto &name : params_name) {
    if (name_index != 0)
      this->stream << "    ";
    // 调用真正的kernel
    this->stream << "api->" << name;

    if (name_index == name_len - 1)
      this->stream << ");";
    else
      this->stream << ",";
    this->stream << "\n";
    name_index += 1;
  }
  // poll——等待TPU完成
  this->stream << "  tpu_poll();\n"
               << "  return 0;\n}\n";
  // 注册kernel给runtime
  this->stream << "TPUKERNEL_FUNC_REGISTER(" << "main_kernel)\n";
}

} // namespace codegen
} // namespace tvm
