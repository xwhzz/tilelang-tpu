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

#include <cmath>
#include <string>
#include <utility>
#include <vector>

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
  decl_stream << "static data_type_t __ppl_get_dtype(int type) {\n"
              << "  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  "
                 "DT_BFP16,\n"
              << "    DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,\n"
              << "    DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,\n"
              << "    DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};\n"
              << "  return __dtype[type];\n"
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
  PrintStmt(op->body);
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

static inline const char* AsBDTypeStr(const DataType& dtype_) {
  if (dtype_ == DataType::Float(32)) {
    return "DT_FP32";
  } else if (dtype_ == DataType::Float(16)) {
    return "DT_FP16";
  } else if (dtype_ == DataType::BFloat(16)) {
    return "DT_BFP16";
  }

  // 其它类型回退为FP32
  return "DT_FP32";
}

void CodeGenTileLangPPL::VisitExpr_(const CallNode *op, std::ostream &os) {
  auto process_stride = [&,
                         this](const std::vector<int> &src0_shape,
                               const std::vector<int> &src1_shape,
                               const std::string &src0, const std::string &src1,
                               const std::string &dtype) -> std::stringstream {
    std::stringstream src1_stride;
    if (src1_shape[1] == 1 && src0_shape[1] != 1) {
      // void tpu_aligned_stride(dim4 *stride, int start_idx, const dim4 *shape,
      // data_type_t dtype) we must construct explicit stride
      // 1. initialize a dim4 struct
      this->PrintIndent();
      this->stream << "dim4 " << src1 << "_stride;\n";
      // 2. call tpu_aligned_stride
      this->PrintIndent();
      this->stream << "tpu_aligned_stride(&" << src1 << "_stride, 0, &" << src1
                   << ".shape, " << dtype << ");\n";
      this->PrintIndent();
      this->stream << src1 << "_stride.w = 0;\n";
      src1_stride << "&" << src1 << "_stride, ";
    } else if (src1_shape[1] == src0_shape[1]) {
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
    std::string dtype;
    if (dtype_ == DataType::Float(16)) {
      dtype = "DT_FP16";
    } else if (dtype_ == DataType::Float(32)) {
      dtype = "DT_FP32";
    }
    else if (dtype_ == DataType::BFloat(16)){
      dtype = "DT_BFP16";
    }
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
  // *src_stride, data_type_t dtype)
  auto handle_elementwise_const = [&, this](std::string op_name) {
    auto dst = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
    auto src0 = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
    float value = Downcast<FloatImm>(op->args[3])->value;
    auto src0_shape = buffer_shape[src0];
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
                  << "(" << dst << ".default_stride ? NULL : &" << dst
                  << ".stride), " << dtype << ");\n";
    }
    else{
      this->PrintIndent();
      this->stream << "scalar_t " << dst << "_scalar_" << dtype << " = {.f32 = " << "(float)" << value << "};\n";
      this->PrintIndent();
      // x_neg_scalar_DT_BFP16 = tpu_cast(x_neg_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
      this->stream << dst << "_scalar_" << dtype << " = tpu_cast(" << dst << "_scalar_" << dtype << ", " << dtype << ", DT_FP32, RM_HALF_TO_EVEN);\n";
      this->PrintIndent();
      this->stream << op_name << "( " << dst << ".addr, " << src0 << ".addr, " <<  dst << "_scalar_" << dtype << ", " << "&" << dst << ".shape, " << "(" << dst << ".default_stride ? NULL : &" << dst << ".stride), " << "(" << src0 << ".default_stride ? NULL : &" << src0 << ".stride), " <<  dtype << ");\n";
    }
  };
  std::vector<std::string> inst;
  if (op->op.same_as(builtin::call_extern())) {
    std::string op_name = Downcast<StringImm>(op->args[0])->value;
    if (op_name == "ppl.copy") {
      tl::BufferMap buffer_map;
      auto process_copy = [&, this](const tl::RegionOp &src)
          -> std::tuple<std::string, std::string, std::string> {
        auto src_buffer = src.GetBuffer();
        auto src_ranges = src.GetRanges();

        auto src_id = var_idmap_[src_buffer->data.get()];
        if (src_id.empty()) {
          src_id = this->parameter_map[src_buffer->name];
        }
        std::string src_shape;
        std::string new_src_var =
            name_supply_->FreshName(src_buffer->data->name_hint);
        int i = 0;
        if (src_ranges.size() == 2) {
          src_shape =
              "{1, " +
              std::to_string(src_ranges[i]->extent.as<IntImmNode>()->value) +
              ", 1, " +
              std::to_string(
                  src_ranges[i + 1]->extent.as<IntImmNode>()->value) +
              "}";
        } else if (src_ranges.size() == 4) {
          src_shape = "{";
          for (auto &sr : src_ranges) {
            src_shape +=
                std::to_string(sr->extent.as<IntImmNode>()->value) + ", ";
          }
          src_shape[src_shape.size() - 2] = '}';
        }
        std::string dtype;
        int bytes_size = 0;
        if (src_buffer->dtype == DataType::Float(16)) {
          dtype = "DT_FP16";
          bytes_size = 2;
        } else if (src_buffer->dtype == DataType::BFloat(16)) {
          dtype = "DT_BFP16";
          bytes_size = 2;
        } else if (src_buffer->dtype == DataType::Float(32)) {
          dtype = "DT_FP32";
          bytes_size = 4;
        } else {
          LOG(FATAL) << "Unsupported dtype " << src_buffer->dtype;
        }
        if (src_buffer.scope() == "global") {
          std::string src_strides;

          auto strides = buffer_stride[src_buffer->name];
          src_strides = vector2string(strides);
          std::string min_expr;
          std::vector<int> stride_map = {1, 3};
          for (int i = 0; i < src_ranges.size(); i++) {
            auto sr = src_ranges[i];
            const PrimExpr &e = sr->min;
            std::string idx_str;
            if (const RampNode* ramp = e.as<RampNode>()) { // 如果是Ramp，只取base部分
              idx_str = PrintExpr(ramp->base);
            } else { // 否则直接打印整个表达式
              idx_str = PrintExpr(e);
            }
            min_expr +=
                "(" + idx_str + ") * " + std::to_string(strides[stride_map[i]]) + "+";
          }
          min_expr[min_expr.size() - 1] = ' ';
          min_expr = "(" + min_expr + ")" + " * " + std::to_string(bytes_size);
          std::cout << "min_expr: " << min_expr << std::endl;
          inst.push_back("__ppl_tensor_info " + new_src_var +
                         " = {.shape = " + src_shape +
                         ", .stride = " + src_strides + ", .addr = " + src_id +
                         ".addr + " + min_expr + ", .dtype = " + dtype +
                         ", .mode = 2, .size = 1, .offset = " + min_expr +
                         ", .unsigned_flag = 0, .default_stride = false};\n");
        } else if (src_buffer.scope() == "shared.dyn") {

          inst.push_back(
              "__ppl_tensor_info " + new_src_var + " = {.shape = " + src_shape +
              ", .stride = NULL, .addr = " +
              var_idmap_[src_buffer->data.get()] + ".addr, .dtype = " + dtype +
              ", .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, "
              ".default_stride = true};\n");
        }
        return std::make_tuple(new_src_var, src_buffer.scope(), dtype);
      };
      tvm::Dump(op);
      tl::RegionOp src =
          tl::RegionOp(op->args[1].as<CallNode>()->args, buffer_map);
      // tvm::Dump(src);
      tl::RegionOp dst =
          tl::RegionOp(op->args[2].as<CallNode>()->args, buffer_map);
      auto [src_var_id, src_flag, src_dtype] = process_copy(src);
      auto [dst_var_id, dst_flag, dst_dtype] = process_copy(dst);
      std::string ppl_inst;
      if (src_dtype != dst_dtype) {
        // void tpu_bdc_cast(local_addr_t dst_addr, local_addr_t src_addr, const
        // dim4 *shape, const dim4 *dst_stride, const dim4 *src_stride,
        // data_type_t dst_dtype, data_type_t src_dtype, rounding_mode_t mode)
        // 使用RM_HALF_TO_EVEN舍入模式，只有在浮点数据类型参与的转换时使用
        ppl_inst += "tpu_bdc_cast(" + dst_var_id + ".addr, " + src_var_id +
                    ".addr, " + "&" + dst_var_id + ".shape, " + "(" +
                    dst_var_id + ".default_stride ? NULL : &" + dst_var_id +
                    ".stride), " + "(" + src_var_id +
                    ".default_stride ? NULL : &" + src_var_id + ".stride), " +
                    dst_dtype + ", " + src_dtype + ", " + "RM_HALF_TO_EVEN" +
                    ");\n";
        inst.push_back(ppl_inst);
        for (auto &i : inst) {
          this->PrintIndent();
          this->stream << i;
        }
      } else {
        if (src_flag == "global" && dst_flag == "shared.dyn") {
          ppl_inst += "tpu_gdma_cpy_S2L";
        } else if (src_flag == "shared.dyn" && dst_flag == "global") {
          ppl_inst += "tpu_gdma_cpy_L2S";
        } else {
          // local mem -> local mem copy within the same NPU
          ppl_inst += "tpu_bdc_cpy";
        }

        ppl_inst += "(" + dst_var_id + ".addr, " + src_var_id + ".addr, &" +
                    dst_var_id + ".shape, " + "(" + dst_var_id +
                    ".default_stride ? NULL : &" + dst_var_id + ".stride), " +
                    "(" + src_var_id + ".default_stride ? NULL : &" +
                    src_var_id + ".stride), " + src_dtype + ");\n";
        inst.push_back(ppl_inst);
        for (auto &i : inst) {
          this->PrintIndent();
          this->stream << i;
        }
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
      int value = Downcast<FloatImm>(op->args[2])->value;
      this->PrintIndent();
      this->stream << "scalar_t " << data_ << "_scalar_" << dtype_1 << " = {."
                   << dtype_1 << " = " << value << "};\n";
      this->PrintIndent();
      this->stream << "tpu_bdc_set_C(" << data_ << ".addr, " << data_
                   << "_scalar_" << dtype_1 << ", &" << data_ << ".shape, ("
                   << data_ << ".default_stride ? NULL : &" << data_
                   << ".stride), " << dtype_2 << ");\n";
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
      // 非转置路径：累加需要数据类型为FP32
      const char* output_dtype_accum = "DT_FP32"; 
      // 转置路径：若C与A/B同精度则可用同精度，否则用FP32
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
      }

      this->PrintIndent();
      int sid = this->BeginScope();
      this->stream << "{\n";

      // 计算EU数和对齐尺寸
      this->PrintIndent();
      this->stream << "int eu_num = " << eu_num << ";\n";
      this->PrintIndent();
      this->stream << "int align_w = " << align_w << ";\n";

      // 创建pad_val
      this->PrintIndent();
      this->stream << "scalar_t pad_val = {."
                   << (dtype_ == DataType::Float(16) ? "f16" : "f16")
                   << " = FP_NEG_MAX(" << dtype << ")};\n";

      // 判断是否需要填充 - 只有在宽度不是EU数的倍数时才需要填充
      this->PrintIndent();
      this->stream << "if (align_w > " << input_tensor << ".shape.w) {\n";

      // 计算填充区域大小和偏移
      this->PrintIndent();
      this->stream << "  dim4 fill_shape = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, align_w - " << input_tensor
                   << ".shape.w};\n";
      this->PrintIndent();
      this->stream << "  int elem_size = "
                   << (dtype_ == DataType::Float(16) ? "2" : "4") << ";\n";
      this->PrintIndent();
      this->stream << "  int offset = " << input_tensor
                   << ".shape.w * elem_size;\n";
      this->PrintIndent();
      this->stream << "  dim4 fill_tensor_stride = {" << stride_n
                   << ", align_w, " << input_tensor << ".shape.w, 1};\n";

      // 创建填充视图
      this->PrintIndent();
      this->stream
          << "  __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride "
             "= fill_tensor_stride, "
          << ".addr = " << input_tensor << ".addr + offset, .dtype = " << dtype
          << ", "
          << ".mode = 0, .align_mode = 4, .size = 1, .offset = offset, "
          << ".unsigned_flag = 0, .default_stride = false};\n";

      // 填充
      this->PrintIndent();
      this->stream
          << "  tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, "
          << "(fill_tensor.default_stride ? NULL : &fill_tensor.stride), "
          << dtype << ");\n";

      this->PrintIndent();
      this->stream << "}\n";

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

      // 创建输入和临时视图
      this->PrintIndent();
      this->stream << "__ppl_tensor_info input_view = {.shape = in_reduce_h, "
                      ".stride = {0}, "
                   << ".addr = " << input_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";

      this->PrintIndent();
      this->stream << "__ppl_tensor_info tmp_view = {.shape = out_reduce_h, "
                      ".stride = {0}, "
                   << ".addr = " << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";

      // 第一次最大池化
      this->PrintIndent();
      this->stream << "tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, "
                      "&input_view.shape, "
                   << "&kernel, &pad, &stride, &dilation, " << dtype
                   << ", pad_val);\n";

      // 修改kernel大小
      this->PrintIndent();
      this->stream << "dim2 kernel2 = {1, eu_num};\n";

      // 创建输出视图 - 更紧凑的格式
      this->PrintIndent();
      this->stream << "__ppl_tensor_info output_view = {.shape = out_reduce_w, "
                      ".stride = {0}, "
                   << ".addr = " << output_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";

      this->PrintIndent();
      this->stream << "__ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, "
                      ".stride = {0}, "
                   << ".addr = " << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";

      // 重置pad_val
      this->PrintIndent();
      this->stream << "pad_val.u32 = FP_NEG_MAX(" << dtype << ");\n";

      // 第二次最大池化
      this->PrintIndent();
      this->stream << "tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, "
                      "&tmp_view2.shape, "
                   << "&kernel2, &pad, &stride, &dilation, " << dtype
                   << ", pad_val);\n";

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

      // 判断是否需要填充 - 只有在宽度不是EU数的倍数时才需要填充
      this->PrintIndent();
      this->stream << "if (align_w > " << input_tensor << ".shape.w) {\n";

      // 计算填充区域大小和偏移
      this->PrintIndent();
      this->stream << "  dim4 fill_shape = {" << input_tensor << ".shape.n, "
                   << input_tensor << ".shape.c, 1, align_w - " << input_tensor
                   << ".shape.w};\n";
      this->PrintIndent();
      int elem_size = 1;
      if (dtype_ == DataType::Float(16)) {
        elem_size = 2;
      } else if (dtype_ == DataType::Float(32)) {
        elem_size = 4;
      } else if (dtype_ == DataType::Int(32)) {
        elem_size = 4;
      } else if (dtype_ == DataType::UInt(32)) {
        elem_size = 4;
      } else if (dtype_ == DataType::Int(16)) {
        elem_size = 2;
      } else {
        LOG(FATAL) << "Unsupported dtype " << dtype_;
      }
      this->stream << "  int elem_size = " << elem_size << ";\n";
      this->PrintIndent();
      this->stream << "  int offset = " << input_tensor
                   << ".shape.w * elem_size;\n";
      this->PrintIndent();
      this->stream << "  dim4 fill_tensor_stride = {" << stride_n
                   << ", align_w, " << input_tensor << ".shape.w, 1};\n";

      // 创建填充视图
      this->PrintIndent();
      this->stream
          << "  __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride "
             "= fill_tensor_stride, "
          << ".addr = " << input_tensor << ".addr + offset, .dtype = " << dtype
          << ", "
          << ".mode = 0, .align_mode = 4, .size = 1, .offset = offset, "
          << ".unsigned_flag = 0, .default_stride = false};\n";

      // 填充
      this->PrintIndent();
      this->stream
          << "  tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, "
          << "(fill_tensor.default_stride ? NULL : &fill_tensor.stride), "
          << dtype << ");\n";

      this->PrintIndent();
      this->stream << "}\n";

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

      // 创建输入和临时视图
      this->PrintIndent();
      this->stream << "__ppl_tensor_info input_view = {.shape = in_reduce_h, "
                      ".stride = {0}, "
                   << ".addr = " << input_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";

      this->PrintIndent();
      this->stream << "__ppl_tensor_info tmp_view = {.shape = out_reduce_h, "
                      ".stride = {0}, "
                   << ".addr = " << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";

      this->PrintIndent();
      this->stream << "scalar_t scale = {.f32 = (float)1.000000000e+00};\n";
      if (dtype_ == DataType::Float(16)) {
        this->PrintIndent();
        this->stream
            << "scale = tpu_cast(scale, DT_FP16, DT_FP32, RM_HALF_TO_EVEN);\n";
      } else if (dtype_ == DataType::Int(32)) {
        this->PrintIndent();
        this->stream << "scale = tpu_cast(scale, DT_INT32, DT_FP32, "
                        "RM_HALF_TO_EVEN);\n"; // todo
      } else if (dtype_ == DataType::UInt(32)) {
        this->PrintIndent();
        this->stream << "scale = tpu_cast(scale, DT_UINT32, DT_FP32, "
                        "RM_HALF_TO_EVEN);\n"; // todo
      }
      // 第一次均值池化（通过设置scale实现求和）
      this->PrintIndent();
      this->stream << "tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, "
                      "&input_view.shape, "
                   << "&kernel, &pad, &stride, &dilation, " << dtype
                   << ", scale);\n";

      // 修改kernel大小
      this->PrintIndent();
      this->stream << "dim2 kernel2 = {1, eu_num};\n";

      // 创建输出视图 - 更紧凑的格式
      this->PrintIndent();
      this->stream << "__ppl_tensor_info output_view = {.shape = out_reduce_w, "
                      ".stride = {0}, "
                   << ".addr = " << output_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";

      this->PrintIndent();
      this->stream << "__ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, "
                      ".stride = {0}, "
                   << ".addr = " << tmp_tensor << ".addr, .dtype = " << dtype
                   << ", "
                   << ".mode = 0, .align_mode = 1, .size = 1, .offset = 0, "
                   << ".unsigned_flag = 0, .default_stride = true};\n";

      // 第二次均值池化（通过设置scale实现求和）
      this->PrintIndent();
      this->stream << "tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, "
                      "&tmp_view2.shape, "
                   << "&kernel2, &pad, &stride, &dilation, " << dtype
                   << ", scale);\n";
      // this->indent_ -= 2;

      // 结束块作用域
      this->EndScope(sid);
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "ppl.rsqrt") {
      auto dst = var_idmap_[op->args[1].as<CallNode>()->args[1].as<VarNode>()];
      auto src0 = var_idmap_[op->args[2].as<CallNode>()->args[1].as<VarNode>()];
      auto src0_shape = buffer_shape[src0];
      // void tpu_bdc_fp32_rsqrt(local_addr_t dst_addr, local_addr_t src_addr,
      // const dim4 *shape)
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
      this->stream << "dim4 half_stride;\n";
      this->PrintIndent();
      this->stream << "tpu_aligned_stride(&half_stride, 0, &" << dst << ".shape, " << dtype << ");\n";
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

void CodeGenTileLangPPL::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  auto buffer_shape = op->extents;
  if (buffer_shape.size() == 2)
    buffer_shape.insert(buffer_shape.begin(), make_const(DataType::Int(32), 1));
  std::string bv_shape = "{ 1, ";
  std::vector<int> shapes;
  shapes.push_back(buffer_shape[1].as<IntImmNode>()->value);
  shapes.push_back(buffer_shape[2].as<IntImmNode>()->value);
  bv_shape += std::to_string(buffer_shape[1].as<IntImmNode>()->value);
  bv_shape += ", 1, ";
  bv_shape += std::to_string(buffer_shape[2].as<IntImmNode>()->value);
  bv_shape += "}";
  std::string op_dtype;
  int bytes_size = 0;
  if (op->dtype == DataType::Float(16)) {
    op_dtype = "DT_FP16";
    bytes_size = 2;
  } else if (op->dtype == DataType::Float(32)) {
    op_dtype = "DT_FP32";
    bytes_size = 4;
  } else if (op->dtype == DataType::BFloat(16)){
    op_dtype = "DT_BFP16";
    bytes_size = 2;
  }
  auto buffer_num = buffer_shape[0].as<IntImmNode>()->value;
  for (size_t iter{0}; iter < buffer_num; iter++) {
    std::string vid = AllocVarID(op->buffer_var.get());
    this->PrintIndent();
    int tensor_size = shapes[0] * shapes[1] / lane_num * bytes_size;
    auto addr = f_attrs.GetAttr(vid, PrimExpr(0)).as<IntImmNode>()->value;
    buffer_addrs_[op->buffer_var.get()] = addr;
    stream << "__ppl_tensor_info " << vid << " = {.shape = " << bv_shape
           << ", .stride = NULL"
           << ", .addr = " << addr << ", .dtype = " << op_dtype << ", .mode = 2"
           << ", .align_mode = 1"
           << ", .size = " << tensor_size
           << ", .unsigned_flag = 0, .default_stride = true};\n";
    this->buffer_shape[vid] = shapes;
    // store local tensor shape
  }

  this->PrintStmt(op->body);
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
  std::string global_name = static_cast<std::string>(global_symbol.value());

  this->stream << " " << global_name << "(";
  std::vector<std::string> params_name;
  // auto bf_map = f->buffer_map;
  std::unordered_map<const tir::VarNode *, std::string> var_global_mem_map;

  auto default_stride = [this](const std::string &node) {
    auto buf_shape = buffer_shape[node];
    buffer_stride[node] = {1, 1, 1, 1};
    for (int i = 2; i >= 0; i--) {
      buffer_stride[node][i] = buf_shape[i + 1] * buffer_stride[node][i + 1];
    }
  };

  // don't use name hint, but can remove later.
  auto allocate_name = [&, this](const Var &v, int index, int length) {
    auto v_node = v.get();
    std::string vid = "v" + std::to_string(index + 1);
    std::string rid = "v" + std::to_string(index + 1 + length);

    auto buffer_node = buffer_map[v];
    auto shape = buffer_node->shape;

    std::string shape_s = "{";
    int tensor_size = 1;
    if (shape.size() == 2) {
      buffer_shape[buffer_node->name] = {1, shape[0].as<IntImmNode>()->value, 1,
                                         shape[1].as<IntImmNode>()->value};
      default_stride(buffer_node->name);
      shape_s += "1 ,";
      shape_s += std::to_string(shape[0].as<IntImmNode>()->value);
      tensor_size *= shape[0].as<IntImmNode>()->value;
      shape_s += ", 1, ";
      shape_s += std::to_string(shape[1].as<IntImmNode>()->value);
      tensor_size *= shape[1].as<IntImmNode>()->value;
    } else if (shape.size() == 4) {
      buffer_shape[buffer_node->name] = {};
      for (auto s : shape) {
        buffer_shape[buffer_node->name].push_back(s.as<IntImmNode>()->value);
      }
      default_stride(buffer_node->name);
      // 用下标循环来拼接带逗号的字符串
      for (size_t i = 0; i < shape.size(); ++i) {
        int dim_i = shape[i].as<IntImmNode>()->value;
        shape_s += std::to_string(dim_i);
        tensor_size *= dim_i;
        if (i + 1 < shape.size()) {
          shape_s += ", ";
        }
      }
    }

    shape_s += "}";
    std::string dtype;
    int bytes_size = 0;

    if (buffer_node->dtype == DataType::Float(16)) {
      dtype = "DT_FP16";
      bytes_size = 2;
    } else if (buffer_node->dtype == DataType::Float(32)) {
      dtype = "DT_FP32";
      bytes_size = 4;
    } else if (buffer_node->dtype == DataType::BFloat(16)){
      dtype = "DT_BFP16";
      bytes_size = 2;
    }
    tensor_size *= bytes_size;
    std::string inst =
        "__ppl_tensor_info " + rid + " = {.shape = " + shape_s +
        ", .stride = NULL, .addr = " + vid + ", .dtype = " + dtype +
        ", .mode = 2, .align_mode = 0, .size = " + std::to_string(tensor_size) +
        ", .unsigned_flag = 0, .default_stride = true};\n";
    var_global_mem_map[v_node] = inst;
    std::string name_hint = v_node->name_hint;
    this->var_idmap_[v_node] = rid;

    // remove "_handle"
    for (int i{0}; i < 7; i++) {
      name_hint.pop_back();
    }
    this->parameter_map[name_hint] = rid;
    return vid;
  };
  int param_len = f->params.size();
  for (size_t i = 0; i < param_len; ++i) {
    tir::Var v = f->params[i];
    std::string vid = allocate_name(v, i, param_len);
    params_name.push_back(vid);
    if (i != 0)
      stream << ", ";
    stream << restrict_keyword_ << ' ' << vid;
  }
  stream << ") {\n";

  this->PreFunctionBody(f); // none
  int func_scope = this->BeginScope();

  for (auto [v, inst] : var_global_mem_map) {
    this->PrintIndent();
    this->stream << inst;
  }
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";

  this->stream << "typedef struct {\n";
  for (auto &name : params_name) {
    this->stream << "  " << restrict_keyword_ << " " << name << ";\n";
  }
  std::string api_name = "tpu_kernel_api_" + global_name + "_args_t";
  this->stream << "} " << api_name << ";\n";
  this->stream << "void " << global_name << "_kernel(const void * args) {\n"
               << "  " << api_name << " *api = (" << api_name << "*)args;\n"
               << "  " << global_name << "(";
  int name_index = 0;
  int name_len = params_name.size();
  for (auto &name : params_name) {
    if (name_index != 0)
      this->stream << "    ";
    this->stream << "api->" << name;

    if (name_index == name_len - 1)
      this->stream << ");";
    else
      this->stream << ",";
    this->stream << "\n";
    name_index += 1;
  }
  this->stream << "  tpu_poll();\n}\n";
  this->stream << "TPUKERNEL_FUNC_REGISTER(" << global_name << "_kernel)\n";
}

} // namespace codegen
} // namespace tvm
