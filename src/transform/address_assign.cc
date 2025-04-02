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
#include <tvm/relay/expr.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

// #include <map>
#include <unordered_map>
#include <unordered_set>

#include "../op/builtin.h"
#include "../op/bulk_copy.h"
#include "../op/gemm.h"

namespace tvm {
namespace tl {
using namespace tir;

class AddressAllocator : public StmtExprVisitor {
  public:
  explicit AddressAllocator() {}

  std::vector<const BufferNode*> collectAllocOp(tir::Stmt body) {
    this->VisitStmt(body);
    return alloc_ops_;
  }

  void VisitStmt_(const DeclBufferNode* op) {
    alloc_ops_.emplace_back(op->buffer.get());
    this->VisitStmt(op->body);
  }


  private:
  std::vector<const BufferNode*> alloc_ops_;

};

struct TensorLive {
  int start, end;
  uint32_t tensor_size;
};

struct OpAddr {
  const BufferNode* op;
  int64_t start = 0;
  int64_t end = 0;
  uint32_t size = 0;
  uint32_t first_pos = 0;
  uint32_t end_pos = 0;

  OpAddr(const BufferNode* _op, uint32_t _size, uint32_t _first_pos, uint32_t _end_pos) {
    op = _op;
    size = _size;
    first_pos = _first_pos;
    end_pos = _end_pos;
  }
};

class MemAllocBankConflictAware {
  public:
    MemAllocBankConflictAware(int64_t bank_num, int64_t bank_size)
      : bank_num_(bank_num), bank_size_(bank_size) {
      total_consumption_ = 0;
      mem_size_ = bank_num * bank_size;
      bank_ops.resize(bank_num);
    }
  
    bool assignAddr(std::vector<const BufferNode*> &ops,
                    std::unordered_map<const BufferNode*, TensorLive> &liveRange,
                    std::unordered_map<const BufferNode*, std::unordered_set<const BufferNode*>> &conflictMap,
                    std::unordered_map<const BufferNode*, int64_t> &addrMap, int64_t &totalSize) {
      std::list<const BufferNode*> op_list;
      std::copy(ops.begin(), ops.end(), std::back_inserter(op_list));
      
      op_list.sort([&liveRange](const BufferNode *a, const BufferNode *b) {
        return liveRange[a].tensor_size >= liveRange[b].tensor_size;
      });
      for (auto &op : op_list) {
        std::shared_ptr<OpAddr> best_addr;
        int64_t min_conflict_count = std::numeric_limits<int64_t>::max();
        for (int i = 0; i < bank_num_; ++i) {
          int64_t offset = i * bank_size_;
          int64_t mem_cross_bank_num =
              std::ceil(static_cast<float>(liveRange[op].tensor_size) / bank_size_);
          int64_t end_offset = offset + (mem_cross_bank_num + 1) * bank_size_;
          if (i + mem_cross_bank_num >= bank_num_)  {
            break;
          }
          auto op_addr = searchAddr(op, liveRange, offset, end_offset);
      
          // op can insert
          if (op_addr->start + op_addr->size < std::min(end_offset, mem_size_)) {
            int64_t conf_count = getConflictCount(op_addr, conflictMap);
            if (conf_count < min_conflict_count) {
              min_conflict_count = conf_count;
              best_addr = op_addr;
            }
          }
        }
        if (!best_addr) {
          std::cout << "Errrror" << std::endl;
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
      auto iter = std::find_if(allocated_op_list_.begin(), allocated_op_list_.end(),
                               [&opAddr](std::shared_ptr<OpAddr> &p) {
                                 return p->start >= opAddr->start;
                               });
      allocated_op_list_.emplace(iter, opAddr);
      int64_t bank_start = opAddr->start / bank_size_;
      int64_t bank_end = opAddr->end / bank_size_;
      for (int i = bank_start; i <= bank_end; i++) {
        bank_ops[i].push_back(opAddr->op);
      }
    }
  
    int64_t getConflictCount(std::shared_ptr<OpAddr> &opAddr,
                             std::unordered_map<const BufferNode*, std::unordered_set<const BufferNode*>> &conflictMap) {
      int64_t bank_start = opAddr->start / bank_size_;
      int64_t bank_end = opAddr->end / bank_size_;
      int count = 0;
      for (int i = bank_start; i <= bank_end; i++) {
        for (auto op : bank_ops[i]) {
          if (conflictMap[opAddr->op].count(op)) {
            ++count;
          }
        }
      }
      return count;
    }
    
    std::shared_ptr<OpAddr>
    searchAddr(const BufferNode* op, std::unordered_map<const BufferNode*, TensorLive> &liveRange, int64_t offset, int64_t end_offset) {

      std::shared_ptr<OpAddr> op_addr = std::make_shared<OpAddr>(
          op, liveRange[op].tensor_size, liveRange[op].start, liveRange[op].end);
      int64_t prev_offset = offset;
      int64_t best_offset = -1;
      int64_t smallest_gap = std::numeric_limits<int64_t>::max();
      for (auto &allocated_op_addr : allocated_op_list_) {
        if (allocated_op_addr->start > end_offset) {
          break;
        }
        uint32_t max_first_pos =
            std::max(op_addr->first_pos, allocated_op_addr->first_pos);
        uint32_t min_last_pos =
            std::min(op_addr->end_pos, allocated_op_addr->end_pos);
        if (max_first_pos < min_last_pos) {
          int64_t gap = allocated_op_addr->start - prev_offset;
          if (gap >= op_addr->size && gap < smallest_gap) {
            smallest_gap = gap;
            best_offset = prev_offset;
          }
          prev_offset = std::max(prev_offset, allocated_op_addr->end);
        }
      }
      if (best_offset == -1) {
        best_offset = prev_offset;
      }
      op_addr->start = best_offset;
      op_addr->end = op_addr->start + op_addr->size;
      return op_addr;
    }
  
  protected:
    std::list<std::shared_ptr<OpAddr>> allocated_op_list_;
    std::vector<std::vector<const BufferNode*>> bank_ops;
    int64_t total_consumption_;
    int64_t bank_num_;
    int64_t bank_size_;
    int64_t mem_size_;
};


PrimFunc InferAddress(PrimFunc f) {
  int bank_num = 16, bank_size = 16 * 1024;
  std::unordered_map<const BufferNode*, std::unordered_set<const BufferNode*>> bank_conflict_map;
  std::unordered_map<const BufferNode*, TensorLive> live_ranges;
  std::vector<const BufferNode*> alloc_ops = AddressAllocator().collectAllocOp(f->body);
  // for (auto &op : alloc_ops) {
  //   bank_conflict_map.Set(op, {});
  // }
  std::cout << "collect ops num:" << alloc_ops.size() << std::endl;

  for (auto &op : alloc_ops) {
    for (auto &op2 : alloc_ops) {
      if (op == op2) {
        continue;
      }
      auto current_set = bank_conflict_map[op];
      // current_set.insert(op2);
      // bank_conflict_map.Set(op, current_set);
      bank_conflict_map[op].insert(op2);
    }
    auto shape = op->shape;
    auto dtype = op->dtype;
    uint32_t op_size = 1;
    for (auto s: shape) {
      int ss = s.as<IntImmNode>()->value;
      op_size *= ss;
    }
    op_size /= bank_num;
    int bytes_size = 1;
    if (dtype == DataType::Float(16)) {
      bytes_size = 2;
    } else if (dtype == DataType::Float(32)) {
      bytes_size = 4;
    }
    op_size *= bytes_size;
    // live_ranges.Set(op, {1, 4, op_size});
    live_ranges[op] = {1, 4, op_size};
  }

  std::unordered_map<const BufferNode*, int64_t> addrMapWithBC;
  int64_t memUsedWithBC = 0;
  MemAllocBankConflictAware allocatorBC(bank_num, bank_size);
  auto success =
      allocatorBC.assignAddr(alloc_ops, live_ranges, bank_conflict_map,
                  addrMapWithBC, memUsedWithBC);
  
  if (success) {
    // std::unordered_map<String, PrimExpr> result;
    auto fn = f.CopyOnWrite();
    auto fn_attr = fn->attrs.CopyOnWrite();
    for (auto op : alloc_ops) {
      int32_t address = addrMapWithBC[op];
      fn_attr->dict.Set(op->name, PrimExpr(address));
    }
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

TVM_REGISTER_GLOBAL("tl.transform.AddressAssign")
    .set_body_typed(AddressAssign);


}  // namespace tir
}  // namespace tvm
