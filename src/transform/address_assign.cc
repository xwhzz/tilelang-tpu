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
#include <algorithm>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "../op/bulk_copy.h"
#include "../op/gemm.h"
#include "../target/bm1690_lmem.h"

namespace tvm {
namespace tl {
using namespace tir;

namespace {

int64_t AlignUp(int64_t value, int64_t align) {
  return bm1690::AlignUp(value, align);
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

    op_list.sort([&liveRange](const BufferNode *a, const BufferNode *b) {
      auto &lhs = liveRange[a];
      auto &rhs = liveRange[b];
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
      for (int i = 0; i < bank_num_; ++i) {
        int64_t offset = i * bank_size_;
        int64_t bytes = liveRange[op].tensor_size;
        int64_t mem_cross_bank_num =
            static_cast<int64_t>(bm1690::DivUp(bytes, bank_size_));
        mem_cross_bank_num = std::max<int64_t>(mem_cross_bank_num, 1);

        if (i + mem_cross_bank_num > bank_num_) {
          break;
        }
        int64_t end_offset = offset + mem_cross_bank_num * bank_size_;
        auto op_addr = searchAddr(op, liveRange, offset, end_offset);

        // op can insert
        if (op_addr->start + op_addr->size <= std::min(end_offset, mem_size_)) {
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
        if (gap >= op_addr->size) {
          best_offset = candidate;
          break;
        }
        prev_offset = std::max(prev_offset, allocated_op_addr->end);
      }
    }
    if (best_offset == -1) {
      best_offset = AlignUp(prev_offset, bm1690::kTensorAlignBytes);
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
    for (auto lhs : current_op_buffers_) {
      for (auto rhs : current_op_buffers_) {
        if (lhs != rhs) {
          (*conflict_map_)[lhs].insert(rhs);
        }
      }
    }
    current_op_buffers_.clear();
  }

  void MarkUse(const BufferNode *buffer) {
    auto it = live_ranges_->find(buffer);
    if (it == live_ranges_->end()) {
      return;
    }
    if (!inside_op_) {
      current_loc_ = NextLoc();
    } else {
      current_op_buffers_.insert(buffer);
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

  void VisitExpr_(const VarNode *op) {
    auto it = buffer_var_to_buffer_.find(op);
    if (it != buffer_var_to_buffer_.end()) {
      MarkUse(it->second);
    }
  }

  void VisitExpr_(const BufferLoadNode *op) {
    OpScope scope(this);
    MarkUse(op->buffer.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) {
    OpScope scope(this);
    MarkUse(op->buffer.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) {
    OpScope scope(this);
    StmtExprVisitor::VisitExpr_(op);
  }

  std::unordered_map<const VarNode *, const BufferNode *> buffer_var_to_buffer_;
  std::unordered_set<const BufferNode *> seen_buffers_;
  std::unordered_set<const BufferNode *> current_op_buffers_;
  std::unordered_map<const BufferNode *, TensorLive> *live_ranges_;
  std::unordered_map<const BufferNode *,
                     std::unordered_set<const BufferNode *>> *conflict_map_;
  uint32_t loc_ = 0;
  uint32_t current_loc_ = 0;
  bool inside_op_ = false;
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
      int32_t address = addrMapWithBC[op];
      fn_attr->dict.Set(op->name, PrimExpr(address));
    }
  }
  std::cerr << "[AddressAssign] success=" << std::boolalpha << success
            << " buffers=" << alloc_ops.size() << " total=" << memUsedWithBC
            << " bytes\n";

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
