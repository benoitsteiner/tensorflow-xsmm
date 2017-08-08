/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_BUFFER_H_

#include <ostream>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// A container which can hold one or more HloValues. An HLO buffer abstractly
// represents the allocation which HLO instructions write into and read
// from. Generally there is a one-to-one correspondence between HloBuffers and
// HloValue where each HloValue in the module is held in a unique HloBuffer. An
// exception is the while instruction which updates the loop state in-place. In
// this case, we have a single HloBuffer for each HloPosition in the loop state,
// but multiple HloValues. For example:
//
//   %init = ...
//   %while = While(%init, body, condition)
//
//  body:
//   %body_param = Param(0)
//     ...
//   %body_root = ...
//
//  condition:
//   %cond_param = Param(0)
//     ...
//
// For simplicity, assume that %while is array-shaped. In this case, we have a
// single HloBuffer which holds the following HloValues: HloValue{%init},
// HloValue{%while}, HloValue{%body_param}, HloValue{%body_root}, and
// HloValue{%cond_param}.
//
// HloBuffers may appear at different HloPositions in the module mirroring the
// same propery of HloValues. For example:
//
//   %sub = Sub(...)
//   %add = Add(...)
//   %tuple = Tuple(%add, %sub)
//   %gte = GetTupleElement(%tuple, 0)
//
// In this case, the HloBuffer containing %add appears at the following
// positions: HloPosition{%add, {}}, HloPosition{%tuple, {0}}, and
// HloPosition{%gte, {}}.
//
// Different HloPositions which share the same HloBuffer indicate mandatory
// aliasing in the HLO module. These positions must share the same memory
// allocation for correctness (the backends rely on this property). This differs
// from incidental aliasing introduced by memory reuse in BufferAssignment where
// different instructions may happen to get the same allocation.
class HloBuffer {
 public:
  using Id = int64;

  // Predicate comparing HloBuffers by increasing id, useful for std::sort.
  static bool IdLessThan(const HloBuffer* a, const HloBuffer* b) {
    return a->id() < b->id();
  }

  // Predicate comparing HloBuffers by equal id, useful for std::unique.
  static bool IdEqual(const HloBuffer* a, const HloBuffer* b) {
    return a->id() == b->id();
  }

  HloBuffer(Id id) : id_(id) {}

  // Return the unique identifier for this HloBuffer.
  Id id() const { return id_; }

  // Add a value to the set of values held by this buffer. Also adds the
  // HloPositions of the value to the positions vector of the buffer. If the
  // buffer already contains this value, then this method is a nop.
  void AddValue(const HloValue& value);

  // Return all values contained in this buffer.
  const std::vector<const HloValue*>& values() const {
    return values_.values();
  }

  // Return the unique HLO value in the buffer. CHECK fails if the buffer does
  // not contain exactly one value.
  const HloValue& GetUniqueValue() const { return values_.GetUniqueValue(); }

  // Return the positions (output of which instruction and at what index) where
  // the buffer is used. This is exactly the union of the positions of the
  // HloValues contained by the buffer.
  const std::vector<HloPosition>& positions() const { return positions_; }

  string ToString() const;

  bool operator==(const HloBuffer& other) const;
  bool operator!=(const HloBuffer& other) const { return !(*this == other); }

 private:
  // Unique identifier for this HloBuffer.
  const Id id_;

  // The set of values contained in this buffer.
  HloValueSet values_;

  // The set of positions where this buffer is used.
  std::vector<HloPosition> positions_;
};

std::ostream& operator<<(std::ostream& out, const HloBuffer& buffer);

// A class representing the set of possible HloBuffers at a particular
// HloPosition (shape index in the output of an instruction) in the XLA
// graph. In most cases, the buffer set will have a single HloBuffer indicating
// that the HloBuffer which appears at that particular position is known
// unambiguously at compile-time.  However, tuple-shaped Select instructions can
// introduce ambiguity as the tuple elements of the operands are passed by
// reference into the output of the Select. For example:
//
//   %pred = ...
//   %tuple0 = Tuple(%a, %b)
//   %tuple1 = Tuple(%x, %y)
//   %select = Select(%pred, %tuple0, %tuple1)
//
// In this case the HloBufferSet at HloPosition{%select, {0}} contains the
// HloBuffer holding %a and the HloBuffer holding %x.
class HloBufferSet {
 public:
  HloBufferSet() = default;

  // Add the given buffer to this buffer set. If the buffer already exists in
  // the set, then this is a NOP.
  void AddBuffer(const HloBuffer* buffer);

  // Removes the given buffer from this buffer set. CHECK fails in the buffer is
  // not contained in this set.
  void RemoveBufferOrDie(HloBuffer::Id buffer_id);

  // Returns the unique buffer in this set. CHECK fails if the set does not
  // contain exactly one buffer.
  const HloBuffer& GetUniqueBuffer() const {
    CHECK_EQ(buffers_.size(), 1);
    return *buffers_[0];
  }

  // Returns the vector of HloBuffers in the set, sorted by HloBuffer::Id.
  const std::vector<const HloBuffer*>& buffers() const { return buffers_; }

  string ToString() const;

 private:
  // HloBuffers sorted by HloBuffer::Id.
  std::vector<const HloBuffer*> buffers_;
};

std::ostream& operator<<(std::ostream& out, const HloBufferSet& buffer_set);

// A class collecting the HloBuffers in the output of an HLO instruction. For
// array-shaped instructions, an InstructionBufferSet trivially holds a single
// HloBufferSet. Tuple-shaped InstructionBufferSets hold multiple
// HloBufferSets.
class InstructionBufferSet : public ShapeTree<HloBufferSet> {
 public:
  InstructionBufferSet(const Shape& shape) : ShapeTree<HloBufferSet>(shape) {}

  // Returns true if any HloBufferSet contained in this InstructionBufferSet
  // is not a singleton.
  bool IsAmbiguous() const;

  // Returns true if any HloBuffer appears in more than one HloBufferSet
  // contained in this InstructionBufferSet.
  bool IsDistinct() const;

  string ToString() const;
};

std::ostream& operator<<(std::ostream& out,
                         const InstructionBufferSet& buffer_set);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_BUFFER_H_
