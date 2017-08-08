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

#include "tensorflow/compiler/xla/service/hlo_buffer.h"

#include <algorithm>
#include <ostream>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using ::tensorflow::str_util::Join;
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

void HloBuffer::AddValue(const HloValue& value) {
  // If the value is already contained in this buffer, just return.
  if (!values_.AddValue(&value)) {
    return;
  }

  // Add all of the positions of the HloValue to this buffer.
  for (const HloPosition& position : value.positions()) {
    if (std::find(positions_.begin(), positions_.end(), position) ==
        positions_.end()) {
      positions_.push_back(position);
    }
  }
}

bool HloBuffer::operator==(const HloBuffer& other) const {
  bool equal = id() == other.id();
  if (equal) {
    // DCHECK because these comparisons are expensive (linear time).
    DCHECK(values_ == other.values_);
    DCHECK(positions() == other.positions());
  }
  return equal;
}

string HloBuffer::ToString() const {
  return StrCat("HloBuffer ", id_, ", values: ", values_.ToString());
}

std::ostream& operator<<(std::ostream& out, const HloBuffer& buffer) {
  out << buffer.ToString();
  return out;
}

void HloBufferSet::AddBuffer(const HloBuffer* buffer) {
  auto it = std::lower_bound(buffers_.begin(), buffers_.end(), buffer,
                             HloBuffer::IdLessThan);
  if (it == buffers_.end() || (*it)->id() != buffer->id()) {
    buffers_.insert(it, buffer);
  }
}

void HloBufferSet::RemoveBufferOrDie(HloBuffer::Id buffer_id) {
  auto it = std::lower_bound(buffers_.begin(), buffers_.end(), buffer_id,
                             [](const HloBuffer* buffer, HloBuffer::Id id) {
                               return buffer->id() < id;
                             });
  CHECK(it != buffers_.end() && (*it)->id() == buffer_id)
      << "HloBuffer " << buffer_id << " doesn't exist in set: " << ToString();
  buffers_.erase(it);
}

string HloBufferSet::ToString() const {
  return StrCat(
      "HloBufferSet, buffers: ",
      Join(buffers_, ", ", [](string* result, const HloBuffer* buffer) {
        result->append(buffer->ToString());
      }));
}

std::ostream& operator<<(std::ostream& out, const HloBufferSet& buffer_set) {
  out << buffer_set.ToString();
  return out;
}

bool InstructionBufferSet::IsAmbiguous() const {
  bool is_ambiguous = false;
  ForEachElement(
      [&is_ambiguous](const ShapeIndex& index, const HloBufferSet& buffer_set) {
        is_ambiguous |= buffer_set.buffers().size() > 1;
      });
  return is_ambiguous;
}

bool InstructionBufferSet::IsDistinct() const {
  bool is_distinct = true;
  tensorflow::gtl::FlatSet<HloBuffer::Id> seen_ids;
  ForEachElement([&is_distinct, &seen_ids](const ShapeIndex& /*index*/,
                                           const HloBufferSet& buffer_set) {
    for (const HloBuffer* buffer : buffer_set.buffers()) {
      auto pair = seen_ids.insert(buffer->id());
      if (!pair.second) {
        is_distinct = false;
      }
    }
  });
  return is_distinct;
}

string InstructionBufferSet::ToString() const {
  string out =
      StrCat("InstructionBufferSet(", ShapeUtil::HumanString(shape()), ")\n");
  ForEachElement([this, &out](const ShapeIndex& index,
                              const HloBufferSet& value_set) {
    StrAppend(&out, "  ", index.ToString(), " : ", value_set.ToString(), "\n");
  });
  return out;
}

std::ostream& operator<<(std::ostream& out,
                         const InstructionBufferSet& buffer_set) {
  out << buffer_set.ToString();
  return out;
}

}  // namespace xla
