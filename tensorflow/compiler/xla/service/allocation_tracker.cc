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

#include "tensorflow/compiler/xla/service/allocation_tracker.h"

#include <utility>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

StatusOr<GlobalDataHandle> AllocationTracker::Register(
    std::unique_ptr<ShapedBuffer> shaped_buffer, const string& tag) {
  tensorflow::mutex_lock lock(mutex_);
  VLOG(2) << "Register";
  std::vector<std::unique_ptr<ShapedBuffer>> replicated_buffers;
  replicated_buffers.emplace_back(std::move(shaped_buffer));
  return RegisterInternal(std::move(replicated_buffers), tag);
}

StatusOr<GlobalDataHandle> AllocationTracker::RegisterReplicatedBuffers(
    std::vector<std::unique_ptr<ShapedBuffer>> replicated_buffers,
    const string& tag) {
  tensorflow::mutex_lock lock(mutex_);
  VLOG(2) << "RegisterReplicatedBuffers";
  return RegisterInternal(std::move(replicated_buffers), tag);
}

StatusOr<GlobalDataHandle> AllocationTracker::RegisterInternal(
    std::vector<std::unique_ptr<ShapedBuffer>> replicated_buffers,
    const string& tag) {
  VLOG(2) << "RegisterInternal("
          << "tag: \"" << tag << "\" with " << replicated_buffers.size()
          << " shaped_buffers.";
  for (const auto& shaped_buffer : replicated_buffers) {
    VLOG(2) << "shaped_buffer:" << *shaped_buffer;
    if (shaped_buffer->platform() != backend_->platform()) {
      return InvalidArgument(
          "AllocationTracker for platform %s cannot register buffer from "
          "platform %s",
          backend_->platform()->Name().c_str(),
          shaped_buffer->platform()->Name().c_str());
    }
  }

  int64 handle = next_handle_++;
  for (auto& shaped_buffer : replicated_buffers) {
    std::vector<ShapeIndex> shape_indices;
    ShapeUtil::ForEachSubshape(shaped_buffer->on_device_shape(),
                               [this, &shape_indices](const Shape& /*subshape*/,
                                                      const ShapeIndex& index) {
                                 shape_indices.push_back(index);
                               });
    for (const ShapeIndex& index : shape_indices) {
      AddAllocationOrIncrementRefCount(shaped_buffer->buffer(index),
                                       shaped_buffer->device_ordinal());
    }
    handle_to_shaped_buffers_[handle].emplace_back(std::move(shaped_buffer));
  }

  GlobalDataHandle result;
  result.set_handle(handle);
  VLOG(2) << "handle: " << handle;
  return result;
}

tensorflow::Status AllocationTracker::Unregister(const GlobalDataHandle& data) {
  tensorflow::mutex_lock lock(mutex_);
  VLOG(2) << "Unregister("
          << "handle: " << data.handle() << ")";
  TF_ASSIGN_OR_RETURN(std::vector<const ShapedBuffer*> replicated_buffers,
                      ResolveInternal(data));
  for (const auto& shaped_buffer : replicated_buffers) {
    std::vector<ShapeIndex> shape_indices;
    ShapeUtil::ForEachSubshape(shaped_buffer->on_device_shape(),
                               [this, &shape_indices](const Shape& /*subshape*/,
                                                      const ShapeIndex& index) {
                                 shape_indices.push_back(index);
                               });
    for (const ShapeIndex& index : shape_indices) {
      TF_RETURN_IF_ERROR(DecrementRefCount(shaped_buffer->buffer(index),
                                           shaped_buffer->device_ordinal()));
    }
  }
  return Reset(data);
}

Status AllocationTracker::Reset(const GlobalDataHandle& data) {
  // Keep a nullptr as a tombstone for unregistered handles. This enables
  // better error messages. That is, "handle has been deallocated" versus
  // "handle does not exist".
  auto it = handle_to_shaped_buffers_.find(data.handle());
  if (it == handle_to_shaped_buffers_.end()) {
    return NotFound("no allocation record for global data handle: %lld",
                    data.handle());
  }
  for (auto& shaped_buffer : it->second) {
    shaped_buffer.reset();
  }
  return tensorflow::Status::OK();
}

StatusOr<std::vector<GlobalDataHandle>> AllocationTracker::DeconstructTuple(
    const GlobalDataHandle& data) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(std::vector<const ShapedBuffer*> replicated_buffers,
                      ResolveInternal(data));
  // We only need to care about replica id 0 here, since the GlobalDataHandle is
  // the same for all buffers across replicas.
  const ShapedBuffer* shaped_buffer = replicated_buffers[0];
  if (!ShapeUtil::IsTuple(shaped_buffer->on_host_shape())) {
    return InvalidArgument("global data handle %lld is not a tuple",
                           data.handle());
  }
  // If the on-host representation is a tuple, then the on-device one should be
  // as well.
  TF_RET_CHECK(ShapeUtil::IsTuple(shaped_buffer->on_device_shape()));

  if (ShapeUtil::IsNestedTuple(shaped_buffer->on_device_shape())) {
    return Unimplemented("Deconstructing nested tuples is not implemented.");
  }

  std::vector<GlobalDataHandle> element_handles;
  for (int i = 0;
       i < ShapeUtil::TupleElementCount(shaped_buffer->on_device_shape());
       ++i) {
    auto element_buffer = MakeUnique<ShapedBuffer>(
        ShapeUtil::GetTupleElementShape(shaped_buffer->on_host_shape(), i),
        ShapeUtil::GetTupleElementShape(shaped_buffer->on_device_shape(), i),
        shaped_buffer->platform(), shaped_buffer->device_ordinal());
    element_buffer->set_buffer(shaped_buffer->buffer(/*index=*/{i}),
                               /*index=*/{});
    std::vector<std::unique_ptr<ShapedBuffer>> replicated_buffers;
    replicated_buffers.emplace_back(std::move(element_buffer));
    TF_ASSIGN_OR_RETURN(
        GlobalDataHandle element_handle,
        RegisterInternal(std::move(replicated_buffers), "deconstructed tuple"));

    element_handles.push_back(element_handle);
  }
  return std::move(element_handles);
}

StatusOr<std::vector<const ShapedBuffer*>> AllocationTracker::Resolve(
    const GlobalDataHandle& data) {
  tensorflow::mutex_lock lock(mutex_);
  return AllocationTracker::ResolveInternal(data);
}

StatusOr<const ShapedBuffer*> AllocationTracker::ResolveForReplica(
    const GlobalDataHandle& data, int replica_id) {
  tensorflow::mutex_lock lock(mutex_);
  TF_ASSIGN_OR_RETURN(std::vector<const ShapedBuffer*> replicated_buffers,
                      ResolveInternal(data));
  if (replica_id >= replicated_buffers.size()) {
    return InvalidArgument(
        "Requesting buffer for replica %d, but found buffers only for %lu "
        "replicas.",
        replica_id, replicated_buffers.size());
  }
  return replicated_buffers[replica_id];
}

StatusOr<std::vector<const ShapedBuffer*>> AllocationTracker::ResolveInternal(
    const GlobalDataHandle& data) {
  VLOG(2) << "resolve:" << data.handle();
  auto it = handle_to_shaped_buffers_.find(data.handle());
  if (it == handle_to_shaped_buffers_.end()) {
    return NotFound("no allocation record for global data handle: %lld",
                    data.handle());
  }
  std::vector<const ShapedBuffer*> replicated_buffers;
  for (const auto& shaped_buffer : it->second) {
    if (shaped_buffer == nullptr) {
      return InvalidArgument(
          "global data handle %lld was previously deallocated", data.handle());
    }
    replicated_buffers.push_back(shaped_buffer.get());
  }

  return replicated_buffers;
}

void AllocationTracker::AddAllocationOrIncrementRefCount(
    perftools::gputools::DeviceMemoryBase device_memory, int device_ordinal) {
  AllocationMap& allocation_map = opaque_to_allocation_map_[device_ordinal];
  auto it = allocation_map.find(device_memory.opaque());
  if (it == allocation_map.end()) {
    allocation_map[device_memory.opaque()] = {device_memory, device_ordinal,
                                              /*ref_count=*/1};
  } else {
    it->second.ref_count++;
  }
}

Status AllocationTracker::DecrementRefCount(
    perftools::gputools::DeviceMemoryBase device_memory, int device_ordinal) {
  AllocationMap& allocation_map = opaque_to_allocation_map_[device_ordinal];
  auto it = allocation_map.find(device_memory.opaque());
  TF_RET_CHECK(it != allocation_map.end());
  Allocation& allocation = it->second;
  TF_RET_CHECK(allocation.ref_count >= 1);
  if (allocation.ref_count == 1) {
    TF_RETURN_IF_ERROR(backend_->memory_allocator()->Deallocate(
        device_ordinal, &device_memory));
    allocation_map.erase(it);
  } else {
    allocation.ref_count--;
  }
  return tensorflow::Status::OK();
}

}  // namespace xla
