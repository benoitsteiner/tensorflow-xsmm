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

#include "tensorflow/compiler/jit/xla_device_context.h"

#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

// The allocator used for Tensors assigned to the XLA device.
XlaDeviceAllocator::XlaDeviceAllocator() {}
XlaDeviceAllocator::~XlaDeviceAllocator() = default;

string XlaDeviceAllocator::Name() { return "xla"; }

void* XlaDeviceAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  // We always return an empty XlaTensor object, encoded as an opaque tagged
  // pointer. We can return an empty object and ignore num_bytes here because we
  // have control over all of the uses of this device tensor, and can lazily
  // allocate memory when used. This allows us to also know the shape of the
  // allocated Tensor, which is useful if the device's tensor representation
  // differs from the host.
  return XlaTensor::ToOpaquePointer(new XlaTensor());
}

void XlaDeviceAllocator::DeallocateRaw(void* ptr) {
  delete XlaTensor::FromOpaquePointer(ptr);
}

void XlaDeviceAllocator::GetStats(AllocatorStats* stats) { stats->Clear(); }

XlaTransferManager::XlaTransferManager(
    se::Stream* stream, xla::LocalClient* client, bool transfer_as_literal,
    XlaCompiler::ShapeRepresentationFn shape_representation_fn)
    : stream_(stream),
      client_(client),
      transfer_manager_(client->backend().transfer_manager()),
      transfer_as_literal_(transfer_as_literal),
      shape_representation_fn_(std::move(shape_representation_fn)) {
  if (!shape_representation_fn_) {
    shape_representation_fn_ = [](const TensorShape& shape, DataType dtype) {
      return shape;
    };
  }
}

Status XlaTransferManager::TransferLiteralToDevice(
    const Tensor& host_tensor, Tensor* device_tensor) const {
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(host_tensor.dtype(),
                                           host_tensor.shape(), &xla_shape));
  xla::BorrowingLiteral literal(
      static_cast<const char*>(DMAHelper::base(&host_tensor)), xla_shape);

  const xla::ShapedBuffer& shaped_buffer =
      XlaTensor::FromTensor(device_tensor)->shaped_buffer();
  VLOG(1) << "Transfer to device as literal: " << literal.ToString() << " "
          << shaped_buffer.ToString();
  return transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                    shaped_buffer);
}

Status XlaTransferManager::TransferLiteralFromDevice(
    Tensor* host_tensor, const Tensor& device_tensor) const {
  const xla::ShapedBuffer& shaped_buffer =
      XlaTensor::FromTensor(&device_tensor)->shaped_buffer();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::Literal> literal,
      transfer_manager_->TransferLiteralFromDevice(stream_, shaped_buffer));
  VLOG(1) << "Transfer from device as literal: " << literal->ToString() << " "
          << shaped_buffer.ToString();
  Tensor tensor;
  TF_RETURN_IF_ERROR(
      LiteralToHostTensor(*literal, host_tensor->dtype(), &tensor));
  // Reshape the tensor back to its declared shape.
  if (!host_tensor->CopyFrom(tensor, device_tensor.shape())) {
    return errors::Internal(
        "Tensor::CopyFrom failed when copying from XLA device to CPU");
  }
  return Status::OK();
}

void XlaTransferManager::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                               Device* device,
                                               Tensor* device_tensor,
                                               StatusCallback done) const {
  if (cpu_tensor->NumElements() > 0) {
    VLOG(2) << "CopyCPUTensorToDevice "
            << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
            << " "
            << reinterpret_cast<const void*>(
                   device_tensor->tensor_data().data())
            << " " << cpu_tensor->NumElements() << " "
            << cpu_tensor->shape().DebugString() << " "
            << device_tensor->shape().DebugString();

    void* src_ptr = const_cast<void*>(DMAHelper::base(cpu_tensor));
    const int64 total_bytes = cpu_tensor->TotalBytes();

    XlaTensor* xla_tensor = XlaTensor::FromTensor(device_tensor);
    CHECK(xla_tensor);

    TensorShape shape = shape_representation_fn_(device_tensor->shape(),
                                                 device_tensor->dtype());
    if (!xla_tensor->has_shaped_buffer()) {
      Status s = xla_tensor->AllocateShapedBuffer(
          device_tensor->dtype(), shape, client_,
          stream_->parent()->device_ordinal());
      if (!s.ok()) {
        done(s);
        return;
      }
    }

    Status status;
    if (transfer_as_literal_) {
      Tensor reshaped_cpu_tensor;
      if (!reshaped_cpu_tensor.CopyFrom(*cpu_tensor, shape)) {
        done(errors::Internal(
            "Tensor::CopyFrom failed when copying from CPU to XLA device"));
        return;
      }
      status = TransferLiteralToDevice(reshaped_cpu_tensor, device_tensor);
    } else {
      se::DeviceMemoryBase dev_dst_ptr =
          XlaTensor::DeviceMemoryFromTensor(*device_tensor);
      stream_->ThenMemcpy(&dev_dst_ptr, src_ptr, total_bytes);
      // TODO(hpucha): Make this asynchronous.
      Status block_status = stream_->BlockHostUntilDone();
      if (!block_status.ok()) {
        status = xla::InternalError(
            "Failed to complete data transfer on stream %p: %s", stream_,
            block_status.error_message().c_str());
      }
    }
    xla_tensor->set_host_tensor(*cpu_tensor);

    done(status);
    return;
  }

  VLOG(2) << "CopyCPUTensorToDevice empty tensor";
  done(Status::OK());
}

void XlaTransferManager::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                               StringPiece tensor_name,
                                               Device* device,
                                               Tensor* cpu_tensor,
                                               StatusCallback done) {
  if (device_tensor->NumElements() > 0) {
    VLOG(2) << "CopyDeviceTensorToCPU "
            << reinterpret_cast<const void*>(
                   device_tensor->tensor_data().data())
            << " "
            << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
            << " " << device_tensor->NumElements() << " "
            << cpu_tensor->shape().DebugString() << " "
            << device_tensor->shape().DebugString();

    const int64 total_bytes = cpu_tensor->TotalBytes();
    se::DeviceMemoryBase dev_src_ptr =
        XlaTensor::DeviceMemoryFromTensor(*device_tensor);
    void* dst_ptr = DMAHelper::base(cpu_tensor);

    Status status;
    if (transfer_as_literal_) {
      status = TransferLiteralFromDevice(cpu_tensor, *device_tensor);
    } else {
      stream_->ThenMemcpy(dst_ptr, dev_src_ptr, total_bytes);
      // TODO(hpucha): Make this asynchronous.
      Status block_status = stream_->BlockHostUntilDone();
      if (!block_status.ok()) {
        status = xla::InternalError(
            "Failed to complete data transfer on stream %p: %s", stream_,
            block_status.error_message().c_str());
      }
    }

    done(status);
    return;
  }

  VLOG(2) << "CopyDeviceTensorToCPU empty tensor";
  done(Status::OK());
}

void XlaTransferManager::CopyDeviceTensorToDevice(const Tensor& src_tensor,
                                                  Tensor* dst_tensor,
                                                  const StatusCallback& done) {
  // TODO(phawkins): replace this code with an asynchronous implementation.
  auto body = [&]() {
    if (src_tensor.NumElements() == 0) {
      return Status::OK();
    }
    XlaTensor* xla_src = XlaTensor::FromTensor(&src_tensor);
    XlaTensor* xla_dst = XlaTensor::FromTensor(dst_tensor);
    CHECK(xla_src && xla_dst)
        << "Missing destination tensor for device-to-device copy";
    if (!xla_dst->has_shaped_buffer()) {
      TensorShape shape =
          shape_representation_fn_(src_tensor.shape(), src_tensor.dtype());
      TF_RETURN_IF_ERROR(
          xla_dst->AllocateShapedBuffer(src_tensor.dtype(), shape, client_,
                                        stream_->parent()->device_ordinal()));
    }
    TF_RETURN_IF_ERROR(
        xla_dst->shaped_buffer().buffers().ForEachMutableElementWithStatus(
            [&](const xla::ShapeIndex& index, se::DeviceMemoryBase* buffer) {
              const se::DeviceMemoryBase& from_buffer =
                  xla_src->shaped_buffer().buffers().element(index);
              CHECK_EQ(buffer->size(), from_buffer.size());
              if (!stream_->parent()->SynchronousMemcpy(buffer, from_buffer,
                                                        buffer->size())) {
                return errors::Internal("Device to device memcpy failed");
              }
              return Status::OK();
            }));
    return Status::OK();
  };
  done(body());
}

XlaDeviceContext::XlaDeviceContext(
    se::Stream* stream, xla::LocalClient* client, bool transfer_as_literal,
    XlaCompiler::ShapeRepresentationFn shape_representation_fn)
    : manager_(stream, client, transfer_as_literal,
               std::move(shape_representation_fn)) {}

void XlaDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                             Device* device,
                                             Tensor* device_tensor,
                                             StatusCallback done) const {
  manager_.CopyCPUTensorToDevice(cpu_tensor, device, device_tensor, done);
}

void XlaDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                             StringPiece tensor_name,
                                             Device* device, Tensor* cpu_tensor,
                                             StatusCallback done) {
  manager_.CopyDeviceTensorToCPU(device_tensor, tensor_name, device, cpu_tensor,
                                 done);
}

void XlaDeviceContext::CopyDeviceTensorToDevice(const Tensor& src_tensor,
                                                Tensor* dst_tensor,
                                                const StatusCallback& done) {
  manager_.CopyDeviceTensorToDevice(src_tensor, dst_tensor, done);
}

}  // namespace tensorflow
