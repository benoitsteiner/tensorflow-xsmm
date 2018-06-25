/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/tensorrt/shape_fn/trt_shfn.h"
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"

#include <string>
#include <vector>

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace shape_inference {

tensorflow::Status TRTEngineOpShapeInference(InferenceContext* context) {
  std::vector<tensorflow::TensorShape> shapes;
  for (int i = 0; i < context->num_outputs(); ++i) {
    context->set_output(i, context->UnknownShape());
  }
  auto status = context->GetAttr("input_shapes", &shapes);
  // it is ok to not to have shapes
  if (!status.ok()) return Status::OK();
  if ((int)shapes.size() != context->num_inputs()) return Status::OK();
  bool different_input = false;
  for (int i = 0; i < context->num_inputs(); ++i) {
    if (shapes.at(i) != context->input_tensor(i)->shape())
      different_input = true;
  }
  if (different_input) return Status::OK();
  shapes.resize(0);
  status = context->GetAttr("output_shapes", &shapes);
  if (!status.ok()) return Status::OK();
  if ((int)shapes.size() != context->num_outputs()) return Status::OK();
  std::vector<ShapeHandle> shape_handles(shapes.size());
  for (size_t i = 0; i < shapes.size(); ++i) {
    status =
        context->MakeShapeFromTensorShape(shapes.at(i), &shape_handles.at(i));
    if (!status.ok()) return Status::OK();
  }
  for (int i = 0; i < context->num_outputs(); ++i) {
    context->set_output(i, shape_handles.at(i));
  }
  return Status::OK();
}
}  // namespace shape_inference
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
