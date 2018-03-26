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

#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {

namespace {
// Wrangles the minimum number of proto fields to set up a matrix.
void DescribeMatrix(int rows, int columns, OpInfo* op_features) {
  auto input = op_features->add_inputs();
  auto shape = input->mutable_shape();
  auto shape_rows = shape->add_dim();
  shape_rows->set_size(rows);
  auto shape_columns = shape->add_dim();
  shape_columns->set_size(columns);
  input->set_dtype(DT_FLOAT);
}

void SetCpuDevice(OpInfo* op_features) {
  auto device = op_features->mutable_device();
  device->set_type("CPU");
  device->set_num_cores(10);
  device->set_bandwidth(10000000);  // 10000000 KB/s = 10 GB/s
  device->set_frequency(1000);      // 1000 Mhz = 1 GHz
}

// Returns an OpInfo for MatMul with the minimum set of fields set up.
OpContext DescribeMatMul(int m, int n, int l, int k) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("MatMul");

  DescribeMatrix(m, l, &op_context.op_info);
  DescribeMatrix(k, n, &op_context.op_info);
  return op_context;
}

// Wrangles the minimum number of proto fields to set up an input of
// arbitrary rank and type.
void DescribeArbitraryRankInput(const std::vector<int>& dims, DataType dtype,
                                OpInfo* op_info) {
  auto input = op_info->add_inputs();
  input->set_dtype(dtype);
  auto shape = input->mutable_shape();
  for (auto d : dims) {
    shape->add_dim()->set_size(d);
  }
}

// Wrangles the minimum number of proto fields to set up an output of
// arbitrary rank and type.
void DescribeArbitraryRankOutput(const std::vector<int>& dims, DataType dtype,
                                 OpInfo* op_info) {
  auto output = op_info->add_outputs();
  output->set_dtype(dtype);
  auto shape = output->mutable_shape();
  for (auto d : dims) {
    shape->add_dim()->set_size(d);
  }
}

// Returns an OpInfo for a BatchMatMul
OpContext DescribeBatchMatMul(const std::vector<int>& dims_a,
                              const std::vector<int>& dims_b) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("BatchMatMul");

  DescribeArbitraryRankInput(dims_a, DT_FLOAT, &op_context.op_info);
  DescribeArbitraryRankInput(dims_b, DT_FLOAT, &op_context.op_info);
  return op_context;
}

// Wrangles the minimum number of proto fields to set up a 4D Tensor for cost
// estimation purposes.
void DescribeTensor4D(int dim0, int dim1, int dim2, int dim3,
                      OpInfo::TensorProperties* tensor) {
  auto shape = tensor->mutable_shape();
  shape->add_dim()->set_size(dim0);
  shape->add_dim()->set_size(dim1);
  shape->add_dim()->set_size(dim2);
  shape->add_dim()->set_size(dim3);
  tensor->set_dtype(DT_FLOAT);
}

// DescribeConvolution constructs an OpContext for a Conv2D applied to an input
// tensor with shape (batch, ix, iy, iz1) and a kernel tensor with shape
// (kx, ky, iz2, oz).
OpContext DescribeConvolution(int batch, int ix, int iy, int iz1, int iz2,
                              int kx, int ky, int oz) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("Conv2D");

  DescribeTensor4D(batch, ix, iy, iz1, op_context.op_info.add_inputs());
  DescribeTensor4D(kx, ky, iz2, oz, op_context.op_info.add_inputs());

  return op_context;
}

// DescribeUnaryOp constructs an OpContext for the given operation applied to
// a 4-tensor with shape (size1, 1, 1, 1).
OpContext DescribeUnaryOp(const string& op, int size1) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op(op);

  DescribeTensor4D(size1, 1, 1, 1, op_context.op_info.add_inputs());
  DescribeTensor4D(size1, 1, 1, 1, op_context.op_info.add_outputs());

  return op_context;
}

// DescribeBinaryOp constructs an OpContext for the given operation applied to
// a 4-tensor with dimensions (size1, 1, 1, 1) and a 4-tensor with dimensions
// (2 * size1, size2, 1, 1).
//
// The choice of dimension here is arbitrary, and is used strictly to test the
// cost model for applying elementwise operations to tensors with unequal
// dimension values.
OpContext DescribeBinaryOp(const string& op, int size1, int size2) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op(op);

  DescribeTensor4D(size1, 1, 1, 1, op_context.op_info.add_inputs());
  DescribeTensor4D(2 * size1, size2, 1, 1, op_context.op_info.add_inputs());
  DescribeTensor4D(2 * size1, size2, 1, 1, op_context.op_info.add_outputs());

  return op_context;
}

// DescribeBiasAdd constructs an OpContext for a BiasAdd applied to a 4-tensor
// with dimensions (1, 1, size2, size1) and a bias with dimension (size1),
// according to the constraint that the bias must be 1D with size equal to that
// of the last dimension of the input value.
OpContext DescribeBiasAdd(int size1, int size2) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("BiasAdd");

  DescribeTensor4D(1, 1, size2, size1, op_context.op_info.add_inputs());
  DescribeTensor4D(1, 1, size2, size1, op_context.op_info.add_outputs());

  auto bias = op_context.op_info.add_inputs();
  bias->mutable_shape()->add_dim()->set_size(size1);
  bias->set_dtype(DT_FLOAT);

  return op_context;
}

}  // namespace

class OpLevelCostEstimatorTest : public ::testing::Test {
 protected:
  Costs PredictCosts(const OpContext& op_context) const {
    return estimator_.PredictCosts(op_context);
  }

  int64 CountMatMulOperations(const OpInfo& op_features,
                              bool* found_unknown_shapes) const {
    return estimator_.CountMatMulOperations(op_features, found_unknown_shapes);
  }

  int64 CountBatchMatMulOperations(const OpInfo& op_features,
                                   bool* found_unknown_shapes) const {
    return estimator_.CountBatchMatMulOperations(op_features,
                                                 found_unknown_shapes);
  }

  void SetComputeMemoryOverlap(bool value) {
    estimator_.compute_memory_overlap_ = value;
  }

  OpLevelCostEstimator estimator_;
};

TEST_F(OpLevelCostEstimatorTest, TestGatherCosts) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("Gather");

  // Huge first input shouldn't affect Gather execution and memory costs.
  DescribeArbitraryRankInput({10000000, 10}, DT_FLOAT, &op_context.op_info);
  DescribeArbitraryRankInput({16}, DT_INT64, &op_context.op_info);
  DescribeArbitraryRankOutput({16, 10}, DT_FLOAT, &op_context.op_info);

  auto cost = estimator_.PredictCosts(op_context);
  EXPECT_EQ(Costs::Duration(130), cost.memory_time);
  EXPECT_EQ(Costs::Duration(16), cost.compute_time);
  EXPECT_EQ(Costs::Duration(146), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, TestSliceCosts) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("Slice");

  // Huge first input shouldn't affect Slice execution and memory costs.
  DescribeArbitraryRankInput({10000000, 10}, DT_FLOAT, &op_context.op_info);
  DescribeArbitraryRankInput({2}, DT_INT64, &op_context.op_info);
  DescribeArbitraryRankInput({2}, DT_INT64, &op_context.op_info);
  DescribeArbitraryRankOutput({10, 10}, DT_FLOAT, &op_context.op_info);

  auto cost = estimator_.PredictCosts(op_context);
  EXPECT_EQ(Costs::Duration(81), cost.memory_time);
  EXPECT_EQ(Costs::Duration(10), cost.compute_time);
  EXPECT_EQ(Costs::Duration(91), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, BiasAddExecutionTime) {
  auto cost = PredictCosts(DescribeBiasAdd(1000, 10));
  EXPECT_EQ(Costs::Duration(8400), cost.memory_time);
  EXPECT_EQ(Costs::Duration(1000), cost.compute_time);
  EXPECT_EQ(Costs::Duration(9400), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, Conv2DExecutionTime) {
  auto cost = PredictCosts(DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256));
  EXPECT_EQ(Costs::Duration(233780), cost.memory_time);
  EXPECT_EQ(Costs::Duration(354877440), cost.compute_time);
  EXPECT_EQ(Costs::Duration(355111220), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, DummyExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Dummy", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(0), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2000), cost.execution_time);
  EXPECT_TRUE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, ExecutionTimeSumOrMax) {
  SetComputeMemoryOverlap(true);
  auto cost = PredictCosts(DescribeBinaryOp("Dummy", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(0), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2000), cost.execution_time);  // max(2000, 200)
  EXPECT_TRUE(cost.inaccurate);
  SetComputeMemoryOverlap(false);  // Set it back to default.
}

TEST_F(OpLevelCostEstimatorTest, MulExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Mul", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(200), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2200), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, MulBroadcastExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Mul", 1000, 2));
  EXPECT_EQ(Costs::Duration(3600), cost.memory_time);
  EXPECT_EQ(Costs::Duration(400), cost.compute_time);
  EXPECT_EQ(Costs::Duration(4000), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, ModExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Mod", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(1600), cost.compute_time);
  EXPECT_EQ(Costs::Duration(3600), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, ReluExecutionTime) {
  auto cost = PredictCosts(DescribeUnaryOp("Relu", 1000));
  EXPECT_EQ(Costs::Duration(800), cost.memory_time);
  EXPECT_EQ(Costs::Duration(100), cost.compute_time);
  EXPECT_EQ(Costs::Duration(900), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, UnknownOrPartialShape) {
  EXPECT_FALSE(PredictCosts(DescribeMatMul(2, 4, 7, 7)).inaccurate);
  EXPECT_TRUE(PredictCosts(DescribeMatMul(-1, 4, 7, 7)).inaccurate);
  EXPECT_TRUE(PredictCosts(DescribeMatMul(2, 4, -1, 7)).inaccurate);

  EXPECT_FALSE(PredictCosts(DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256))
                   .inaccurate);
  EXPECT_TRUE(PredictCosts(DescribeConvolution(16, -1, 19, 48, 48, 5, 5, 256))
                  .inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, BatchMatMul) {
  EXPECT_TRUE(PredictCosts(DescribeBatchMatMul({}, {})).inaccurate);
  EXPECT_TRUE(PredictCosts(DescribeBatchMatMul({2, 4}, {})).inaccurate);
  EXPECT_FALSE(PredictCosts(DescribeBatchMatMul({2, 4}, {4, 2})).inaccurate);
  EXPECT_FALSE(
      PredictCosts(DescribeBatchMatMul({1, 2, 4}, {1, 4, 2})).inaccurate);
  EXPECT_FALSE(
      PredictCosts(DescribeBatchMatMul({2, 4}, {1, 3, 4, 2})).inaccurate);
  bool matmul_inaccurate = false;
  bool batch_matmul_inaccurate = false;
  EXPECT_EQ(
      CountMatMulOperations(DescribeMatMul(2, 2, 4, 4).op_info,
                            &matmul_inaccurate),
      CountBatchMatMulOperations(DescribeBatchMatMul({2, 4}, {4, 2}).op_info,
                                 &batch_matmul_inaccurate));
  EXPECT_EQ(matmul_inaccurate, batch_matmul_inaccurate);
  EXPECT_EQ(10 * CountMatMulOperations(DescribeMatMul(2, 2, 4, 4).op_info,
                                       &matmul_inaccurate),
            CountBatchMatMulOperations(
                DescribeBatchMatMul({10, 2, 4}, {-1, 10, 4, 2}).op_info,
                &batch_matmul_inaccurate));
  EXPECT_NE(matmul_inaccurate, batch_matmul_inaccurate);
  EXPECT_EQ(20 * CountMatMulOperations(DescribeMatMul(2, 2, 4, 4).op_info,
                                       &matmul_inaccurate),
            CountBatchMatMulOperations(
                DescribeBatchMatMul({2, 10, 2, 4}, {-1, 10, 4, 2}).op_info,
                &batch_matmul_inaccurate));
  EXPECT_NE(matmul_inaccurate, batch_matmul_inaccurate);
}

// Helper functions for testing GetTensorShapeProtoFromTensorProto().
void GetTensorProto(const DataType dtype, const std::vector<int64>& shape,
                    const std::vector<int64> values, const bool tensor_content,
                    TensorProto* tensor_proto) {
  tensor_proto->Clear();
  TensorProto temp_tensor_proto;
  temp_tensor_proto.set_dtype(dtype);
  for (const auto& x : shape) {
    temp_tensor_proto.mutable_tensor_shape()->add_dim()->set_size(x);
  }
  for (const auto& x : values) {
    if (dtype == DT_INT64) {
      temp_tensor_proto.add_int64_val(x);
    } else if (dtype == DT_INT32 || dtype == DT_INT16 || dtype == DT_INT8 ||
               dtype == DT_UINT8) {
      temp_tensor_proto.add_int_val(x);
    } else if (dtype == DT_UINT32) {
      temp_tensor_proto.add_uint32_val(x);
    } else if (dtype == DT_UINT64) {
      temp_tensor_proto.add_uint64_val(x);
    } else {
      CHECK(false) << "Unsupported dtype: " << dtype;
    }
  }
  Tensor tensor(dtype);
  CHECK(tensor.FromProto(temp_tensor_proto));
  if (tensor_content) {
    tensor.AsProtoTensorContent(tensor_proto);
  } else {
    tensor.AsProtoField(tensor_proto);
  }
}

void ExpectTensorShape(const std::vector<int64>& expected,
                       const TensorShapeProto& tensor_shape_proto) {
  TensorShape tensor_shape_expected(expected);
  TensorShape tensor_shape(tensor_shape_proto);

  LOG(INFO) << "Expected: " << tensor_shape_expected.DebugString();
  LOG(INFO) << "TensorShape: " << tensor_shape.DebugString();
  EXPECT_TRUE(tensor_shape_expected == tensor_shape);
}

TEST_F(OpLevelCostEstimatorTest, GetTensorShapeProtoFromTensorProto) {
  TensorProto tensor_proto;
  TensorShapeProto tensor_shape_proto;

  // Dimension larger than max value; should fail while converting to Tensor
  // class.
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(255);
  EXPECT_FALSE(
      GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));

  tensor_proto.Clear();
  // Expect only 1D shape.
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(1);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(2);
  EXPECT_FALSE(
      GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));

  // Expect only handle integer data types.
  GetTensorProto(DT_FLOAT, {}, {}, /*tensor_content=*/false, &tensor_proto);
  EXPECT_FALSE(
      GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));

  // Check GetTensorShapeProtoFromTensorProto() resturns correct values.
  {
    std::vector<int64> shape_expected = {10, 20, 30, 40};
    GetTensorProto(DT_INT32, {4}, shape_expected, /*tensor_content=*/false,
                   &tensor_proto);
    EXPECT_TRUE(
        GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));
    ExpectTensorShape(shape_expected, tensor_shape_proto);
  }

  {
    std::vector<int64> shape_expected = {40, 20, 90, 40};
    GetTensorProto(DT_INT64, {4}, shape_expected, /*tensor_content=*/false,
                   &tensor_proto);
    EXPECT_TRUE(
        GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));
    ExpectTensorShape(shape_expected, tensor_shape_proto);
  }

  {
    std::vector<int64> shape_expected = {10, 20, 30, 40};
    GetTensorProto(DT_INT32, {4}, shape_expected, /*tensor_content=*/true,
                   &tensor_proto);
    EXPECT_TRUE(
        GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));
    ExpectTensorShape(shape_expected, tensor_shape_proto);
  }

  {
    std::vector<int64> shape_expected = {40, 20, 90, 40};
    GetTensorProto(DT_INT64, {4}, shape_expected, /*tensor_content=*/true,
                   &tensor_proto);
    EXPECT_TRUE(
        GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));
    ExpectTensorShape(shape_expected, tensor_shape_proto);
  }
}

}  // end namespace grappler
}  // end namespace tensorflow
