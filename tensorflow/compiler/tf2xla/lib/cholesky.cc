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

#include "tensorflow/compiler/tf2xla/lib/cholesky.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2xla/lib/batch_dot.h"
#include "tensorflow/compiler/tf2xla/lib/triangular_solve.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/lib/while_loop.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace {

// The Cholesky–Banachiewicz algorithm. See
// https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky–Banachiewicz_and_Cholesky–Crout_algorithms
// for a description.
//
// def cholesky_unblocked(a):
//   assert len(a.shape) == 2 and a.shape[-2] == a.shape[-1]
//   n = a.shape[-2]
//   l = np.zeros_like(a)
//   for j in xrange(n):
//     row = l[..., j, :j]
//     row_t = np.swapaxes(row, -1, -2)
//     l[..., j, j] = np.sqrt(a[..., j, j] - np.dot(row, row_t))
//     l[..., j+1:, j] = (a[..., j+1:, j] - np.dot(l[..., j+1:, :j], row_t)) /
//                       l[..., j, j]
//   return l
xla::StatusOr<xla::XlaOp> CholeskyUnblocked(xla::XlaBuilder* builder,
                                            const xla::XlaOp& a) {
  TF_ASSIGN_OR_RETURN(xla::Shape a_shape, builder->GetShape(a));
  const int n_dims = xla::ShapeUtil::Rank(a_shape);
  const int64 n = xla::ShapeUtil::GetDimension(a_shape, -1);
  gtl::ArraySlice<int64> major_dims(xla::AsInt64Slice(a_shape.dimensions()),
                                    /*pos=*/0,
                                    /*len=*/n_dims - 2);

  xla::XlaOp l = Zeros(builder, a_shape);

  // Construct the for loop body to iterate over rows.
  auto body_fn = [&](xla::XlaOp i, gtl::ArraySlice<xla::XlaOp> loop_vars,
                     xla::XlaBuilder* body_builder)
      -> xla::StatusOr<std::vector<xla::XlaOp>> {
    xla::Shape col_shape;
    xla::Shape row_shape;
    for (int64 d : major_dims) {
      row_shape.add_dimensions(d);
      col_shape.add_dimensions(d);
    }
    row_shape.add_dimensions(1);
    row_shape.add_dimensions(n);
    row_shape.set_element_type(a_shape.element_type());
    auto mask_zeros_row = Zeros(body_builder, row_shape);

    col_shape.add_dimensions(n);
    col_shape.add_dimensions(1);
    col_shape.set_element_type(a_shape.element_type());
    auto mask_zeros_col = Zeros(body_builder, col_shape);

    std::vector<int32> mask_vector(n);
    std::iota(mask_vector.begin(), mask_vector.end(), 0);
    auto mask_range = body_builder->ConstantR1<int32>(mask_vector);
    auto mask_range_row = body_builder->Broadcast(
        body_builder->Reshape(mask_range, {0}, {1, n}), major_dims);
    auto mask_range_col = body_builder->Broadcast(
        body_builder->Reshape(mask_range, {0}, {n, 1}), major_dims);
    auto body_a = loop_vars[0];
    auto body_l = loop_vars[1];

    // row = l[..., i, :i]
    // select the whole i-th row, then mask out all columns past i-1
    auto zero = body_builder->ConstantR0<int32>(0);
    TF_ASSIGN_OR_RETURN(auto l_i, DynamicSliceInMinorDims(body_builder, body_l,
                                                          {i, zero}, {1, n}));
    auto row = body_builder->Select(body_builder->Ge(mask_range_row, i),
                                    mask_zeros_row, l_i);
    // a[..., i, i]
    TF_ASSIGN_OR_RETURN(auto a_ii, DynamicSliceInMinorDims(body_builder, body_a,
                                                           {i, i}, {1, 1}));
    // np.dot(row, np.swapaxes(row, -1, -2))
    xla::XlaOp diag_dot;
    TF_ASSIGN_OR_RETURN(diag_dot, BatchDot(body_builder, row, row,
                                           /*transpose_x=*/false,
                                           /*transpose_y=*/true));
    // l[..., i, i] = np.sqrt(a[..., i, i] - np.dot(row,
    //                                              np.swapaxes(row, -1, -2)))
    auto l_ii = body_builder->Pow(
        body_builder->Sub(a_ii, diag_dot),
        FloatLiteral(body_builder, a_shape.element_type(), 0.5));

    // a[..., i+1:, i]
    // select the whole i-th column, then mask out all rows above i+1
    TF_ASSIGN_OR_RETURN(
        auto a_0i, DynamicSliceInMinorDims(body_builder, body_a, {i}, {1}));
    auto a_ip1i = body_builder->Select(body_builder->Le(mask_range_col, i),
                                       mask_zeros_col, a_0i);

    // l[..., i+1:, i] = (a[..., i+1:, i] - np.dot(l[..., i+1:, :i], r.T)) /
    //                   l[..., i, i]
    // The columns in [i, n] are zeroed out in `row`, so we just have to
    // zero out rows above i+1 after the BatchDot. np.dot(l[..., :, :i],
    // r.T)
    TF_ASSIGN_OR_RETURN(auto dot, BatchDot(body_builder, body_l, row,
                                           /*transpose_x=*/false,
                                           /*transpose_y=*/true));
    // np.dot(l[..., i+1:, :i], r.T)
    auto dot_ip1 = body_builder->Select(body_builder->Le(mask_range_col, i),
                                        mask_zeros_col, dot);

    auto col_update =
        body_builder->Div(body_builder->Sub(a_ip1i, dot_ip1), l_ii);
    TF_ASSIGN_OR_RETURN(body_l, DynamicUpdateSliceInMinorDims(
                                    body_builder, body_l, col_update, {i}));
    // Assign the diagonal after the rest of the column because otherwise the
    // column assign will wrap around and overwrite the diagonal assign.
    TF_ASSIGN_OR_RETURN(body_l, DynamicUpdateSliceInMinorDims(
                                    body_builder, body_l, l_ii, {i, i}));

    return std::vector<xla::XlaOp>{body_a, body_l};
  };

  TF_ASSIGN_OR_RETURN(
      auto cholesky_while,
      XlaForEachIndex(n, xla::S32, body_fn, {a, l}, "unblocked", builder));

  return cholesky_while[1];
}

}  // namespace

xla::StatusOr<xla::XlaOp> Cholesky(xla::XlaBuilder* builder, xla::XlaOp a,
                                   int64 block_size) {
  TF_ASSIGN_OR_RETURN(xla::Shape a_shape, builder->GetShape(a));
  const int ndims = xla::ShapeUtil::Rank(a_shape);
  if (ndims < 2) {
    return errors::InvalidArgument(
        "Arguments to Cholesky must have rank >= 2: ", ndims);
  }

  const int64 n = xla::ShapeUtil::GetDimension(a_shape, -1);
  if (n != xla::ShapeUtil::GetDimension(a_shape, -2)) {
    return errors::InvalidArgument(
        "Arguments to Cholesky must be square matrices: ",
        xla::ShapeUtil::HumanString(a_shape));
  }

  if (block_size < 1) {
    return errors::InvalidArgument(
        "block_size argument to Cholesky must be >= 1; got ", block_size);
  }

  // Blocked left-looking Cholesky factorization.
  // Algorithm 1 from
  // Haidar, Azzam, et al. "High-performance Cholesky factorization for GPU-only
  // execution." Proceedings of General Purpose GPUs. ACM, 2017.
  xla::XlaOp l = Zeros(builder, a_shape);
  for (int64 i = 0; i < n; i += block_size) {
    int64 k = std::min(block_size, n - i);
    if (i > 0) {
      // TODO(phawkins): consider implementing SYRK for the diagonal part of
      // the panel.
      // a[i:, i:i+k] -= np.dot(l[i:, :i], np.transpose(l[i:i+k, :i]))
      TF_ASSIGN_OR_RETURN(auto lhs,
                          SliceInMinorDims(builder, l, {i, 0}, {n, i}));
      TF_ASSIGN_OR_RETURN(auto rhs,
                          SliceInMinorDims(builder, l, {i, 0}, {i + k, i}));
      TF_ASSIGN_OR_RETURN(auto delta,
                          BatchDot(builder, lhs, rhs, /*transpose_x=*/false,
                                   /*transpose_y=*/true, /*conjugate_x=*/false,
                                   /*conjugate_y=*/false));
      TF_ASSIGN_OR_RETURN(auto before,
                          SliceInMinorDims(builder, a, {i, i}, {n, i + k}));
      TF_ASSIGN_OR_RETURN(
          a, UpdateSliceInMinorDims(builder, a, builder->Sub(before, delta),
                                    {i, i}));
    }

    // l[i:i+k, i:i+k] = cholesky_unblocked(a[i:i+k, i:i+k])
    TF_ASSIGN_OR_RETURN(auto x,
                        SliceInMinorDims(builder, a, {i, i}, {i + k, i + k}));
    TF_ASSIGN_OR_RETURN(auto factorized, CholeskyUnblocked(builder, x));
    TF_ASSIGN_OR_RETURN(l,
                        UpdateSliceInMinorDims(builder, l, factorized, {i, i}));

    if (i + k < n) {
      // l[i+k:, i:i+k] = trsm_right_transpose(l[i:i+k, i:i+k], a[i+k:, i:i+k])
      TF_ASSIGN_OR_RETURN(auto panel,
                          SliceInMinorDims(builder, a, {i + k, i}, {n, i + k}));
      TF_ASSIGN_OR_RETURN(auto update,
                          TriangularSolve(builder, factorized, panel,
                                          /*left_side=*/false,
                                          /*lower=*/true,
                                          /*transpose_a=*/true,
                                          /*conjugate_a=*/false,
                                          /*block_size=*/block_size));
      TF_ASSIGN_OR_RETURN(
          l, UpdateSliceInMinorDims(builder, l, update, {i + k, i}));
    }
  }
  return l;
}

}  // namespace tensorflow
