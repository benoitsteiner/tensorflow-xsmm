# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An in-process, local XLA client in Python, supporting AOT compilation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum  # pylint: disable=g-bad-import-order
import itertools

import numpy as np

from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.python import pywrap_xla as c_api


class PaddingType(enum.Enum):
  VALID = 1
  SAME = 2


_UNARY_OPS = [
    'Not',
    'Abs',
    'Exp',
    'Floor',
    'Ceil',
    'Log',
    'Sign',
    'Cos',
    'Sin',
    'Tanh',
    'SqrtF32',
    'SquareF32',
    'IsFinite',
    'ReciprocalF32',
    'Neg',
    'Sort',
]

_BINARY_OPS = [
    'Eq',
    'Ne',
    'Ge',
    'Gt',
    'Lt',
    'Le',
    'Add',
    'Sub',
    'Mul',
    'Div',
    'Rem',
    'Max',
    'Min',
    'And',
    'Or',
    'Pow',
]

# Most functions are snake_case for consistency with other modules,
# whereas method names of ComputationBuilder and LocalComputation are
# CamelCase for consistency with XLA.
# pylint: disable=invalid-name

XLA_ELEMENT_TYPE_TO_DTYPE = {
    xla_data_pb2.F32: np.dtype(np.float32),
    xla_data_pb2.F64: np.dtype(np.float64),
    xla_data_pb2.S32: np.dtype(np.int32),
    xla_data_pb2.S64: np.dtype(np.int64),
    xla_data_pb2.U32: np.dtype(np.uint32),
    xla_data_pb2.U64: np.dtype(np.uint64),
    xla_data_pb2.PRED: np.dtype(np.bool),
    xla_data_pb2.TUPLE: np.dtype(np.object),
}

# Note the conversion on the key. Numpy has a known issue wherein dtype hashing
# doesn't work as expected (https://github.com/numpy/numpy/issues/7242). Thus,
# when keying by dtype in this dict, we use the string form of dtypes.
DTYPE_TO_XLA_ELEMENT_TYPE = {
    str(v): k
    for k, v in XLA_ELEMENT_TYPE_TO_DTYPE.items()
}


class LocalBuffer(object):
  """Represents a handle to data owned by XLA.

  The referent is ready for use in executing a local, compiled
  Computation. On XLA platforms involving a device (e.g. GPU), this
  means the referent is in device memory.
  """

  def __init__(self, c_local_shaped_buffer):
    self.c_local_shaped_buffer = c_local_shaped_buffer
    self._delete = c_api.DeleteLocalShapedBuffer

  @staticmethod
  def from_py(npval):
    npval = require_numpy_array_layout(npval)
    return LocalBuffer(c_api.LocalShapedBuffer.FromLiteral(npval))

  def to_py(self):
    return self.c_local_shaped_buffer.ToLiteral()

  def delete(self):
    if self.c_local_shaped_buffer is not None:
      self._delete(self.c_local_shaped_buffer)
      self.c_local_shaped_buffer = None

  def is_deleted(self):
    return self.c_local_shaped_buffer is None

  def __del__(self):
    self.delete()


class Shape(object):
  """XLA shape.

  Represents an XLA shape by a corresponding Python/Numpy type and a
  list of dimensions, which are themselves Shapes in case this one
  represents an XLA tuple.
  """

  def __init__(self, np_dtype, dimensions):
    self.np_dtype = np_dtype
    self._dimensions = dimensions

  def __repr__(self):
    return 'xla_client.Shape(np_dtype={!r}, dimensions={!r})'.format(
        self.np_dtype, self._dimensions)

  def element_type(self):
    return DTYPE_TO_XLA_ELEMENT_TYPE[str(self.np_dtype)]

  def is_tuple(self):
    return self.element_type() == xla_data_pb2.TUPLE

  def dimensions(self):
    if self.is_tuple():
      raise ValueError('Tuple shape has no dimensions')
    return self._dimensions

  def tuple_shapes(self):
    if not self.is_tuple():
      raise ValueError('Shape is not a tuple shape')
    return self._dimensions

  @staticmethod
  def from_numpy(npval):

    def convert(npval):
      if isinstance(npval, tuple):
        return Shape(np.dtype('O'), tuple(convert(elt) for elt in npval))
      else:
        return Shape(npval.dtype, np.shape(npval))

    return convert(require_numpy_array_layout(npval))


def _wrap_shape(shape_info):
  dtype, dims = shape_info
  element_type = DTYPE_TO_XLA_ELEMENT_TYPE[str(dtype)]
  if element_type == xla_data_pb2.TUPLE:
    dims = [_wrap_shape(subshape_info) for subshape_info in dims]
  return Shape(dtype, dims)


def _unwrap_shape(shape):
  if shape.is_tuple():
    components = tuple(
        _unwrap_shape(subshape) for subshape in shape.tuple_shapes())
  else:
    components = shape.dimensions()
  return (shape.np_dtype, components)


def _unwrap_shapes(shapes):
  return [_unwrap_shape(shape) for shape in shapes]


def _wrap_data_handle(handle):
  cdh = xla_data_pb2.ComputationDataHandle()
  cdh.handle = handle
  return cdh


def _unwrap_data_handle(handle_proto):
  return handle_proto.handle


def _unwrap_data_handles(handle_protos):
  return [_unwrap_data_handle(cdh) for cdh in handle_protos]


def require_numpy_array_layout(value):
  if isinstance(value, tuple):
    return tuple(require_numpy_array_layout(x) for x in value)
  else:
    return np.require(value, requirements=['C', 'A'])


def transfer_to_infeed(value, replica_number=None):
  """Transfers the given value into the XLA infeed queue.

  XLA's infeed queue is a single queue that feeds the "XLA virtual machine" with
  a totally ordered stream of values. This is dequeued from XLA computations via
  the Infeed() operation.

  Args:
    value: the value that the caller would like to enqueue into the XLA infeed
      queue
    replica_number: the replica number to infeed the value to -- if not
      provided, then the default replica (trivially replica 0) is used.
  """
  if replica_number is None:
    c_api.TransferToInfeedLocal(require_numpy_array_layout(value))
  else:
    c_api.TransferToInfeedLocalReplica(
        require_numpy_array_layout(value), replica_number)


def transfer_from_outfeed(shape, replica_number=None):
  """Transfers a literal of the given shape from replica_number's outfeed.

  Args:
    shape: The shape of the value to transfer from outfeed.
    replica_number: The replica number ordinal to transfer the outfeed value
      from. (Each replica has a distinct outfeed queue.)

  Returns:
    The literal value that is produced from the outfeed queue.
  """
  return c_api.TransferFromOutfeedLocalReplica(
      _unwrap_shape(shape), replica_number or 0)


class LocalComputation(object):
  """Python wrapper for a local XLA Computation.

  A LocalComputation can be executed if it is compiled. Otherwise, it
  can still be used as a Computation where required by the
  ComputationBuilder methods.
  """

  def __init__(self, c_local_computation, is_compiled):
    self.c_local_computation = c_local_computation
    self.is_compiled = is_compiled

    # Ensure a reference to C-based destructor for use in __del__.
    if is_compiled:
      self._delete = c_api.DeleteCompiledLocalComputation
    else:
      self._delete = c_api.DeleteLocalComputation

  def Compile(self, argument_shapes=()):
    if self.is_compiled:
      raise ValueError('Attempt to compile a compiled local XLA computation.')
    return LocalComputation(
        self.c_local_computation.Compile(_unwrap_shapes(argument_shapes)),
        is_compiled=True)

  def CompileWithExampleArguments(self, arguments=()):
    return self.Compile(
        argument_shapes=[Shape.from_numpy(arg) for arg in arguments])

  def Execute(self, arguments=()):
    if not self.is_compiled:
      raise ValueError('Cannot execute an uncompiled local XLA computation.')
    arguments = tuple(map(require_numpy_array_layout, arguments))
    return self.c_local_computation.Execute(arguments)

  def ExecuteWithLocalBuffers(self, arguments=()):
    """Execute with LocalBuffer arguments and return value."""
    if not self.is_compiled:
      raise ValueError('Cannot execute an uncompiled local XLA computation.')
    arguments = tuple(arguments)
    if any(arg.is_deleted() for arg in arguments):
      raise ValueError('Executing with deleted local buffer argument')
    return LocalBuffer(
        self.c_local_computation.ExecuteWithShapedBuffers(
            [arg.c_local_shaped_buffer for arg in arguments]))

  def __del__(self):
    self._delete(self.c_local_computation)


class ComputationBuilder(object):
  """XLA computation builder.

  Enqueues XLA ops in sequence and in order to build a
  LocalComputation, which in turn can be compiled into a
  CompiledLocalComputation, which in turn can be locally executed.
  """

  # The methods of this class map 1-to-1 onto the XLA C++
  # computation builder API. Therefore, there's no need to laboriously list
  # arguments and return values for every method, especially where it's obvious.
  #
  # pylint: disable=g-doc-return-or-yield
  # pylint: disable=g-doc-args

  def __init__(self, name):
    self._client = c_api.LocalComputationBuilder(name.encode('utf8'))
    self._parameter_numbering = itertools.count()

  def Build(self):
    return LocalComputation(self._client.Build(), is_compiled=False)

  def Infeed(self, shape):
    """Enqueues an infeed op onto the computation.

    Infeed operations dequeue data of the given shape from the device's infeed
    queue for subsequent use in the computation.

    Returns:
      A  ComputationDataHandle message.
    """
    return _wrap_data_handle(self._client.Infeed(_unwrap_shape(shape)))

  def Outfeed(self, operand):
    """Enqueues an outfeed op onto the computation.

    Outfeed operations enqueue data, using the given operand, onto the XLA
    outfeed queue for subsequent dequeue via the client API.
    """
    self._client.Outfeed(
        _unwrap_data_handle(operand), _unwrap_shape(self.GetShape(operand)),
        ''.encode('utf-8'))

  def Constant(self, value):
    """Enqueues a constant op onto the computation.

    Args:
      value: value for the constant, as a np.array with an explicit dtype set
             to one of the supported types.

    Returns:
      A ComputationDataHandle message.
    """
    value = require_numpy_array_layout(value)
    return _wrap_data_handle(self._client.ConstantLiteral(value))

  def ConstantF32Scalar(self, value):
    """Convenience method to enqueue a scalar F32 constant op.

    Args:
      value: a floating-point number.

    Returns:
      A ComputationDataHandle message.
    """
    return self.Constant(np.array(value, dtype=np.float32))

  def ConstantF64Scalar(self, value):
    """Convenience method to enqueue a scalar F32 constant op.

    Args:
      value: a floating-point number.

    Returns:
      A ComputationDataHandle message.
    """
    return self.Constant(np.array(value, dtype=np.float64))

  def ConstantS32Scalar(self, value):
    """Convenience method to enqueue a scalar S32 constant op.

    Args:
      value: a floating-point number.

    Returns:
      A ComputationDataHandle message.
    """
    return self.Constant(np.array(value, dtype=np.int32))

  def ConstantS64Scalar(self, value):
    """Convenience method to enqueue a scalar S64 constant op.

    Args:
      value: a floating-point number.

    Returns:
      A ComputationDataHandle message.
    """
    return self.Constant(np.array(value, dtype=np.int64))

  def ConstantPredScalar(self, value):
    """Convenience method to enqueue a scalar PRED constant op.

    Args:
      value: a boolean value.

    Returns:
      A ComputationDataHandle message.
    """
    return self.Constant(np.array(value, dtype=np.bool))

  def ParameterWithShape(self, shape, name=None, parameter_num=None):
    """Enqueues a Parameter op onto the computation, given a shape.

    Args:
      shape: the parameter's shape as a Shape object.
      name: optional string name for the parameter.
      parameter_num: parameter number in the computation function. If None,
        the next linear parameter number is used. The default value capability
        can be used for auto-numbering. If you're using auto-numbering for some
        parameters, use it for *all* parameters to avoid clashes.

    Returns:
      A ComputationDataHandle message.
    """
    if name is None:
      name = ''
    if parameter_num is None:
      parameter_num = next(self._parameter_numbering)

    return _wrap_data_handle(
        self._client.Parameter(
            parameter_num, _unwrap_shape(shape), name.encode('utf8')))

  def ParameterFromNumpy(self, value, name=None, parameter_num=None):
    """Enqueues a Parameter op onto the computation.

    Args:
      value: a Numpy array, or a nested tuple thereof, from which the
        shape is inferred.
      name: as in ParameterWithShape.
      parameter_num: as in ParameterWithShape.

    Returns:
      A ComputationDataHandle message.
    """
    return self.ParameterWithShape(
        Shape.from_numpy(value), name=name, parameter_num=parameter_num)

  def Broadcast(self, operand, sizes):
    """Enqueues a broadcast operation onto the computation.

    Args:
      operand: the operand ComputationDataHandle to broadcast.
      sizes: an iterable of broadcast sizes.

    Returns:
      A ComputationDataHandle representing the added broadcast op.
    """
    return _wrap_data_handle(
        self._client.Broadcast(_unwrap_data_handle(operand), sizes))

  def Concatenate(self, operands, dimension):
    """Enqueues a concatenate operation onto the computation.

    Args:
      operands: the operands to concatenate.
      dimension: the dimension in which to perform the concatenation.

    Returns:
      A ComputationDataHandle representing the added concatenate op.
    """
    return _wrap_data_handle(
        self._client.ConcatInDim(_unwrap_data_handles(operands), dimension))

  def ConvertElementType(self, operand, new_element_type):
    """Enqueues an element type conversion operation onto the computation.

    Args:
      operand: the operand to convert.
      new_element_type: the target primitive type.

    Returns:
      A ComputationDataHandle representing the added conversion op.
    """
    return _wrap_data_handle(
        self._client.ConvertElementType(
            _unwrap_data_handle(operand), new_element_type))

  def GetShape(self, operand):
    return _wrap_shape(self._client.GetShape(_unwrap_data_handle(operand)))

  def GetComputationStats(self):
    raise NotImplementedError()

  def Reshape(self, operand, dimensions, new_sizes):
    """Reshape op."""
    return _wrap_data_handle(
        self._client.Reshape(
            _unwrap_data_handle(operand), dimensions, new_sizes))

  def CrossReplicaSum(self, operand):
    """CrossReplicaSum op.

    Args:
      operand: the operand to sum across replica instances.

    Returns:
      A ComputationDataHandle that has the sum of the value among all replicas.
    """
    return _wrap_data_handle(
        self._client.CrossReplicaSum(_unwrap_data_handle(operand)))

  def Collapse(self, operand, dimensions):
    """Collapse op."""
    return _wrap_data_handle(
        self._client.Collapse(_unwrap_data_handle(operand), dimensions))

  def Trans(self, operand):
    """Specialized matrix transpose op."""
    return _wrap_data_handle(
        self._client.Transpose(_unwrap_data_handle(operand), [1, 0]))

  def Transpose(self, operand, permutation):
    """Transpose op."""
    return _wrap_data_handle(
        self._client.Transpose(_unwrap_data_handle(operand), permutation))

  def Rev(self, operand, dimensions):
    """Rev op."""
    return _wrap_data_handle(
        self._client.Rev(_unwrap_data_handle(operand), dimensions))

  def Select(self, pred, on_true, on_false):
    """Element-wise selection op.

    Constructs an output array from elements of two input arrays, based on the
    values of a predicate array.
    """
    return _wrap_data_handle(
        self._client.Select(
            _unwrap_data_handle(pred),
            _unwrap_data_handle(on_true),
            _unwrap_data_handle(on_false)))

  def Slice(self, operand, start_indices, limit_indices, strides=None):
    """Enqueues a slice operation onto the computation.

    Args:
      operand: ComputationDataHandle for the N dimensional array to be sliced.
      start_indices: iterable of N integers containing the starting indices of
        the slice for each dimension.
      limit_indices: iterable of N integers containing the ending indices
        (exclusive) of the slice for each dimension.
      strides: optional iterable of N integers containing the stride sizes for
        each dimension.

    Returns:
      A ComputationDataHandle representing the added Slice op.
    """
    if strides is None:
      start_indices = list(start_indices)
      strides = [1] * len(start_indices)
    return _wrap_data_handle(
        self._client.Slice(
            _unwrap_data_handle(operand),
            start_indices,
            limit_indices,
            strides))

  def DynamicSlice(self, operand, start_indices, slice_sizes):
    """Enqueues a slice op with dynamic start indices onto the computation.

    Args:
      operand: ComputationDataHandle for the N dimensional array to be sliced.
      start_indices: ComputationDataHandle for the 1D array of N integers
        containing the starting indices of the slice.
      slice_sizes: iterable of N integers containing the slice sizes in each
        dimension.

    Returns:
      A ComputationDataHandle representing the added DynamicSlice op.
    """
    return _wrap_data_handle(
        self._client.DynamicSlice(
            _unwrap_data_handle(operand),
            _unwrap_data_handle(start_indices),
            slice_sizes))

  def DynamicUpdateSlice(self, operand, update, start_indices):
    """Enqueues a dynamic update slice operation onto the computation.

    Args:
      operand: ComputationDataHandle for the N dimensional array to be updated.
      update: N dimensional array comprising the slice update.
      start_indices: Rank-1 array of N integers comprising the starting indices
        of the slice along each dimension.
    Returns:
      A ComputationDataHandle representing the added DynamicUpdateSlice op.
    """
    return _wrap_data_handle(
        self._client.DynamicUpdateSlice(
            _unwrap_data_handle(operand),
            _unwrap_data_handle(update),
            _unwrap_data_handle(start_indices)))

  def Tuple(self, *ops):
    """Enqueues a tuple operation onto the computation.

    Args:
      ops: a sequence of tuple operands (each a ComputationDataHandle).

    Returns:
      A ComputationDataHandle representing the added Tuple op.
    """
    return _wrap_data_handle(self._client.Tuple(_unwrap_data_handles(ops)))

  def GetTupleElement(self, tup, index):
    """Enqueues a 'get tuple element' operation onto the computation.

    Args:
      tup: the tuple operand (a ComputationDataHandle).
      index: numeric index to select from the tuple.

    Returns:
      A ComputationDataHandle representing the added GetTupleElement op.
    """
    return _wrap_data_handle(
        self._client.GetTupleElement(_unwrap_data_handle(tup), index))

  def Call(self, computation_to_apply, operands):
    """Enqueues a call operation onto the computation.

    Args:
      computation_to_apply: a Computation object.
      operands: an iterable of ComputationDataHandle. The number and types of
        operands must match the arity of computation_to_apply.

    Returns:
      A ComputationDataHandle representing the added call op.
    """
    return _wrap_data_handle(
        self._client.Call(computation_to_apply.c_local_computation,
                          _unwrap_data_handles(operands)))

  def Map(self, operands, computation_to_apply, dimensions, static_operands=()):
    """Enqueues a map operation onto the computation.

    Args:
      operands: an iterable of ComputationDataHandle.
      computation_to_apply: a Computation object.
      dimensions: dimensions over which to apply map the function.
      static_operands: auxiliary arguments passed to the applied computation.

    Returns:
      A ComputationDataHandle representing the added Map op.
    """
    return _wrap_data_handle(
        self._client.Map(
            _unwrap_data_handles(operands),
            computation_to_apply.c_local_computation,
            dimensions,
            _unwrap_data_handles(static_operands)))

  def Reduce(self, operand, init_value, computation_to_apply, dimensions):
    """Enqueues a reduction operation onto the computation.

    Args:
      operand: reduction operand (ComputationDataHandle).
      init_value: reduction initial value (ComputationDataHandle).
      computation_to_apply: a Computation object - binary reduction function.
      dimensions: sequence of dimensions (integers) to reduce on.

    Returns:
      A ComputationDataHandle representing the added Reduce op.
    """
    return _wrap_data_handle(
        self._client.Reduce(
            _unwrap_data_handle(operand),
            _unwrap_data_handle(init_value),
            computation_to_apply.c_local_computation,
            dimensions))

  def RngNormal(self, mu, sigma, dims):
    """Enqueues an RngNormal operation onto the computation.

    Args:
      mu: A ComputationDataHandle to an F32 scalar specifying the mean.
      sigma: A ComputationDataHandle to an F32 scalar specifying the standard
        deviation.
      dims: A 1D array-like of nonnegative integers specifying the dimensions.

    Returns: a ComputationDataHandle to the generated array of F32 values.
    """
    shape = Shape(self.GetShape(mu).np_dtype, dims)
    return _wrap_data_handle(
        self._client.RngNormal(
            _unwrap_data_handle(mu), _unwrap_data_handle(sigma),
            _unwrap_shape(shape)))

  def RngUniform(self, a, b, dims):
    """Enqueues an RngUniform operation onto the computation.

    Args:
      a: a ComputationDataHandle to an F32, S32, or U32 scalar (consistent with
        the type of b) specifying the low end of the interval [a, b) over which
        values are generated.
      b: a ComputationDataHandle to an F32, S32, or U32 scalar (consistent with
        the type of a) specifying the high end of the interval [a, b) over which
        values are generated.
      dims: A 1D array-like of nonnegative integers specifying the dimensions.

    Returns: a ComputationDataHandle to the generated array of values with the
      same numeric type (F32, S32, or U32) as the arguments a and b.
    """
    shape = Shape(self.GetShape(a).np_dtype, dims)
    return _wrap_data_handle(
        self._client.RngUniform(
            _unwrap_data_handle(a), _unwrap_data_handle(b),
            _unwrap_shape(shape)))

  def RngBernoulli(self, mean, dims):
    """Enqueues an RngBernoulli operation onto the computation.

    Args:
      mean: A ComputationDataHandle to an F32 scalar specifying the mean.
      dims: A 1D array-like of nonnegative integers specifying the dimensions.

    Returns: a ComputationDataHandle to the generated array of U32 values.
    """
    shape = Shape(np.dtype(np.uint32), dims)
    return _wrap_data_handle(
        self._client.RngBernoulli(
            _unwrap_data_handle(mean), _unwrap_shape(shape)))

  def While(self, cond, body, init):
    """Enqueues a While operation onto the computation.

    Args:
      cond: a Computation for the loop condition, which has type T -> PRED
      body: a Computation for the loop body, which has type T -> T
      init: an ComputationDataHandle for the initial parameter, which has type T

    Returns: a ComputationDataHandle representing the While operation.
    """
    return _wrap_data_handle(
        self._client.While(cond.c_local_computation,
                           body.c_local_computation,
                           _unwrap_data_handle(init)))

  def Dot(self, lhs, rhs):
    """Matrix multiplication between lhs and rhs."""
    return _wrap_data_handle(
        self._client.Dot(_unwrap_data_handle(lhs), _unwrap_data_handle(rhs)))

  def Conv(self, lhs, rhs, window_strides, padding):
    """Enqueues a Conv operation onto the computation.

    Args:
      lhs: ComputationDataHandle for the rank N+2 array of inputs.
      rhs: ComputationDataHandle for the rank N+2 array of kernel weights.
      window_strides: length-N array-like of integer kernel strides.
      padding: PaddingType representing either 'SAME' or 'VALID' padding.

    Returns: a ComputationDataHandle representing the Conv operation.
    """
    if padding == PaddingType.SAME:
      lhs_dims = self.GetShape(lhs).dimensions()
      rhs_dims = self.GetShape(rhs).dimensions()
      in_shape, filter_shape = lhs_dims[2:], rhs_dims[2:]
      out_shape = np.ceil(np.true_divide(in_shape, window_strides)).astype(int)
      pad_sizes = [max((out_size - 1) * stride + filter_size - in_size, 0)
                   for out_size, stride, filter_size, in_size
                   in zip(out_shape, window_strides, filter_shape, in_shape)]
      pads = [(pad_size // 2, pad_size - pad_size // 2)
              for pad_size in pad_sizes]
    else:
      pads = [(0, 0)] * len(window_strides)
    dimension_numbers = self._GetConvDimensionNumbers(len(window_strides))
    return _wrap_data_handle(
        self._client.ConvGeneralDilated(_unwrap_data_handle(lhs),
                                        _unwrap_data_handle(rhs),
                                        window_strides,
                                        pads,
                                        (),
                                        (),
                                        dimension_numbers))

  def ConvWithGeneralPadding(self, lhs, rhs, window_strides, padding,
                             lhs_dilation, rhs_dilation):
    """Enqueues a ConvWithGeneralPadding operation onto the computation.

    Args:
      lhs: ComputationDataHandle for the rank N+2 array of inputs.
      rhs: ComputationDataHandle for the rank N+2 array of kernel weights.
      window_strides: length-N array-like of kernel strides.
      padding: length-N array-like of pairs of integers of (low, high) padding.
      lhs_dilation: length-N array-like of dilation factors.
      rhs_dilation: length-N array-like of dilation factors.

    Returns:
      A ComputationdataHandle representing the added ConvWithGeneralPadding op.
    """
    dimension_numbers = self._GetConvDimensionNumbers(len(window_strides))
    return _wrap_data_handle(
        self._client.ConvGeneralDilated(_unwrap_data_handle(lhs),
                                        _unwrap_data_handle(rhs),
                                        window_strides,
                                        padding,
                                        lhs_dilation,
                                        rhs_dilation,
                                        dimension_numbers))

  def _GetConvDimensionNumbers(self, num_spatial_dims):
    """Create ConvolutionDimensionNumbers proto for convolutions."""
    nd = num_spatial_dims
    dimension_numbers = xla_data_pb2.ConvolutionDimensionNumbers()
    dimension_numbers.input_batch_dimension = 0
    dimension_numbers.input_feature_dimension = 1
    dimension_numbers.output_batch_dimension = 0
    dimension_numbers.output_feature_dimension = 1
    dimension_numbers.kernel_output_feature_dimension = 0
    dimension_numbers.kernel_input_feature_dimension = 1
    dimension_numbers.input_spatial_dimensions.extend(range(2, 2 + nd))
    dimension_numbers.kernel_spatial_dimensions.extend(range(2, 2 + nd))
    dimension_numbers.output_spatial_dimensions.extend(range(2, 2 + nd))
    return dimension_numbers


def _forward_methods_to_local_builder():
  """Forward remaining ComputationBuilder methods to the C API.

  Set up methods, corresponding to unary and binary XLA operations,
  whose calls are forwarded in a boilerplate manner to the underlying
  LocalComputationBuilder C-extension API.
  """

  def forward_to_local_builder_with_handles(target_method, is_binop=False):
    """Generate a forwarding method that wraps/unwraps data handles."""

    def forward(self, *args, **kwargs):
      unwrapped_args = [_unwrap_data_handle(arg) for arg in args]

      if is_binop and len(unwrapped_args) < 3:
        unwrapped_args.append(kwargs.get('broadcast_dimensions', ()))

      return _wrap_data_handle(
          target_method(
              self._client,  # pylint: disable=protected-access
              *unwrapped_args))

    return forward

  for method_name in _UNARY_OPS:
    forward = forward_to_local_builder_with_handles(
        getattr(c_api.LocalComputationBuilder, method_name))
    forward.__name__ = method_name
    setattr(ComputationBuilder, method_name, forward)

  for method_name in _BINARY_OPS:
    forward = forward_to_local_builder_with_handles(
        getattr(c_api.LocalComputationBuilder, method_name), is_binop=True)
    forward.__name__ = method_name
    setattr(ComputationBuilder, method_name, forward)


_forward_methods_to_local_builder()


def initialize_replica_count(replica_count):
  """Initializes the desired replica count to use on XLA service init.

  Args:
    replica_count: number of replicas that are desired for set up during XLA
      initalization.

  Raises:
    A runtime exception if the XLA service has already been initialized.
  """
  c_api.InitializeReplicaCount(replica_count)


def get_replica_count():
  """Returns the current replica count used for the XLA service.

  Note: this will return a value whether the XLA service has been initialized
  yet or not.
  """
  return c_api.GetReplicaCount()
