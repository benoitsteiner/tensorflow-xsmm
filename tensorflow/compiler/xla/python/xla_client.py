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

import itertools

import numpy as np

from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.python import pywrap_xla as c_api

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
    xla_data_pb2.PRED: np.dtype(np.bool),
    xla_data_pb2.TUPLE: np.dtype(np.object),
}

DTYPE_TO_XLA_ELEMENT_TYPE = {
    str(v): k
    for k, v in XLA_ELEMENT_TYPE_TO_DTYPE.items()
}


class Shape(object):
  """XLA shape.

  Represents an XLA shape by a corresponding Python/Numpy type and a
  list of dimensions, which are themselves Shapes in case this one
  represents an XLA tuple.
  """

  def __init__(self, np_dtype, dimensions):
    self.np_dtype = np_dtype
    self._dimensions = dimensions

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

  def Trans(self, operand):
    """Specialized matrix transpose op."""
    return _wrap_data_handle(
        self._client.Transpose(_unwrap_data_handle(operand), [1, 0]))

  def Transpose(self, operand, permutation):
    """Transpose op."""
    return _wrap_data_handle(
        self._client.Transpose(_unwrap_data_handle(operand), permutation))

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
