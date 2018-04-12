# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow-related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables


def smart_cond(pred, true_fn=None, false_fn=None, name=None):
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Arguments:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if isinstance(pred, variables.Variable):
    return control_flow_ops.cond(
        pred, true_fn=true_fn, false_fn=false_fn, name=name)
  return smart_module.smart_cond(
      pred, true_fn=true_fn, false_fn=false_fn, name=name)


def constant_value(pred):
  """Return the bool value for `pred`, or None if `pred` had a dynamic value.

  Arguments:
    pred: A scalar, either a Python bool or a TensorFlow boolean variable
      or tensor, or the Python integer 1 or 0.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  Raises:
    TypeError: If `pred` is not a Variable, Tensor or bool, or Python
      integer 1 or 0.
  """
  # Allow integer booleans.
  if isinstance(pred, int):
    if pred == 1:
      pred = True
    elif pred == 0:
      pred = False

  if isinstance(pred, variables.Variable):
    return None
  return smart_module.smart_constant_value(pred)
