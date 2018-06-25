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
"""TPU Distribution Strategy.

This is experimental.  It's not ready for general use.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import tpu
from tensorflow.contrib.distribute.python import one_device_strategy
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest


class TPUStrategy(one_device_strategy.OneDeviceStrategy):
  """Experimental TPU distribution strategy implementation."""

  def __init__(self, num_cores_per_host=2):
    # TODO(isaprykin): Generalize the defaults.  They are currently tailored for
    # the unit test.
    super(TPUStrategy, self).__init__('/cpu:0')
    # TODO(isaprykin): Auto-detect number of cores and hosts.
    self._num_cores_per_host = num_cores_per_host
    # TODO(priyag): This should not be hardcoded here.
    self._host = '/task:0/device:CPU:0'

  def distribute_dataset(self, dataset_fn):
    # TODO(priyag): Perhaps distribute across cores here.
    return self._call_dataset_fn(dataset_fn)

  # TODO(priyag): Deal with OutOfRange errors.
  def run_steps_on_dataset(self, fn, iterator, iterations):
    # Enqueue ops
    shapes = nest.flatten(iterator.output_shapes)
    if any([not s.is_fully_defined() for s in shapes]):
      raise ValueError(
          'TPU currently requires fully defined shapes. Either use '
          'set_shape() on the input tensors or use '
          'dataset.apply(map_and_batch(..., drop_remainder=True)).')
    types = nest.flatten(iterator.output_types)

    def enqueue_ops_fn():
      """Enqueue ops for one iteration."""
      control_deps = []
      sharded_inputs = []
      with ops.device(self._host):
        for _ in range(self._num_cores_per_host):
          # Use control dependencies to ensure a deterministic ordering.
          with ops.control_dependencies(control_deps):
            inputs = nest.flatten(iterator.get_next())
            control_deps.extend(inputs)
            sharded_inputs.append(inputs)

      enqueue_ops = []
      for core_id, shard_input in enumerate(sharded_inputs):
        enqueue_ops.append(
            tpu_ops.infeed_enqueue_tuple(
                inputs=shard_input, shapes=shapes, device_ordinal=core_id))
      return enqueue_ops

    def enqueue_ops_loop_body(i):
      with ops.control_dependencies(enqueue_ops_fn()):
        return i + 1

    with ops.device(self._host):
      enqueue_ops = control_flow_ops.while_loop(
          lambda i: i < iterations,
          enqueue_ops_loop_body,
          [constant_op.constant(0)],
          parallel_iterations=1)

    # Dequeue ops
    def dequeue_fn():
      dequeued = tpu.infeed_dequeue_tuple(dtypes=types, shapes=shapes)
      return nest.pack_sequence_as(iterator.output_shapes, dequeued)

    # Wrap `fn` for repeat.
    run_fn = lambda: fn(dequeue_fn())

    # Repeat
    def iterate_on_tpu():
      return tpu.repeat(iterations, run_fn, [])

    # Re-write and distribute computation.
    tpu_result = tpu.batch_parallel(
        iterate_on_tpu, [], num_shards=self._num_cores_per_host)

    return control_flow_ops.group(tpu_result, enqueue_ops)

  def _call_for_each_tower(self, fn, *args, **kwargs):
    kwargs.pop('run_concurrently', None)
    with one_device_strategy._OneDeviceTowerContext(self):  # pylint: disable=protected-access
      return fn(*args, **kwargs)

  def _reduce(self, method_string, value, destinations):
    del destinations  # TPU is graph mode only.  Rely on implicit Send/Recv.
    if method_string == 'mean':
      # TODO(jhseu):  Revisit once we support model-parallelism.
      value *= (1. / self._num_cores_per_host)
    return tpu_ops.cross_replica_sum(value)

  @property
  def num_towers(self):
    return self._num_cores_per_host
