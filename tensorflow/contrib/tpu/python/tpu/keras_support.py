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
"""*Experimental* support for running Keras models on the TPU.

To use, wrap your model with the `keras_support.tpu_model` function.

Example usage:

```
# Must activate before building TPU models
keras_support.setup_tpu_session(master_address)

image = tf.keras.layers.Input(shape=(28, 28, 3), name='image')
c1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3))( image)
flattened = tf.keras.layers.Flatten()(c1)
logits = tf.keras.layers.Dense(10, activation='softmax')(flattened)
model = tf.keras.Model(inputs=[image], outputs=[logits])
model = keras_support.tpu_model(model)

# Only TF optimizers are currently supported.
model.compile(optimizer=tf.train.AdamOptimizer(), ...)

# `images` and `labels` should be Numpy arrays.  Support for tensor input
# (e.g. datasets) is planned.
model.fit(images, labels)

# Invoke before shutting down
keras_support.shutdown_tpu_session()
```
"""

# pylint: disable=protected-access

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import time

from tensorflow.contrib.cluster_resolver.python.training import tpu_cluster_resolver
from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.contrib.tpu.proto import compilation_result_pb2 as tpu_compilation_result
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers as keras_optimizers
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging


class TPUEmbedding(embeddings.Embedding):
  """TPU compatible embedding layer.

  The default Keras layer is not TPU compatible.  This layer is a drop-in
  replacement: it has the same behavior and will work on CPU and GPU devices.
  """

  def build(self, input_shape):
    if input_shape[0] is None:
      raise ValueError(
          'TPUEmbeddings must have a fixed input_length or input shape.')
    return super(TPUEmbedding, self).build(input_shape)

  def call(self, inputs):
    if K.dtype(inputs) != 'int32':
      inputs = math_ops.cast(inputs, 'int32')

    inputs = array_ops.one_hot(inputs, self.input_dim)
    return math_ops.tensordot(inputs, self.embeddings, 1)


class TPUModelOp(
    collections.namedtuple(
        'TPUModelOp',
        ['compile_op', 'execute_op', 'infeed_tensors', 'infeed_op',
         'outfeed_op'])):
  pass


def _valid_name(tensor_name):
  """Return a valid tensor name (strips '/', ':', etc)."""
  return re.sub('[^a-zA-Z0-9_-]+', '', tensor_name)


def _replicated_optimizer(opt, num_replicas):
  """Wrap the optimizer `opt` with CrossShardOptimizer if applicable."""
  if num_replicas == 1:
    return opt
  return keras_optimizers.TFOptimizer(
      optimizer=tpu_optimizer.CrossShardOptimizer(opt.optimizer)
  )


class TPUFunction(object):
  """K.function compatible interface for invoking a TPU compiled function.

  Recompilation is triggered on-demand for each set of new inputs shapes: the
  results are cached for future execution.  We expect most computations will
  be dominated by a standard batch-size, followed by a straggler batch for
  the end of training or evaluation.

  All `inputs` and `outputs` will be loaded via the infeed and outfeed queues
  instead of being injected as `feed_dict` items or fetches.
  """

  def __init__(self, model, execution_mode, num_replicas=1):
    self.model = model
    self.execution_mode = execution_mode
    self._compilation_cache = {}
    self.num_replicas = num_replicas

  def _specialize_model(self, input_specs):
    """Specialize `self.model` (a Keras model) for the given input shapes."""
    # Re-create our input and output layers inside our subgraph.  They will be
    # attached to the true computation when we clone our model in `tpu_fn`.
    K.set_learning_phase(
        self.execution_mode == model_fn_lib.ModeKeys.TRAIN
    )

    # functools.partial and callable objects are not supported by tpu.rewrite
    def _model_fn():
      """Compute fit/eval/predict for the TPU."""
      is_training = self.execution_mode == model_fn_lib.ModeKeys.TRAIN
      is_test = self.execution_mode == model_fn_lib.ModeKeys.EVAL
      is_predict = self.execution_mode == model_fn_lib.ModeKeys.PREDICT

      # During train/eval, we infeed our features as well as labels.
      if is_training or is_test:
        infeed_layers = self.model._input_layers + self.model._output_layers
      else:
        infeed_layers = self.model._input_layers

      # Generate our infeed operation to read features & labels.
      infeed_tensors = tpu_ops.infeed_dequeue_tuple(
          dtypes=[spec.dtype for spec in input_specs],
          shapes=[spec.shape for spec in input_specs],
          name='infeed-%s' % self.execution_mode)

      assert len(infeed_tensors) == len(infeed_layers), (
          'Infeed inputs did not match model: %s vs %s', (infeed_layers,
                                                          infeed_tensors))

      tpu_targets = []
      tpu_inputs = []

      # Sort infeed outputs into inputs and labels for calling our Keras model.
      for tensor, layer in zip(infeed_tensors, infeed_layers):
        if layer in self.model._input_layers:
          tpu_inputs.append(layers.Input(name=layer.name, tensor=tensor))
        if layer in self.model._output_layers:
          tpu_targets.append(tensor)

      # Call our model with our infeed inputs (re-using the weights).
      model_outputs = self.model(tpu_inputs)
      child_model = models.Model(inputs=tpu_inputs, outputs=model_outputs)

      if is_training or is_test:
        child_model.compile(
            optimizer=_replicated_optimizer(self.model.optimizer,
                                            self.num_replicas),
            loss=self.model.loss,
            loss_weights=self.model.loss_weights,
            metrics=self.model.metrics,
            weighted_metrics=self.model.weighted_metrics,
            target_tensors=tpu_targets,
        )

      # Compute our outfeed depending on the execution mode
      if is_training:
        child_model._make_train_function()
        self._outfeed_spec = [
            tensor_spec.TensorSpec(tensor.shape, tensor.dtype, tensor.name)
            for tensor in child_model.train_function.outputs
        ]
        return [
            child_model.train_function.updates_op,
            tpu_ops.outfeed_enqueue_tuple(
                child_model.train_function.outputs,
                name='outfeed-enqueue-train')
        ]
      elif is_test:
        child_model._make_test_function()
        self._outfeed_spec = [
            tensor_spec.TensorSpec(tensor.shape, tensor.dtype, tensor.name)
            for tensor in child_model.test_function.outputs
        ]
        return [
            tpu_ops.outfeed_enqueue_tuple(
                child_model.test_function.outputs,
                name='outfeed-enqueue-test')
        ]
      elif is_predict:
        child_model._make_predict_function()
        self._outfeed_spec = [
            tensor_spec.TensorSpec(tensor.shape, tensor.dtype, tensor.name)
            for tensor in child_model.predict_function.outputs
        ]
        return [
            tpu_ops.outfeed_enqueue_tuple(
                child_model.predict_function.outputs,
                name='outfeed-enqueue-predict',
            )
        ]
      else:
        assert False, 'Unexpected execution mode: %s' % self.execution_mode

    # Capture outfeed metadata computed during the rewrite.
    self._outfeed_spec = None

    # Generate out TPU operations using `tpu.split_compile_and_replicate`.
    # `compile_op` can be used to test the TPU model compiles before execution.
    # `execute op` replicates `_model_fn` `num_replicas` times, with each shard
    # running on a different logical core.
    compile_op, execute_op = tpu.split_compile_and_replicate(
        _model_fn, inputs=[[]] * self.num_replicas)

    # Generate CPU side operations to enqueue features/labels and dequeue
    # outputs from the model call.
    infeed_op = []
    outfeed_op = []
    shard_infeed_tensors = []

    for shard_id in range(self.num_replicas):
      with ops.device('/device:TPU:%d' % shard_id):
        infeed_tensors = []
        for spec in input_specs:
          infeed_tensors.append(
              array_ops.placeholder(
                  dtype=spec.dtype,
                  shape=spec.shape,
                  name='infeed-enqueue-%s-%d' % (spec.name, shard_id)))
        shard_infeed_tensors.append(infeed_tensors)

        infeed_op.append(tpu_ops.infeed_enqueue_tuple(
            infeed_tensors, [spec.shape for spec in input_specs],
            name='infeed-enqueue-%s-%d' % (self.execution_mode, shard_id)))

        outfeed_op.extend(tpu_ops.outfeed_dequeue_tuple(
            dtypes=[spec.dtype for spec in self._outfeed_spec],
            shapes=[spec.shape for spec in self._outfeed_spec],
            name='outfeed-dequeue-%s-%d' % (self.execution_mode, shard_id)))

    return TPUModelOp(
        compile_op, execute_op, infeed_tensors=shard_infeed_tensors,
        infeed_op=infeed_op, outfeed_op=outfeed_op)

  def _test_model_compiles(self, tpu_model_ops):
    """Verifies that the given TPUModelOp can be compiled via XLA."""
    session = K.get_session()

    logging.info('Started compiling')
    start_time = time.clock()

    result = session.run(tpu_model_ops.compile_op)
    proto = tpu_compilation_result.CompilationResultProto()
    proto.ParseFromString(result)
    if proto.status_error_message:
      raise RuntimeError(
          'Compilation failed: {}'.format(proto.status_error_message))

    end_time = time.clock()
    logging.info('Finished compiling. Time elapsed: %s secs',
                 end_time - start_time)

  def _split_tensors(self, inputs):
    """Split input data across shards.

    Each input is sliced along the batch axis.

    Args:
      inputs: List of Numpy arrays to run on the TPU.

    Returns:
      List of lists containing the input to feed to each TPU shard.
    """
    if self.num_replicas == 1:
      return [inputs]

    batch_size = inputs[0].shape[0]
    assert batch_size % self.num_replicas == 0, (
        'batch_size must be divisible by num_replicas')
    shard_size = batch_size // self.num_replicas
    input_list = []
    for index in range(self.num_replicas):
      shard_inputs = [x[index * shard_size:(index + 1) * shard_size]
                      for x in inputs]
      input_list.append(shard_inputs)
    return input_list

  def __call__(self, inputs):
    assert isinstance(inputs, list)

    # Strip sample weight from inputs
    if (self.execution_mode == model_fn_lib.ModeKeys.TRAIN or
        self.execution_mode == model_fn_lib.ModeKeys.EVAL):
      input_tensors = self.model._feed_inputs + self.model._feed_targets
      inputs = inputs[:len(input_tensors)]
    else:
      input_tensors = self.model._feed_inputs

    shard_inputs = self._split_tensors(inputs)
    del inputs  # To avoid accident usage.

    # Compute an input specification (used to generate infeed enqueue and
    # dequeue operations).  We use the shape from our input array and the
    # dtype from our model.  A user may pass in a float64 for a float32
    # input: for model compatibility we still must generate a float32 infeed.
    input_specs = []

    # We use the shape and dtype from the first shard to compute the input
    # metadata (`input_specs`); all replicas have the same type and shape.
    for tensor, ary in zip(input_tensors, shard_inputs[0]):
      input_specs.append(
          tensor_spec.TensorSpec(ary.shape, tensor.dtype,
                                 _valid_name(tensor.name)))

    # XLA requires every operation in the graph has a fixed shape.  To
    # handle varying batch sizes we recompile a new sub-graph for each
    # unique input shape.
    shape_key = tuple([tuple(spec.shape.as_list()) for spec in input_specs])

    if shape_key not in self._compilation_cache:
      logging.info('New input shapes; (re-)compiling: mode=%s, %s',
                   self.execution_mode, input_specs)
      new_tpu_model_ops = self._specialize_model(input_specs)
      self._compilation_cache[shape_key] = new_tpu_model_ops
      self._test_model_compiles(new_tpu_model_ops)

    tpu_model_ops = self._compilation_cache[shape_key]

    infeed_dict = {}
    for infeed_tensors, inputs in zip(tpu_model_ops.infeed_tensors,
                                      shard_inputs):
      for tensor, value in zip(infeed_tensors, inputs):
        infeed_dict[tensor] = value

    session = K.get_session()
    _, _, outfeed_outputs = session.run([
        tpu_model_ops.infeed_op, tpu_model_ops.execute_op,
        tpu_model_ops.outfeed_op
    ], infeed_dict)

    # TODO(xiejw): Decide how to reduce outputs, or just discard all but first.
    return outfeed_outputs[:len(outfeed_outputs) // self.num_replicas]


@experimental
def setup_tpu_session(tpu_name_or_address):
  """Initializes and returns a Keras/TF session connected the TPU `master`.

  Args:
    tpu_name_or_address: A string that is either the name of the Cloud TPU,
      the grpc address of the Cloud TPU, or (Googlers only) the BNS name of the
      Cloud TPU. If tpu_name_or_address is None, the TPUClusterResolver will
      examine the environment to determine a potential Cloud TPU to use.

  Returns:
    A `tf.Session`.
  """
  cluster_resolver = tpu_cluster_resolver.TPUClusterResolver(
      tpu_name_or_address)
  cluster_spec = cluster_resolver.cluster_spec()
  session = tf_session.Session(
      target=cluster_resolver.master(),
      config=config_pb2.ConfigProto(
          isolate_session_state=True))
  if cluster_spec:
    session.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
  K.set_session(session)
  K.get_session().run(tpu.initialize_system())
  return session


@experimental
def shutdown_tpu_session(session=None):
  """Shutdown the TPU attached to session.

  This should be called to cleanly shut down the TPU system before the client
  exits.

  Args:
    session: Session to shutdown, or None to use the default session.

  Returns:

  """
  if session is None:
    session = K.get_session()

  session.run(tpu.shutdown_system())


class KerasTPUModel(models.Model):
  """TPU compatible Keras model wrapper."""

  def __init__(self, inputs, outputs, name, replicas=1):
    super(models.Model, self).__init__(  # pylint: disable=bad-super-call
        inputs=inputs,
        outputs=outputs,
        name=name,
    )
    self.predict_function = None
    self.test_function = None
    self.train_function = None
    self.replicas = replicas

  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              **kwargs):
    if sample_weight_mode:
      raise ValueError('sample_weight_mode not supported for TPU execution.')
    if weighted_metrics:
      raise ValueError('weighted_metrics not supported for TPU execution.')
    if target_tensors:
      raise ValueError('target_tensors is not supported for TPU execution.')

    super(KerasTPUModel, self).compile(optimizer, loss, metrics, loss_weights,
                                       sample_weight_mode, weighted_metrics,
                                       target_tensors, **kwargs)

    # Keras optimizers are not compatible with TPU rewrite
    if not isinstance(self.optimizer, keras_optimizers.TFOptimizer):
      raise ValueError(
          'Optimizer must be a TFOptimizer, got: %s' % self.optimizer)

  def _make_train_function(self):
    if not self.train_function:
      self.train_function = TPUFunction(self, model_fn_lib.ModeKeys.TRAIN,
                                        num_replicas=self.replicas)

    return self.train_function

  def _make_test_function(self):
    if not self.test_function:
      self.test_function = TPUFunction(self, model_fn_lib.ModeKeys.EVAL)
    return self.test_function

  def _make_predict_function(self):
    if not self.predict_function:
      self.predict_function = TPUFunction(self, model_fn_lib.ModeKeys.PREDICT)
    return self.predict_function

  def cpu_model(self):
    cpu_model = models.Model(
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name,
    )

    if self.optimizer:
      cpu_model.compile(
          optimizer=self.optimizer,
          loss=self.loss,
          metrics=self.metrics,
          loss_weights=self.loss_weights,
      )

    return cpu_model


def _validate_shapes(model):
  """Validate that all layers in `model` have constant shape."""
  for layer in model.layers:
    if isinstance(layer.input_shape, tuple):
      input_shapes = [layer.input_shape]
    else:
      input_shapes = layer.input_shape

    if isinstance(layer.output_shape, tuple):
      output_shapes = [layer.output_shape]
    else:
      output_shapes = layer.output_shape

    for shape in input_shapes + output_shapes:
      for dim in shape[1:]:
        if dim is None:
          raise ValueError(
              """
Layer %(layer)s has a variable shape in a non-batch dimension.  TPU models must
have constant shapes for all operations.

You may have to specify `input_length` for RNN/TimeDistributed layers.

Layer: %(layer)s
Input shape: %(input_shape)s
Output shape: %(output_shape)s
  """ % {
      'layer': layer,
      'input_shape': layer.input_shape,
      'output_shape': layer.output_shape
      })


@experimental
def tpu_model(model, replicas=None):
  """Runs a model on TPU(s).

  Usage:
  ```
  a = Input(shape=(32,))
  b = Dense(32)(a)
  model = Model(inputs=a, outputs=b)

  model = keras_support.tpu_model(model)
  model.compile(
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=1.0),
      ...)
  ```

  If `replicas` is set, replicates the model computation on all TPU cores. The
  model computation is replicated `num_replicas` times; each shard will run on a
  different TPU core.

  Limitation: Currently, replication is only supported for training.

  Usage:
  ```
  a = Input(shape=(32,))
  b = Dense(32)(a)
  model = Model(inputs=a, outputs=b)

  model = keras_support.tpu_model(model, replicas=2)
  model.compile(
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=1.0),
      ...)
  ```

  Args:
    model: A `KerasTPUModel`.
    replicas: (Optional) Int, number of TPU cores which to create model
        replicas. If `None`, the model runs on single core only, i.e., no
        replication.

  Returns:
    A new `KerasTPUModel` instance.
  """
  _validate_shapes(model)
  # TODO(xiejw): Validate TPU model. TPUModel only?
  # TODO(xiejw): Validate replicas. Full or 1. Shall we allow subset?
  # TODO(xiejw): Adds reduction option.
  replicas = 1 if replicas is None else replicas
  return KerasTPUModel(
      inputs=model.inputs, outputs=model.outputs, name=model.name,
      replicas=replicas)
