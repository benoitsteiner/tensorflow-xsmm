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
"""Keras training and evaluation routines for eager execution.
"""
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np

from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras._impl.keras import backend
from tensorflow.python.keras._impl.keras import callbacks as cbks
from tensorflow.python.keras._impl.keras import losses
from tensorflow.python.keras._impl.keras import metrics as metrics_module
from tensorflow.python.keras._impl.keras.engine import training_utils
from tensorflow.python.keras._impl.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging


def _get_metrics_info(metric, internal_output_shapes=None, loss_func=None):
  if metric == 'accuracy' or metric == 'acc':
    # custom handling of accuracy
    # (because of class mode duality)
    output_shape = internal_output_shapes
    if output_shape[-1] == 1 or loss_func == losses.binary_crossentropy:
      # case: binary accuracy
      acc_fn = metrics_module.binary_accuracy
    elif loss_func == losses.sparse_categorical_crossentropy:
      # case: categorical accuracy with sparse targets
      acc_fn = metrics_module.sparse_categorical_accuracy
    else:
      acc_fn = metrics_module.categorical_accuracy

    metric_name = 'acc'
    return metric_name, acc_fn
  else:
    metric_fn = metrics_module.get(metric)
    metric_name = metric_fn.__name__
    return metric_name, metric_fn


def _eager_loss_fn(outputs, targets, loss_fn, output_name):
  with backend.name_scope(output_name + '_loss'):
    loss = loss_fn(targets, outputs)
  return loss


def _eager_metrics_fn(model, outputs, targets):
  """Calculates the metrics for each output of the given model.

  Arguments:
      model: The model on which metrics are being calculated.
      outputs: The outputs of the given model.
      targets: The predictions or targets of the given model.

  Returns:
      Returns the metric names and metric results for each output of the model.
  """
  metric_names = []
  metric_results = []
  if not isinstance(outputs, list):
    outputs = [outputs]

  if not isinstance(targets, list):
    targets = [targets]

  for i in range(len(model.outputs)):
    output_metrics = model.nested_metrics[i]
    for nested_output_metric in output_metrics:
      metric_name, metric_fn = _get_metrics_info(
          nested_output_metric, backend.int_shape(model.outputs[i]),
          model.loss_functions[i])

      if len(model.output_names) > 1:
        metric_name = model.output_names[i] + '_' + metric_name
        if metric_name not in model.metrics_names:
          model.metrics_names.append(metric_name)

      with backend.name_scope(metric_name):
        metric_result = metric_fn(outputs[i], targets[i])
        metric_names.append(metric_name)
        metric_results.append(backend.mean(metric_result))

  return metric_names, metric_results


def _model_loss(model, inputs, targets, sample_weights=None, training=False):
  """Calculates the loss for a given model.

  Arguments:
      model: The model on which metrics are being calculated.
      inputs: List of input arrays.
      targets: List of target arrays.
      sample_weights: Optional list of sample weight arrays.
      training: Whether the model should be run in inference or training mode.

  Returns:
     Returns the model output, total loss and loss value calculated using the
     specified loss function. The total loss includes regularization losses and
     applies masking and sample weighting to the loss value.
  """
  total_loss = 0
  if len(inputs) == 1:
    if model._expects_training_arg:
      outs = model.call(inputs[0], training=training)
    else:
      outs = model.call(inputs[0])
  else:
    if model._expects_training_arg:
      outs = model.call(inputs, training=training)
    else:
      outs = model.call(inputs)
  if not isinstance(outs, list):
    outs = [outs]

  if not isinstance(targets, list):
    targets = [targets]

  loss_metrics = []
  with backend.name_scope('loss'):
    for i, loss_fn in enumerate(model.loss_functions):
      if sample_weights:
        weights = sample_weights[i]
      else:
        weights = None

      # TODO(fchollet): support masking; in practice `_keras_mask` is never
      # set in this context currently.
      mask = outs[i]._keras_mask

      weighted_masked_fn = training_utils.weighted_masked_objective(loss_fn)
      with backend.name_scope(model.output_names[i] + '_loss'):
        output_loss = weighted_masked_fn(
            outs[i], targets[i], weights, mask=mask)
      loss_metrics.append(backend.mean(output_loss))

      loss_weight = model.loss_weights_list[i]
      if total_loss is None:
        total_loss = loss_weight * output_loss
      else:
        total_loss += loss_weight * output_loss

    total_loss = backend.mean(total_loss)
    # Add regularization losses
    custom_losses = []
    for layer in model.layers:
      if layer.losses:
        custom_losses += layer.losses

    if custom_losses:
      total_loss += sum(custom_losses)

  return outs, total_loss, loss_metrics


def slice_arrays(arrays, indices, contiguous=True):
  """Slices batches out of provided arrays (workaround for eager tensors).

  Unfortunately eager tensors don't have the same slicing behavior as
  Numpy arrays (they folow  the same slicing behavior as symbolic TF tensors),
  hence we cannot use `generic_utils.slice_arrays` directly
  and we have to implement this workaround based on `concat`. This has a
  performance cost.

  Arguments:
    arrays: Single array or list of arrays.
    indices: List of indices in the array that should be included in the output
      batch.
    contiguous: Boolean flag indicating whether the indices are contiguous.

  Returns:
    Slice of data (either single array or list of arrays).
  """
  if any(tensor_util.is_tensor(x) for x in arrays):
    converted_to_list = False
    if not isinstance(arrays, list):
      converted_to_list = True
      arrays = [arrays]
    if not contiguous:
      entries = [[x[i:i + 1] for i in indices] for x in arrays]
      slices = [array_ops.concat(x, axis=0) for x in entries]
    else:
      slices = [x[indices[0]:indices[-1] + 1] for x in arrays]
    if converted_to_list:
      slices = slices[0]
    return slices
  else:
    return generic_utils.slice_arrays(arrays, indices)


def _process_single_batch(model,
                          inputs,
                          targets,
                          sample_weights=None,
                          training=False):
  """Calculate the loss and gradient for one input batch.

     The model weights are updated if training is set to True.

  Arguments:
      model: Model whose loss has to be calculated.
      inputs: List of input arrays.
      targets: List of target arrays.
      sample_weights: Optional list of sample weight arrays.
      training: The boolean represents if the weights of the model are updated.
              'fit' methods will set this to True while 'evaluate' methods will
              set this to False.

  Returns:
      output of the model, total loss and the loss associated with each output.

  Raises:
      ValueError: If the model has no loss to optimize.
  """
  with backend.learning_phase_scope(1 if training else 0):
    with GradientTape() as tape:
      outs, loss, loss_metrics = _model_loss(model, inputs, targets,
                                             sample_weights=sample_weights,
                                             training=training)
      if loss is None:
        raise ValueError('The model cannot be run '
                         'because it has no loss to optimize.')
    if training:
      if not model._collected_trainable_weights:
        logging.warning('The list of trainable weights is empty. Make sure that'
                        ' you are not setting model.trainable to False before '
                        'compiling the model.')
      else:
        grads = tape.gradient(loss, model._collected_trainable_weights)
        model.optimizer.apply_gradients(zip(grads,
                                            model._collected_trainable_weights))
    return outs, loss, loss_metrics


def train_on_batch(model, inputs, targets, sample_weights=None):
  """Calculates the loss and gradient updates for one input batch.

  Arguments:
      model: Model whose loss has to be calculated.
      inputs: Input batch data.
      targets: Target batch data.
      sample_weights: Sample weight batch data.

  Returns:
      total loss and the loss associated with each output.
  """
  inputs = [
      ops.convert_to_tensor(val, dtype=backend.floatx()) for val in inputs]
  targets = [
      ops.convert_to_tensor(val, dtype=backend.floatx()) for val in targets]
  sample_weights = [
      ops.convert_to_tensor(val, dtype=backend.floatx())
      if val is not None else None for val in sample_weights]
  outs, loss, _ = _process_single_batch(
      model, inputs, targets, sample_weights=sample_weights, training=True)
  if not isinstance(outs, list):
    outs = [outs]
  _, metrics_results = _eager_metrics_fn(
      model, outs, targets)
  if not isinstance(loss, list):
    loss = [loss]
  return loss + metrics_results


def test_on_batch(model, inputs, targets, sample_weights=None):
  """Calculates the loss for one input batch.

  Arguments:
      model: Model whose loss has to be calculated.
      inputs: Input batch data.
      targets: Target batch data.
      sample_weights: Sample weight batch data.

  Returns:
      total loss, loss and metrics associated with each output.
  """
  inputs = [
      ops.convert_to_tensor(val, dtype=backend.floatx()) for val in inputs]
  targets = [
      ops.convert_to_tensor(val, dtype=backend.floatx()) for val in targets]
  sample_weights = [
      ops.convert_to_tensor(val, dtype=backend.floatx())
      if val is not None else None for val in sample_weights]
  outs, loss, loss_metrics = _process_single_batch(
      model, inputs, targets, sample_weights=sample_weights, training=False)
  if not isinstance(outs, list):
    outs = [outs]
  _, metrics_results = _eager_metrics_fn(
      model, outs, targets)
  if not isinstance(loss, list):
    loss = [loss]
  return loss + loss_metrics + metrics_results


def fit_loop(
    model,
    inputs,
    targets,
    sample_weights=None,
    val_inputs=None,
    val_targets=None,
    val_sample_weights=None,
    batch_size=None,
    epochs=100,
    verbose=1,
    callbacks=None,
    shuffle=True,
    callback_metrics=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None):
  """Abstract fit function for eager execution.

  Arguments:
      model: Instance of the model that is being executed in Eager mode.
      inputs: List of input arrays.
      targets: List of target arrays.
      sample_weights: Optional list of sample weight arrays.
      val_inputs: Input data for validation.
      val_targets: Target data for validation.
      val_sample_weights: Sample weight data for validation.
      batch_size: Integer batch size or None if unknown.
      epochs: Number of times to iterate over the data
      verbose: Verbosity mode, 0, 1 or 2
      callbacks: List of callbacks to be called during training
      shuffle: Whether to shuffle the data at the beginning of each epoch
      callback_metrics: List of strings, the display names of the metrics
          passed to the callbacks. They should be the
          concatenation of list the display names of the outputs of
           `f` and the list of display names of the outputs of `f_val`.
      initial_epoch: Epoch at which to start training
          (useful for resuming a previous training run)
      steps_per_epoch: Total number of steps (batches of samples)
          before declaring one epoch finished and starting the
          next epoch. Ignored with the default value of `None`.
      validation_steps: Number of steps to run validation for (only if doing
        validation from data tensors). Ignored with default value of `None`.

  Returns:
      `History` object.

  Raises:
    ValueError: In case of invalid argument values.
  """
  if not batch_size:
    raise ValueError('With eager execution, `batch_size` should be specified.')
  if steps_per_epoch or validation_steps:
    raise ValueError('With eager execution, `steps_per_epoch` and '
                     '`validation_steps` are not valid arguments '
                     '(set `batch_size` instead).')
  # Required for Eager mode
  with backend.learning_phase_scope(1):
    do_validation = False
    if val_inputs:
      do_validation = True
      if (verbose and inputs and hasattr(inputs[0], 'shape') and
          hasattr(val_inputs[0], 'shape')):
        print('Train on %d samples, validate on %d samples' %
              (inputs[0].shape[0], val_inputs[0].shape[0]))
    if validation_steps:
      if steps_per_epoch is None:
        raise ValueError('Can only use `validation_steps` when doing step-wise '
                         'training, i.e. `steps_per_epoch` must be set.')
      do_validation = True

    out_labels = model.metrics_names
    if do_validation:
      callback_metrics = copy.copy(out_labels) + [
          'val_' + n for n in out_labels
      ]
    else:
      callback_metrics = copy.copy(out_labels)

    if sample_weights:
      feed_data = inputs + targets + sample_weights
    else:
      feed_data = inputs + targets
    num_train_samples = training_utils.check_num_samples(
        feed_data,
        batch_size=batch_size,
        steps=steps_per_epoch,
        steps_name='steps_per_epoch')

    if num_train_samples is not None:
      index_array = np.arange(num_train_samples)

    model.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [model.history]
    if verbose:
      if steps_per_epoch is not None:
        count_mode = 'steps'
      else:
        count_mode = 'samples'
      callbacks += [cbks.ProgbarLogger(count_mode)]
    callbacks = cbks.CallbackList(callbacks)

    # it's possible to callback a different model than self
    # (used by Sequential models)
    if hasattr(model, 'callback_model') and model.callback_model:
      callback_model = model.callback_model
    else:
      callback_model = model

    callbacks.set_model(callback_model)

    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps_per_epoch,
        'samples': num_train_samples,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics or [],
    })
    callbacks.on_train_begin()
    callback_model.stop_training = False
    for cbk in callbacks:
      if not val_inputs:
        cbk.validation_data = []
      elif val_sample_weights:
        cbk.validation_data = val_inputs + val_targets + val_sample_weights
      else:
        cbk.validation_data = val_inputs + val_targets

    for epoch in range(initial_epoch, epochs):
      callbacks.on_epoch_begin(epoch)
      epoch_logs = {}
      if shuffle == 'batch':
        index_array = model._batch_shuffle(index_array, batch_size)
      elif shuffle:
        np.random.shuffle(index_array)

      batches = generic_utils.make_batches(num_train_samples, batch_size)

      for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        try:
          inputs_batch = slice_arrays(inputs, batch_ids,
                                      contiguous=not shuffle)
          targets_batch = slice_arrays(targets, batch_ids,
                                       contiguous=not shuffle)
          if sample_weights:
            sample_weights_batch = slice_arrays(sample_weights, batch_ids,
                                                contiguous=not shuffle)
          else:
            sample_weights_batch = None
        except TypeError:
          raise TypeError('TypeError while preparing batch. '
                          'If using HDF5 input data, '
                          'pass shuffle="batch".')
        batch_logs = {}
        batch_logs['batch'] = batch_index
        batch_logs['size'] = len(batch_ids)

        callbacks.on_batch_begin(batch_index, batch_logs)

        inputs_batch = [
            ops.convert_to_tensor(val, dtype=backend.floatx())
            for val in inputs_batch]
        targets_batch = [
            ops.convert_to_tensor(val, dtype=backend.floatx())
            for val in targets_batch]
        if sample_weights:
          sample_weights_batch = [
              ops.convert_to_tensor(val, dtype=backend.floatx())
              if val is not None else None
              for val in sample_weights_batch]

        outs, loss, loss_metrics = _process_single_batch(
            model,
            inputs_batch,
            targets_batch,
            sample_weights=sample_weights_batch,
            training=True)

        if not isinstance(outs, list):
          outs = [outs]

        for l, o in zip(out_labels, outs):
          batch_logs[l] = o
        # Required for Eager mode
        metrics_names, metrics_results = _eager_metrics_fn(
            model, outs, targets_batch)
        batch_logs['loss'] = tensor_util.constant_value(backend.mean(loss))

        # TODO(anjalisridhar): Move this to compile to avoid duplicate code.
        # In graph mode we set the metric names in compile. However in
        # Eager mode we calculate the metrics for each batch in fit_loop.
        # We could calculate the metric names and functions in compile.
        # This would avoid setting the callback parameters separately.
        # We need to do this for the first iteration alone
        for m in metrics_names:
          if m not in callback_metrics:
            callback_metrics.append(m)

        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'steps': steps_per_epoch,
            'samples': num_train_samples,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics or [],
        })

        for k, v in zip(model.metrics_names,
                        [backend.mean(loss)] + loss_metrics + metrics_results):
          batch_logs[k] = tensor_util.constant_value(v)

        callbacks.on_batch_end(batch_index, batch_logs)
        if callback_model.stop_training:
          break

        if batch_index == len(batches) - 1:  # Last batch.
          if do_validation:
            val_outs = test_loop(
                model, val_inputs, val_targets,
                sample_weights=val_sample_weights,
                batch_size=batch_size,
                verbose=0)
            if not isinstance(val_outs, list):
              val_outs = [val_outs]
            # Same labels assumed.
            for l, o in zip(out_labels, val_outs):
              epoch_logs['val_' + l] = o
      callbacks.on_epoch_end(epoch, epoch_logs)
      if callback_model.stop_training:
        break
    callbacks.on_train_end()
    return model.history


def test_loop(model, inputs, targets,
              sample_weights=None,
              batch_size=None,
              verbose=0,
              steps=None):
  """Abstract method to loop over some data in batches.

  Arguments:
      model: Model instance that is being evaluated in Eager mode.
      inputs: List of input arrays.
      targets: List of target arrays.
      sample_weights: Optional list of sample weight arrays.
      batch_size: integer batch size or `None`.
      verbose: verbosity mode.
      steps: Total number of steps (batches of samples)
          before declaring predictions finished.
          Ignored with the default value of `None`.

  Returns:
      Scalar loss (if the model has a single output and no metrics)
      or list of scalars (if the model has multiple outputs
      and/or metrics). The attribute `model.metrics_names` will give you
      the display labels for the scalar outputs.
  """
  with backend.learning_phase_scope(0):
    feed_data = inputs + targets
    if sample_weights:
      feed_data += sample_weights
    num_samples = training_utils.check_num_samples(
        feed_data, batch_size=batch_size, steps=steps, steps_name='steps')
    outs = []
    if verbose == 1:
      progbar = generic_utils.Progbar(target=num_samples)
    batches = generic_utils.make_batches(num_samples, batch_size)
    index_array = np.arange(num_samples)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
      batch_ids = index_array[batch_start:batch_end]
      inputs_batch = slice_arrays(inputs, batch_ids)
      targets_batch = slice_arrays(targets, batch_ids)
      if sample_weights:
        sample_weights_batch = slice_arrays(sample_weights, batch_ids)
      else:
        sample_weights_batch = None

      inputs_batch = [
          ops.convert_to_tensor(val, dtype=backend.floatx())
          for val in inputs_batch]
      targets_batch = [
          ops.convert_to_tensor(val, dtype=backend.floatx())
          for val in targets_batch]
      if sample_weights:
        sample_weights_batch = [
            ops.convert_to_tensor(val, dtype=backend.floatx())
            if val is not None else None
            for val in sample_weights_batch]

      loss_outs, loss, loss_metrics = _model_loss(
          model,
          inputs_batch,
          targets_batch,
          sample_weights=sample_weights_batch,
          training=False)
      _, metrics_results = _eager_metrics_fn(model, loss_outs, targets_batch)
      batch_outs = []
      for _, v in zip(model.metrics_names,
                      [backend.mean(loss)] + loss_metrics + metrics_results):
        batch_outs.append(tensor_util.constant_value(v))

      if isinstance(batch_outs, list):
        if batch_index == 0:
          for batch_out in enumerate(batch_outs):
            outs.append(0.)
        for i, batch_out in enumerate(batch_outs):
          outs[i] += batch_out * len(batch_ids)
      else:
        if batch_index == 0:
          outs.append(0.)
        outs[0] += batch_outs * len(batch_ids)

      if verbose == 1:
        progbar.update(batch_end)
    for i in range(len(outs)):
      outs[i] /= num_samples
    if len(outs) == 1:
      return outs[0]
    return outs


def predict_loop(model, inputs,
                 batch_size=32,
                 verbose=0,
                 steps=None):
  """Abstract method to loop over some data in batches.

  Arguments:
      model:
      inputs: List of input arrays.
      batch_size: integer batch size.
      verbose: verbosity mode.
      steps: Total number of steps (batches of samples)
          before declaring `_predict_loop` finished.
          Ignored with the default value of `None`.

  Returns:
      Array of predictions (if the model has a single output)
      or list of arrays of predictions
      (if the model has multiple outputs).
  """
  with backend.learning_phase_scope(0):
    num_samples = training_utils.check_num_samples(
        inputs, batch_size, steps, 'steps')
    if verbose == 1:
      if steps is not None:
        progbar = generic_utils.Progbar(target=steps)
      else:
        progbar = generic_utils.Progbar(target=num_samples)

    outs = []
    batches = generic_utils.make_batches(num_samples, batch_size)
    index_array = np.arange(num_samples)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
      batch_ids = index_array[batch_start:batch_end]
      inputs_batch = slice_arrays(inputs, batch_ids)

      inputs_batch = [
          ops.convert_to_tensor(val, dtype=backend.floatx())
          for val in inputs_batch]

      if len(inputs_batch) == 1:
        if model._expects_training_arg:
          batch_outs = model.call(inputs_batch[0], training=False)
        else:
          batch_outs = model.call(inputs_batch[0])
      else:
        if model._expects_training_arg:
          batch_outs = model.call(inputs_batch, training=False)
        else:
          batch_outs = model.call(inputs_batch)

      if not isinstance(batch_outs, list):
        batch_outs = [batch_outs]
      if batch_index == 0:
        # Pre-allocate the results arrays.
        for batch_out in batch_outs:
          dims = batch_out.shape[1:].dims
          dims_list = [d.value for d in dims]
          shape = (num_samples,) + tuple(dims_list)
          outs.append(np.zeros(shape, dtype=batch_out.dtype.as_numpy_dtype))
      for i, batch_out in enumerate(batch_outs):
        outs[i][batch_start:batch_end] = batch_out
      if verbose == 1:
        progbar.update(batch_end)
    if len(outs) == 1:
      return outs[0]
    return outs
