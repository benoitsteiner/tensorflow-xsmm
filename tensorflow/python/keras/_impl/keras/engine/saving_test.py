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
#,============================================================================
"""Tests for model saving."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.platform import test
from tensorflow.python.training import training as training_module

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


class TestWeightSavingAndLoading(test.TestCase):

  def test_weight_loading(self):
    with self.test_session():
      a = keras.layers.Input(shape=(2,))
      x = keras.layers.Dense(3)(a)
      b = keras.layers.Dense(1)(x)
      model = keras.models.Model(a, b)

      x = np.random.random((3, 2))
      ref_y = model.predict(x)
      weights = model.get_weights()
      model.set_weights(weights)
      y = model.predict(x)
      self.assertAllClose(ref_y, y)

      with self.assertRaises(ValueError):
        model.set_weights(weights[1:])
      with self.assertRaises(ValueError):
        model.set_weights(weights[::-1])

      if h5py is None:
        return  # Skip rest of test if H5py isn't available.

      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir)

      h5_path = os.path.join(temp_dir, 'test.h5')
      model.save_weights(h5_path)
      model.load_weights(h5_path)
      y = model.predict(x)
      self.assertAllClose(ref_y, y)

      model.load_weights(h5_path, by_name=True)
      y = model.predict(x)
      self.assertAllClose(ref_y, y)

  def test_weight_preprocessing(self):
    input_dim = 3
    output_dim = 3
    size = 2
    cases = [
        [
            (keras.layers.Bidirectional(keras.layers.SimpleRNN(2))),
            [np.random.random((2, 1)), np.random.random((2, 1))],
            (None, 3, 2),
        ],
        [
            (keras.layers.TimeDistributed(keras.layers.Dense(1))),
            [np.random.random((2, 1)), np.random.random((1,))],
            (None, 3, 2),
        ],
        [
            (keras.layers.Conv1D(output_dim, size, use_bias=False)),
            [np.random.random((output_dim, input_dim, size, 1))],
            (None, 4, input_dim),
        ],
        [
            (keras.layers.Conv2D(output_dim, size,
                                 use_bias=False, data_format='channels_first')),
            [np.random.random((output_dim, input_dim, size, size))],
            (None, input_dim, 4, 4),
        ],
        [
            (keras.layers.Conv2DTranspose(output_dim, size,
                                          use_bias=False,
                                          data_format='channels_first')),
            [np.random.random((output_dim, input_dim, size, size))],
            (None, input_dim, 4, 4),
        ],
        [
            (keras.layers.Conv2DTranspose(output_dim, size,
                                          use_bias=False,
                                          data_format='channels_last')),
            [np.random.random((size, size, input_dim, output_dim))],
            (None, 4, 4, input_dim),
        ],
        [
            (keras.layers.Conv3D(output_dim, size,
                                 use_bias=False, data_format='channels_first')),
            [np.random.random((output_dim, input_dim, size, size, size))],
            (None, input_dim, 4, 4, 4),
        ],
        [
            (keras.layers.GRU(output_dim)),
            [np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,))],
            (None, 4, input_dim),
        ],
        [
            (keras.layers.LSTM(output_dim)),
            [np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,))],
            (None, 4, input_dim),
        ],
    ]
    for layer, weights, input_shape in cases:
      layer.build(input_shape)
      _ = keras.engine.saving.preprocess_weights_for_loading(
          layer, weights, original_keras_version='1')

    model = keras.models.Sequential([keras.layers.Dense(2, input_dim=2)])
    _ = keras.engine.saving.preprocess_weights_for_loading(
        model, model.weights, original_keras_version='1')

    x = keras.Input((2,))
    y = keras.layers.Dense(2)(x)
    model = keras.models.Model(x, y)
    _ = keras.engine.saving.preprocess_weights_for_loading(
        model, model.weights, original_keras_version='1')

  def test_sequential_weight_loading(self):
    if h5py is None:
      return

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    h5_path = os.path.join(temp_dir, 'test.h5')

    num_hidden = 5
    input_dim = 3
    batch_size = 5
    num_classes = 2

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(num_hidden, input_dim=input_dim))
      model.add(keras.layers.Dense(num_classes))

      x = np.random.random((batch_size, input_dim))
      ref_y = model.predict(x)

      model.save_weights(h5_path)

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(num_hidden, input_dim=input_dim))
      model.add(keras.layers.Dense(num_classes))
      model.load_weights(h5_path)
      y = model.predict(x)

      self.assertAllClose(y, ref_y)


class TestWholeModelSaving(test.TestCase):

  def test_sequential_model_saving(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.RepeatVector(3))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
      model.compile(loss=keras.losses.MSE,
                    optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    metrics=[keras.metrics.categorical_accuracy],
                    sample_weight_mode='temporal')
      x = np.random.random((1, 3))
      y = np.random.random((1, 3, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      new_model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # test that new updates are the same with both models
      x = np.random.random((1, 3))
      y = np.random.random((1, 3, 3))
      model.train_on_batch(x, y)
      new_model.train_on_batch(x, y)
      out = model.predict(x)
      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_sequential_model_saving_2(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      # test with custom optimizer, loss

      class CustomOp(keras.optimizers.RMSprop):
        pass

      def custom_loss(y_true, y_pred):
        return keras.losses.mse(y_true, y_pred)

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss=custom_loss, optimizer=CustomOp(), metrics=['acc'])

      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      model = keras.models.load_model(
          fname,
          custom_objects={'CustomOp': CustomOp,
                          'custom_loss': custom_loss})
      os.close(fd)
      os.remove(fname)

      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_functional_model_saving(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      inputs = keras.layers.Input(shape=(3,))
      x = keras.layers.Dense(2)(inputs)
      output = keras.layers.Dense(3)(x)

      model = keras.models.Model(inputs, output)
      model.compile(loss=keras.losses.MSE,
                    optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    metrics=[keras.metrics.categorical_accuracy])
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_saving_without_compilation(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

  def test_saving_with_tf_optimizer(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse',
                    optimizer=training_module.AdadeltaOptimizer(0.1),
                    metrics=['acc'])

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

  def test_saving_right_after_compilation(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
      model._make_train_function()

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

  def test_saving_lambda_numpy_array_arguments(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    mean = np.random.random((4, 2, 3))
    std = np.abs(np.random.random((4, 2, 3))) + 1e-5
    inputs = keras.layers.Input(shape=(4, 2, 3))
    output = keras.layers.Lambda(lambda image, mu, std: (image - mu) / std,
                                 arguments={'mu': mean, 'std': std})(inputs)
    model = keras.models.Model(inputs, output)
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

    fd, fname = tempfile.mkstemp('.h5')
    keras.models.save_model(model, fname)

    model = keras.models.load_model(fname)
    os.close(fd)
    os.remove(fname)

    self.assertAllClose(mean, model.layers[1].arguments['mu'])
    self.assertAllClose(std, model.layers[1].arguments['std'])

  def test_saving_model_with_long_layer_names(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      # This layer name will make the `layers_name` HDF5 attribute blow
      # out of proportion. Note that it fits into the internal HDF5
      # attribute memory limit on its own but because h5py converts
      # the list of layer names into numpy array, which uses the same
      # amout of memory for every item, it increases the memory
      # requirements substantially.
      x = keras.Input(shape=(2,), name='input_' + ('x' * (2**15)))
      f = x
      for i in range(4):
        f = keras.layers.Dense(2, name='dense_%d' % (i,))(f)
      model = keras.Model(inputs=[x], outputs=[f])
      model.compile(loss='mse', optimizer='adam', metrics=['acc'])

      x = np.random.random((1, 2))
      y = np.random.random((1, 2))
      model.train_on_batch(x, y)
      out = model.predict(x)

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)

      # Check that the HDF5 files contains chunked array
      # of layer names.
      with h5py.File(fname, 'r') as h5file:
        num_names_arrays = len([attr for attr in h5file['model_weights'].attrs
                                if attr.startswith('layer_names')])
      # The chunking of layer names array should have happend.
      self.assertGreater(num_names_arrays, 0)
      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # Cleanup
      os.close(fd)
      os.remove(fname)

  def test_saving_model_with_long_weights_names(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      x = keras.Input(shape=(2,), name='nested_model_input')
      f = x
      for i in range(4):
        f = keras.layers.Dense(2, name='nested_model_dense_%d' % (i,))(f)
      # This layer name will make the `weights_name`
      # HDF5 attribute blow out of proportion.
      f = keras.layers.Dense(2, name='nested_model_output' + ('x' * (2**15)))(f)
      nested_model = keras.Model(inputs=[x], outputs=[f], name='nested_model')

      x = keras.Input(shape=(2,), name='outer_model_input')
      f = nested_model(x)
      f = keras.layers.Dense(2, name='outer_model_output')(f)

      model = keras.Model(inputs=[x], outputs=[f])
      model.compile(loss='mse', optimizer='adam', metrics=['acc'])

      x = np.random.random((1, 2))
      y = np.random.random((1, 2))
      model.train_on_batch(x, y)
      out = model.predict(x)

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)

      # Check that the HDF5 files contains chunked array
      # of weight names.
      with h5py.File(fname, 'r') as h5file:
        num_weight_arrays = len(
            [attr for attr in h5file['model_weights']['nested_model'].attrs
             if attr.startswith('weight_names')])
      # The chunking of layer names array should have happend.
      self.assertGreater(num_weight_arrays, 0)
      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # Cleanup
      os.close(fd)
      os.remove(fname)


if __name__ == '__main__':
  test.main()
