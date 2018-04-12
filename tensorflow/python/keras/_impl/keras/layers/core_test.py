# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras core layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras._impl import keras
from tensorflow.python.keras._impl.keras import testing_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class CoreLayersTest(test.TestCase):

  def test_masking(self):
    with self.test_session():
      testing_utils.layer_test(
          keras.layers.Masking, kwargs={}, input_shape=(3, 2, 3))

  def test_dropout(self):
    with self.test_session():
      testing_utils.layer_test(
          keras.layers.Dropout, kwargs={'rate': 0.5}, input_shape=(3, 2))

    with self.test_session():
      testing_utils.layer_test(
          keras.layers.Dropout,
          kwargs={'rate': 0.5,
                  'noise_shape': [3, 1]},
          input_shape=(3, 2))

    # https://github.com/tensorflow/tensorflow/issues/14819
    with self.test_session():
      dropout = keras.layers.Dropout(0.5)
      self.assertEqual(True, dropout.supports_masking)

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_spatial_dropout(self):
    testing_utils.layer_test(
        keras.layers.SpatialDropout1D,
        kwargs={'rate': 0.5},
        input_shape=(2, 3, 4))

    testing_utils.layer_test(
        keras.layers.SpatialDropout2D,
        kwargs={'rate': 0.5},
        input_shape=(2, 3, 4, 5))

    testing_utils.layer_test(
        keras.layers.SpatialDropout2D,
        kwargs={'rate': 0.5, 'data_format': 'channels_first'},
        input_shape=(2, 3, 4, 5))

    testing_utils.layer_test(
        keras.layers.SpatialDropout3D,
        kwargs={'rate': 0.5},
        input_shape=(2, 3, 4, 4, 5))

    testing_utils.layer_test(
        keras.layers.SpatialDropout3D,
        kwargs={'rate': 0.5, 'data_format': 'channels_first'},
        input_shape=(2, 3, 4, 4, 5))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_activation(self):
    # with string argument
    testing_utils.layer_test(
        keras.layers.Activation,
        kwargs={'activation': 'relu'},
        input_shape=(3, 2))

    # with function argument
    testing_utils.layer_test(
        keras.layers.Activation,
        kwargs={'activation': keras.backend.relu},
        input_shape=(3, 2))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_reshape(self):
    testing_utils.layer_test(
        keras.layers.Reshape,
        kwargs={'target_shape': (8, 1)},
        input_shape=(3, 2, 4))

    testing_utils.layer_test(
        keras.layers.Reshape,
        kwargs={'target_shape': (-1, 1)},
        input_shape=(3, 2, 4))

    testing_utils.layer_test(
        keras.layers.Reshape,
        kwargs={'target_shape': (1, -1)},
        input_shape=(3, 2, 4))

    testing_utils.layer_test(
        keras.layers.Reshape,
        kwargs={'target_shape': (-1, 1)},
        input_shape=(None, None, 2))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_permute(self):
    testing_utils.layer_test(
        keras.layers.Permute, kwargs={'dims': (2, 1)}, input_shape=(3, 2, 4))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_flatten(self):
    testing_utils.layer_test(
        keras.layers.Flatten, kwargs={}, input_shape=(3, 2, 4))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_repeat_vector(self):
    testing_utils.layer_test(
        keras.layers.RepeatVector, kwargs={'n': 3}, input_shape=(3, 2))

  def test_lambda(self):
    testing_utils.layer_test(
        keras.layers.Lambda,
        kwargs={'function': lambda x: x + 1},
        input_shape=(3, 2))

    testing_utils.layer_test(
        keras.layers.Lambda,
        kwargs={
            'function': lambda x, a, b: x * a + b,
            'arguments': {
                'a': 0.6,
                'b': 0.4
            }
        },
        input_shape=(3, 2))

    # test serialization with function
    def f(x):
      return x + 1

    ld = keras.layers.Lambda(f)
    config = ld.get_config()
    ld = keras.layers.deserialize({
        'class_name': 'Lambda',
        'config': config
    })

    # test with lambda
    ld = keras.layers.Lambda(
        lambda x: keras.backend.concatenate([math_ops.square(x), x]))
    config = ld.get_config()
    ld = keras.layers.Lambda.from_config(config)

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_dense(self):
    testing_utils.layer_test(
        keras.layers.Dense, kwargs={'units': 3}, input_shape=(3, 2))

    testing_utils.layer_test(
        keras.layers.Dense, kwargs={'units': 3}, input_shape=(3, 4, 2))

    testing_utils.layer_test(
        keras.layers.Dense, kwargs={'units': 3}, input_shape=(None, None, 2))

    testing_utils.layer_test(
        keras.layers.Dense, kwargs={'units': 3}, input_shape=(3, 4, 5, 2))

  def test_dense_regularization(self):
    with self.test_session():
      layer = keras.layers.Dense(
          3,
          kernel_regularizer=keras.regularizers.l1(0.01),
          bias_regularizer='l1',
          activity_regularizer='l2',
          name='dense_reg')
      layer(keras.backend.variable(np.ones((2, 4))))
      self.assertEqual(3, len(layer.losses))

  def test_dense_constraints(self):
    with self.test_session():
      k_constraint = keras.constraints.max_norm(0.01)
      b_constraint = keras.constraints.max_norm(0.01)
      layer = keras.layers.Dense(
          3, kernel_constraint=k_constraint, bias_constraint=b_constraint)
      layer(keras.backend.variable(np.ones((2, 4))))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_activity_regularization(self):
    with self.test_session():
      layer = keras.layers.ActivityRegularization(l1=0.1)
      layer(keras.backend.variable(np.ones((2, 4))))
      self.assertEqual(1, len(layer.losses))
      _ = layer.get_config()

  def test_lambda_output_shape(self):
    with self.test_session():
      l = keras.layers.Lambda(lambda x: x + 1, output_shape=(1, 1))
      l(keras.backend.variable(np.ones((1, 1))))
      self.assertEqual((1, 1), l.get_config()['output_shape'])

  def test_lambda_output_shape_function(self):
    def get_output_shape(input_shape):
      return 1 * input_shape

    with self.test_session():
      l = keras.layers.Lambda(lambda x: x + 1, output_shape=get_output_shape)
      l(keras.backend.variable(np.ones((1, 1))))
      self.assertEqual('lambda', l.get_config()['output_shape_type'])

  def test_lambda_config_serialization(self):
    with self.test_session():
      # test serialization with output_shape and output_shape_type
      layer = keras.layers.Lambda(lambda x: x + 1, output_shape=(1, 1))
      layer(keras.backend.variable(np.ones((1, 1))))
      config = layer.get_config()
      layer = keras.layers.deserialize({
          'class_name': 'Lambda',
          'config': config
      })

      layer = keras.layers.Lambda.from_config(config)

if __name__ == '__main__':
  test.main()
