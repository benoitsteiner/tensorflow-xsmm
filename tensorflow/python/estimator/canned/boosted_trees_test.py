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
"""Tests boosted_trees estimators and model_fn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.kernels.boosted_trees import boosted_trees_pb2
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator.canned import boosted_trees
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_boosted_trees_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import checkpoint_utils

NUM_FEATURES = 3

BUCKET_BOUNDARIES = [-2., .5, 12.]  # Boundaries for all the features.
INPUT_FEATURES = np.array(
    [
        [12.5, 1.0, -2.001, -2.0001, -1.999],  # feature_0 quantized:[3,2,0,0,1]
        [2.0, -3.0, 0.5, 0.0, 0.4995],         # feature_1 quantized:[2,0,2,1,1]
        [3.0, 20.0, 50.0, -100.0, 102.75],     # feature_2 quantized:[2,3,3,0,3]
    ],
    dtype=np.float32)
CLASSIFICATION_LABELS = [[0.], [1.], [1.], [0.], [0.]]
REGRESSION_LABELS = [[1.5], [0.3], [0.2], [2.], [5.]]
FEATURES_DICT = {'f_%d' % i: INPUT_FEATURES[i] for i in range(NUM_FEATURES)}

# EXAMPLE_ID is not exposed to Estimator yet, but supported at model_fn level.
EXAMPLE_IDS = np.array([0, 1, 2, 3, 4], dtype=np.int64)
EXAMPLE_ID_COLUMN = '__example_id__'


def _make_train_input_fn(is_classification):
  """Makes train input_fn for classification/regression."""

  def _input_fn():
    features = dict(FEATURES_DICT)
    features[EXAMPLE_ID_COLUMN] = constant_op.constant(EXAMPLE_IDS)
    if is_classification:
      labels = CLASSIFICATION_LABELS
    else:
      labels = REGRESSION_LABELS
    return features, labels

  return _input_fn


class BoostedTreesEstimatorTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._feature_columns = {
        feature_column.bucketized_column(
            feature_column.numeric_column('f_%d' % i, dtype=dtypes.float32),
            BUCKET_BOUNDARIES)
        for i in range(NUM_FEATURES)
    }

  def _assert_checkpoint(self, model_dir, global_step, finalized_trees,
                         attempted_layers):
    reader = checkpoint_utils.load_checkpoint(model_dir)
    self.assertEqual(global_step, reader.get_tensor(ops.GraphKeys.GLOBAL_STEP))
    serialized = reader.get_tensor('boosted_trees:0_serialized')
    ensemble_proto = boosted_trees_pb2.TreeEnsemble()
    ensemble_proto.ParseFromString(serialized)
    self.assertEqual(
        finalized_trees,
        sum([1 for t in ensemble_proto.tree_metadata if t.is_finalized]))
    self.assertEqual(attempted_layers,
                     ensemble_proto.growing_metadata.num_layers_attempted)

  def testTrainAndEvaluateBinaryClassifier(self):
    input_fn = _make_train_input_fn(is_classification=True)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)

  def testInferBinaryClassifier(self):
    train_input_fn = _make_train_input_fn(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(train_input_fn, steps=num_steps)

    predictions = list(est.predict(input_fn=predict_input_fn))
    # All labels are correct.
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  def testTrainAndEvaluateRegressor(self):
    input_fn = _make_train_input_fn(is_classification=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        max_depth=5)

    # It will stop after 10 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=10, finalized_trees=2, attempted_layers=10)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 1.008551)

  def testInferRegressor(self):
    train_input_fn = _make_train_input_fn(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(train_input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)

    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])


class ModelFnTests(test_util.TensorFlowTestCase):
  """Tests bt_model_fn including unexposed internal functionalities."""

  def setUp(self):
    self._feature_columns = {
        feature_column.bucketized_column(
            feature_column.numeric_column('f_%d' % i, dtype=dtypes.float32),
            BUCKET_BOUNDARIES) for i in range(NUM_FEATURES)
    }
    self._tree_hparams = boosted_trees._TreeHParams(  # pylint:disable=protected-access
        n_trees=2,
        max_depth=2,
        learning_rate=0.1,
        l1=0.,
        l2=0.01,
        tree_complexity=0.)

  def _get_expected_ensembles_for_classification(self):
    first_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.387675
            }
          }
          nodes {
            leaf {
              scalar: -0.181818
            }
          }
          nodes {
            leaf {
              scalar: 0.0625
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    second_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.387675
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 3
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 0.0
              original_leaf {
                scalar: -0.181818
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.105518
              original_leaf {
                scalar: 0.0625
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.348397
            }
          }
          nodes {
            leaf {
              scalar: -0.181818
            }
          }
          nodes {
            leaf {
              scalar: 0.224091
            }
          }
          nodes {
            leaf {
              scalar: 0.056815
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 0
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
        """
    third_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.387675
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 3
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 0.0
              original_leaf {
                scalar: -0.181818
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.105518
              original_leaf {
                scalar: 0.0625
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.348397
            }
          }
          nodes {
            leaf {
              scalar: -0.181818
            }
          }
          nodes {
            leaf {
              scalar: 0.224091
            }
          }
          nodes {
            leaf {
              scalar: 0.056815
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 0
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.287131
            }
          }
          nodes {
            leaf {
              scalar: 0.162042
            }
          }
          nodes {
            leaf {
              scalar: -0.086986
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 3
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    return (first_round, second_round, third_round)

  def _get_expected_ensembles_for_regression(self):
    first_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.169714
            }
          }
          nodes {
            leaf {
              scalar: 0.241322
            }
          }
          nodes {
            leaf {
              scalar: 0.083951
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    second_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.169714
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 1
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.673407
              original_leaf {
                scalar: 0.241322
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.324102
              original_leaf {
                scalar: 0.083951
              }
            }
          }
          nodes {
            leaf {
              scalar: 0.563167
            }
          }
          nodes {
            leaf {
              scalar: 0.247047
            }
          }
          nodes {
            leaf {
              scalar: 0.095273
            }
          }
          nodes {
            leaf {
              scalar: 0.222102
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 0
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
        """
    third_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.169714
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 1
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.673407
              original_leaf {
                scalar: 0.241322
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.324102
              original_leaf {
                scalar: 0.083951
              }
            }
          }
          nodes {
            leaf {
              scalar: 0.563167
            }
          }
          nodes {
            leaf {
              scalar: 0.247047
            }
          }
          nodes {
            leaf {
              scalar: 0.095273
            }
          }
          nodes {
            leaf {
              scalar: 0.222102
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 0
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.981026
            }
          }
          nodes {
            leaf {
              scalar: 0.005166
            }
          }
          nodes {
            leaf {
              scalar: 0.180281
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 3
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    return (first_round, second_round, third_round)

  def _get_train_op_and_ensemble(self, head, config, is_classification,
                                 train_in_memory):
    """Calls bt_model_fn() and returns the train_op and ensemble_serialzed."""
    features, labels = _make_train_input_fn(is_classification)()
    estimator_spec = boosted_trees._bt_model_fn(  # pylint:disable=protected-access
        features=features,
        labels=labels,
        mode=model_fn.ModeKeys.TRAIN,
        head=head,
        feature_columns=self._feature_columns,
        tree_hparams=self._tree_hparams,
        example_id_column_name=EXAMPLE_ID_COLUMN,
        n_batches_per_layer=1,
        config=config,
        train_in_memory=train_in_memory)
    resources.initialize_resources(resources.shared_resources()).run()
    variables.global_variables_initializer().run()
    variables.local_variables_initializer().run()

    # Gets the train_op and serialized proto of the ensemble.
    shared_resources = resources.shared_resources()
    self.assertEqual(1, len(shared_resources))
    train_op = estimator_spec.train_op
    with ops.control_dependencies([train_op]):
      _, ensemble_serialized = (
          gen_boosted_trees_ops.boosted_trees_serialize_ensemble(
              shared_resources[0].handle))
    return train_op, ensemble_serialized

  def testTrainClassifierInMemory(self):
    ops.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_classification())
    with self.test_session() as sess:
      # Train with train_in_memory mode.
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_classification_head(n_classes=2),
            run_config.RunConfig(),
            is_classification=True,
            train_in_memory=True)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

  def testTrainClassifierNonInMemory(self):
    ops.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_classification())
    with self.test_session() as sess:
      # Train without train_in_memory mode.
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_classification_head(n_classes=2),
            run_config.RunConfig(),
            is_classification=True,
            train_in_memory=False)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

  def testTrainRegressorInMemory(self):
    ops.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_regression())
    with self.test_session() as sess:
      # Train with train_in_memory mode.
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_regression_head(label_dimension=1),
            run_config.RunConfig(),
            is_classification=False,
            train_in_memory=True)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

  def testTrainRegressorNonInMemory(self):
    ops.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_regression())
    with self.test_session() as sess:
      # Train without train_in_memory mode.
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_regression_head(label_dimension=1),
            run_config.RunConfig(),
            is_classification=False,
            train_in_memory=False)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)


if __name__ == '__main__':
  googletest.main()
