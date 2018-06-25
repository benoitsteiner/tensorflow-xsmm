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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.contrib.data.python.kernel_tests import reader_dataset_ops_test_base
from tensorflow.contrib.data.python.ops import readers
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class ReadBatchFeaturesTest(
    reader_dataset_ops_test_base.ReadBatchFeaturesTestBase):

  def testRead(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 10]:
        with ops.Graph().as_default() as g:
          with self.test_session(graph=g) as sess:
            # Basic test: read from file 0.
            self.outputs = self.make_batch_feature(
                filenames=self.test_filenames[0],
                num_epochs=num_epochs,
                batch_size=batch_size).make_one_shot_iterator().get_next()
            self.verify_records(sess, batch_size, 0, num_epochs=num_epochs)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess)

        with ops.Graph().as_default() as g:
          with self.test_session(graph=g) as sess:
            # Basic test: read from file 1.
            self.outputs = self.make_batch_feature(
                filenames=self.test_filenames[1],
                num_epochs=num_epochs,
                batch_size=batch_size).make_one_shot_iterator().get_next()
            self.verify_records(sess, batch_size, 1, num_epochs=num_epochs)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess)

        with ops.Graph().as_default() as g:
          with self.test_session(graph=g) as sess:
            # Basic test: read from both files.
            self.outputs = self.make_batch_feature(
                filenames=self.test_filenames,
                num_epochs=num_epochs,
                batch_size=batch_size).make_one_shot_iterator().get_next()
            self.verify_records(sess, batch_size, num_epochs=num_epochs)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess)

  def testReadWithEquivalentDataset(self):
    features = {
        "file": parsing_ops.FixedLenFeature([], dtypes.int64),
        "record": parsing_ops.FixedLenFeature([], dtypes.int64),
    }
    dataset = (
        core_readers.TFRecordDataset(self.test_filenames)
        .map(lambda x: parsing_ops.parse_single_example(x, features))
        .repeat(10).batch(2))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for file_batch, _, _, _, record_batch in self._next_expected_batch(
          range(self._num_files), 2, 10):
        actual_batch = sess.run(next_element)
        self.assertAllEqual(file_batch, actual_batch["file"])
        self.assertAllEqual(record_batch, actual_batch["record"])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testReadWithFusedShuffleRepeatDataset(self):
    num_epochs = 5
    total_records = num_epochs * self._num_records
    for batch_size in [1, 2]:
      # Test that shuffling with same seed produces the same result.
      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
          outputs1 = self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5).make_one_shot_iterator().get_next()
          outputs2 = self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5).make_one_shot_iterator().get_next()
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              self.assertAllEqual(batch1[i], batch2[i])

      # Test that shuffling with different seeds produces a different order.
      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
          outputs1 = self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5).make_one_shot_iterator().get_next()
          outputs2 = self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=15).make_one_shot_iterator().get_next()
          all_equal = True
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              all_equal = all_equal and np.array_equal(batch1[i], batch2[i])
          self.assertFalse(all_equal)

  def testParallelReadersAndParsers(self):
    num_epochs = 5
    for batch_size in [1, 2]:
      for reader_num_threads in [2, 4]:
        for parser_num_threads in [2, 4]:
          with ops.Graph().as_default() as g:
            with self.test_session(graph=g) as sess:
              self.outputs = self.make_batch_feature(
                  filenames=self.test_filenames,
                  num_epochs=num_epochs,
                  batch_size=batch_size,
                  reader_num_threads=reader_num_threads,
                  parser_num_threads=parser_num_threads).make_one_shot_iterator(
                  ).get_next()
              self.verify_records(
                  sess,
                  batch_size,
                  num_epochs=num_epochs,
                  interleave_cycle_length=reader_num_threads)
              with self.assertRaises(errors.OutOfRangeError):
                self._next_actual_batch(sess)

  def testDropFinalBatch(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 10]:
        with ops.Graph().as_default():
          # Basic test: read from file 0.
          self.outputs = self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              drop_final_batch=True).make_one_shot_iterator().get_next()
          for _, tensor in self.outputs.items():
            if isinstance(tensor, ops.Tensor):  # Guard against SparseTensor.
              self.assertEqual(tensor.shape[0], batch_size)


class MakeCsvDatasetTest(test.TestCase):

  COLUMN_TYPES = [
      dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64, dtypes.string
  ]
  COLUMNS = ["col%d" % i for i in range(len(COLUMN_TYPES))]
  DEFAULT_VALS = [[], [], [], [], ["NULL"]]
  DEFAULTS = [
      constant_op.constant([], dtype=dtypes.int32),
      constant_op.constant([], dtype=dtypes.int64),
      constant_op.constant([], dtype=dtypes.float32),
      constant_op.constant([], dtype=dtypes.float64),
      constant_op.constant(["NULL"], dtype=dtypes.string)
  ]
  LABEL = COLUMNS[0]

  def setUp(self):
    super(MakeCsvDatasetTest, self).setUp()
    self._num_files = 2
    self._num_records = 11
    self._test_filenames = self._create_files()

  def _csv_values(self, fileno, recordno):
    return [
        fileno,
        recordno,
        fileno * recordno * 0.5,
        fileno * recordno + 0.5,
        "record %d" % recordno if recordno % 2 == 1 else "",
    ]

  def _write_file(self, filename, rows):
    for i in range(len(rows)):
      if isinstance(rows[i], list):
        rows[i] = ",".join(str(v) if v is not None else "" for v in rows[i])
    fn = os.path.join(self.get_temp_dir(), filename)
    f = open(fn, "w")
    f.write("\n".join(rows))
    f.close()
    return fn

  def _create_file(self, fileno, header=True):
    rows = []
    if header:
      rows.append(self.COLUMNS)
    for recno in range(self._num_records):
      rows.append(self._csv_values(fileno, recno))
    return self._write_file("csv_file%d.csv" % fileno, rows)

  def _create_files(self):
    filenames = []
    for i in range(self._num_files):
      filenames.append(self._create_file(i))
    return filenames

  def _make_csv_dataset(
      self,
      filenames,
      defaults,
      column_names=COLUMNS,
      label_name=LABEL,
      select_cols=None,
      batch_size=1,
      num_epochs=1,
      shuffle=False,
      shuffle_seed=None,
      header=True,
      na_value="",
  ):
    return readers.make_csv_dataset(
        filenames,
        batch_size=batch_size,
        column_names=column_names,
        column_defaults=defaults,
        label_name=label_name,
        num_epochs=num_epochs,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        header=header,
        na_value=na_value,
        select_columns=select_cols,
    )

  def _next_actual_batch(self, file_indices, batch_size, num_epochs, defaults):
    features = {col: list() for col in self.COLUMNS}
    for _ in range(num_epochs):
      for i in file_indices:
        for j in range(self._num_records):
          values = self._csv_values(i, j)
          for n, v in enumerate(values):
            if v == "":  # pylint: disable=g-explicit-bool-comparison
              values[n] = defaults[n][0]
          values[-1] = values[-1].encode("utf-8")

          # Regroup lists by column instead of row
          for n, col in enumerate(self.COLUMNS):
            features[col].append(values[n])
          if len(list(features.values())[0]) == batch_size:
            yield features
            features = {col: list() for col in self.COLUMNS}

  def _run_actual_batch(self, outputs, sess):
    features, labels = sess.run(outputs)
    batch = [features[k] for k in self.COLUMNS if k != self.LABEL]
    batch.append(labels)
    return batch

  def _verify_records(
      self,
      sess,
      dataset,
      file_indices,
      defaults=tuple(DEFAULT_VALS),
      label_name=LABEL,
      batch_size=1,
      num_epochs=1,
  ):
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    for expected_features in self._next_actual_batch(file_indices, batch_size,
                                                     num_epochs, defaults):
      actual_features = sess.run(get_next)

      if label_name is not None:
        expected_labels = expected_features.pop(label_name)
        # Compare labels
        self.assertAllEqual(expected_labels, actual_features[1])
        actual_features = actual_features[0]  # Extract features dict from tuple

      for k in expected_features.keys():
        # Compare features
        self.assertAllEqual(expected_features[k], actual_features[k])

    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  def testMakeCSVDataset(self):
    defaults = self.DEFAULTS

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Basic test: read from file 0.
        dataset = self._make_csv_dataset(self._test_filenames[0], defaults)
        self._verify_records(sess, dataset, [0])
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Basic test: read from file 1.
        dataset = self._make_csv_dataset(self._test_filenames[1], defaults)
        self._verify_records(sess, dataset, [1])
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Read from both files.
        dataset = self._make_csv_dataset(self._test_filenames, defaults)
        self._verify_records(sess, dataset, range(self._num_files))
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Read from both files. Exercise the `batch` and `num_epochs` parameters
        # of make_csv_dataset and make sure they work.
        dataset = self._make_csv_dataset(
            self._test_filenames, defaults, batch_size=2, num_epochs=10)
        self._verify_records(
            sess, dataset, range(self._num_files), batch_size=2, num_epochs=10)

  def testMakeCSVDataset_withBadColumns(self):
    """Tests that exception is raised when input is malformed.
    """
    dupe_columns = self.COLUMNS[:-1] + self.COLUMNS[:1]
    defaults = self.DEFAULTS

    # Duplicate column names
    with self.assertRaises(ValueError):
      self._make_csv_dataset(
          self._test_filenames, defaults, column_names=dupe_columns)

    # Label key not one of column names
    with self.assertRaises(ValueError):
      self._make_csv_dataset(
          self._test_filenames, defaults, label_name="not_a_real_label")

  def testMakeCSVDataset_withNoLabel(self):
    """Tests that CSV datasets can be created when no label is specified.
    """
    defaults = self.DEFAULTS
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Read from both files. Make sure this works with no label key supplied.
        dataset = self._make_csv_dataset(
            self._test_filenames,
            defaults,
            batch_size=2,
            num_epochs=10,
            label_name=None)
        self._verify_records(
            sess,
            dataset,
            range(self._num_files),
            batch_size=2,
            num_epochs=10,
            label_name=None)

  def testMakeCSVDataset_withNoHeader(self):
    """Tests that datasets can be created from CSV files with no header line.
    """
    defaults = self.DEFAULTS
    file_without_header = self._create_file(
        len(self._test_filenames), header=False)
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            file_without_header,
            defaults,
            batch_size=2,
            num_epochs=10,
            header=False,
        )
        self._verify_records(
            sess,
            dataset,
            [len(self._test_filenames)],
            batch_size=2,
            num_epochs=10,
        )

  def testMakeCSVDataset_withTypes(self):
    """Tests that defaults can be a dtype instead of a Tensor for required vals.
    """
    defaults = [d for d in self.COLUMN_TYPES[:-1]]
    defaults.append(constant_op.constant(["NULL"], dtype=dtypes.string))
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(self._test_filenames, defaults)
        self._verify_records(sess, dataset, range(self._num_files))

  def testMakeCSVDataset_withNoColNames(self):
    """Tests that datasets can be created when column names are not specified.

    In that case, we should infer the column names from the header lines.
    """
    defaults = self.DEFAULTS
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        # Read from both files. Exercise the `batch` and `num_epochs` parameters
        # of make_csv_dataset and make sure they work.
        dataset = self._make_csv_dataset(
            self._test_filenames,
            defaults,
            column_names=None,
            batch_size=2,
            num_epochs=10)
        self._verify_records(
            sess, dataset, range(self._num_files), batch_size=2, num_epochs=10)

  def testMakeCSVDataset_withTypeInferenceMismatch(self):
    # Test that error is thrown when num fields doesn't match columns
    with self.assertRaises(ValueError):
      self._make_csv_dataset(
          self._test_filenames,
          column_names=self.COLUMNS + ["extra_name"],
          defaults=None,
          batch_size=2,
          num_epochs=10)

  def testMakeCSVDataset_withTypeInference(self):
    """Tests that datasets can be created when no defaults are specified.

    In that case, we should infer the types from the first N records.
    """
    # Test that it works with standard test files (with header, etc)
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            self._test_filenames, defaults=None, batch_size=2, num_epochs=10)
        self._verify_records(
            sess,
            dataset,
            range(self._num_files),
            batch_size=2,
            num_epochs=10,
            defaults=[[], [], [], [], [""]])

  def testMakeCSVDataset_withTypeInferenceTricky(self):
    # Test on a deliberately tricky file (type changes as we read more rows, and
    # there are null values)
    fn = os.path.join(self.get_temp_dir(), "file.csv")
    expected_dtypes = [
        dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float32,
        dtypes.string, dtypes.string
    ]
    col_names = ["col%d" % i for i in range(len(expected_dtypes))]
    rows = [[None, None, None, "NAN", "",
             "a"], [1, 2**31 + 1, 2**64, 123, "NAN", ""],
            ['"123"', 2, 2**64, 123.4, "NAN", '"cd,efg"']]
    expected = [[0, 0, 0, 0, "", "a"], [1, 2**31 + 1, 2**64, 123, "", ""],
                [123, 2, 2**64, 123.4, "", "cd,efg"]]
    for row in expected:
      row[-1] = row[-1].encode("utf-8")  # py3 expects byte strings
      row[-2] = row[-2].encode("utf-8")  # py3 expects byte strings
    self._write_file("file.csv", [col_names] + rows)

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            fn,
            defaults=None,
            column_names=None,
            label_name=None,
            na_value="NAN",
        )
        features = dataset.make_one_shot_iterator().get_next()
        # Check that types match
        for i in range(len(expected_dtypes)):
          print(features["col%d" % i].dtype, expected_dtypes[i])
          assert features["col%d" % i].dtype == expected_dtypes[i]
        for i in range(len(rows)):
          assert sess.run(features) == dict(zip(col_names, expected[i]))

  def testMakeCSVDataset_withTypeInferenceAllTypes(self):
    # Test that we make the correct inference for all types with fallthrough
    fn = os.path.join(self.get_temp_dir(), "file.csv")
    expected_dtypes = [
        dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64,
        dtypes.string, dtypes.string
    ]
    col_names = ["col%d" % i for i in range(len(expected_dtypes))]
    rows = [[1, 2**31 + 1, 1.0, 4e40, "abc", ""]]
    expected = [[
        1, 2**31 + 1, 1.0, 4e40, "abc".encode("utf-8"), "".encode("utf-8")
    ]]
    self._write_file("file.csv", [col_names] + rows)

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            fn,
            defaults=None,
            column_names=None,
            label_name=None,
            na_value="NAN",
        )
        features = dataset.make_one_shot_iterator().get_next()
        # Check that types match
        for i in range(len(expected_dtypes)):
          self.assertAllEqual(features["col%d" % i].dtype, expected_dtypes[i])
        for i in range(len(rows)):
          self.assertAllEqual(
              sess.run(features), dict(zip(col_names, expected[i])))

  def testMakeCSVDataset_withSelectColsError(self):
    data = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    col_names = ["col%d" % i for i in range(5)]
    fn = self._write_file("file.csv", [col_names] + data)
    with self.assertRaises(ValueError):
      # Mismatch in number of defaults and number of columns selected,
      # should raise an error
      self._make_csv_dataset(
          fn,
          defaults=[[0]] * 5,
          column_names=col_names,
          label_name=None,
          select_cols=[1, 3])
    with self.assertRaises(ValueError):
      # Invalid column name should raise an error
      self._make_csv_dataset(
          fn,
          defaults=[[0]],
          column_names=col_names,
          label_name=None,
          select_cols=["invalid_col_name"])

  def testMakeCSVDataset_withSelectCols(self):
    data = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    col_names = ["col%d" % i for i in range(5)]
    fn = self._write_file("file.csv", [col_names] + data)
    # If select_cols is specified, should only yield a subset of columns
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            fn,
            defaults=[[0], [0]],
            column_names=col_names,
            label_name=None,
            select_cols=[1, 3])
        expected = [[1, 3], [6, 8]]
        features = dataset.make_one_shot_iterator().get_next()
        for i in range(len(data)):
          self.assertAllEqual(
              sess.run(features),
              dict(zip([col_names[1], col_names[3]], expected[i])))
    # Can still do default inference with select_cols
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            fn,
            defaults=None,
            column_names=col_names,
            label_name=None,
            select_cols=[1, 3])
        expected = [[1, 3], [6, 8]]
        features = dataset.make_one_shot_iterator().get_next()
        for i in range(len(data)):
          self.assertAllEqual(
              sess.run(features),
              dict(zip([col_names[1], col_names[3]], expected[i])))
    # Can still do column name inference
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            fn,
            defaults=None,
            column_names=None,
            label_name=None,
            select_cols=[1, 3])
        expected = [[1, 3], [6, 8]]
        features = dataset.make_one_shot_iterator().get_next()
        for i in range(len(data)):
          self.assertAllEqual(
              sess.run(features),
              dict(zip([col_names[1], col_names[3]], expected[i])))
    # Can specify column names instead of indices
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            fn,
            defaults=None,
            column_names=None,
            label_name=None,
            select_cols=[col_names[1], col_names[3]])
        expected = [[1, 3], [6, 8]]
        features = dataset.make_one_shot_iterator().get_next()
        for i in range(len(data)):
          self.assertAllEqual(
              sess.run(features),
              dict(zip([col_names[1], col_names[3]], expected[i])))

  def testMakeCSVDataset_withShuffle(self):
    total_records = self._num_files * self._num_records
    defaults = self.DEFAULTS
    for batch_size in [1, 2]:
      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
          # Test that shuffling with the same seed produces the same result
          dataset1 = self._make_csv_dataset(
              self._test_filenames,
              defaults,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)
          dataset2 = self._make_csv_dataset(
              self._test_filenames,
              defaults,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)
          outputs1 = dataset1.make_one_shot_iterator().get_next()
          outputs2 = dataset2.make_one_shot_iterator().get_next()
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              self.assertAllEqual(batch1[i], batch2[i])

      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
          # Test that shuffling with a different seed produces different results
          dataset1 = self._make_csv_dataset(
              self._test_filenames,
              defaults,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)
          dataset2 = self._make_csv_dataset(
              self._test_filenames,
              defaults,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=6)
          outputs1 = dataset1.make_one_shot_iterator().get_next()
          outputs2 = dataset2.make_one_shot_iterator().get_next()
          all_equal = False
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              all_equal = all_equal and np.array_equal(batch1[i], batch2[i])
          self.assertFalse(all_equal)


class MakeTFRecordDatasetTest(
    reader_dataset_ops_test_base.TFRecordDatasetTestBase):

  def _interleave(self, iterators, cycle_length):
    pending_iterators = iterators
    open_iterators = []
    num_open = 0
    for i in range(cycle_length):
      if pending_iterators:
        open_iterators.append(pending_iterators.pop(0))
        num_open += 1

    while num_open:
      for i in range(min(cycle_length, len(open_iterators))):
        if open_iterators[i] is None:
          continue
        try:
          yield next(open_iterators[i])
        except StopIteration:
          if pending_iterators:
            open_iterators[i] = pending_iterators.pop(0)
          else:
            open_iterators[i] = None
            num_open -= 1

  def _next_expected_batch(self,
                           file_indices,
                           batch_size,
                           num_epochs,
                           cycle_length,
                           drop_final_batch,
                           use_parser_fn):

    def _next_record(file_indices):
      for j in file_indices:
        for i in range(self._num_records):
          yield j, i

    def _next_record_interleaved(file_indices, cycle_length):
      return self._interleave([_next_record([i]) for i in file_indices],
                              cycle_length)

    record_batch = []
    batch_index = 0
    for _ in range(num_epochs):
      if cycle_length == 1:
        next_records = _next_record(file_indices)
      else:
        next_records = _next_record_interleaved(file_indices, cycle_length)
      for f, r in next_records:
        record = self._record(f, r)
        if use_parser_fn:
          record = record[1:]
        record_batch.append(record)
        batch_index += 1
        if len(record_batch) == batch_size:
          yield record_batch
          record_batch = []
          batch_index = 0
    if record_batch and not drop_final_batch:
      yield record_batch

  def _verify_records(self,
                      sess,
                      outputs,
                      batch_size,
                      file_index,
                      num_epochs,
                      interleave_cycle_length,
                      drop_final_batch,
                      use_parser_fn):
    if file_index is not None:
      file_indices = [file_index]
    else:
      file_indices = range(self._num_files)

    for expected_batch in self._next_expected_batch(
        file_indices, batch_size, num_epochs, interleave_cycle_length,
        drop_final_batch, use_parser_fn):
      actual_batch = sess.run(outputs)
      self.assertAllEqual(expected_batch, actual_batch)

  def _read_test(self, batch_size, num_epochs, file_index=None,
                 num_parallel_reads=1, drop_final_batch=False, parser_fn=False):
    if file_index is None:
      file_pattern = self.test_filenames
    else:
      file_pattern = self.test_filenames[file_index]

    if parser_fn:
      fn = lambda x: string_ops.substr(x, 1, 999)
    else:
      fn = None

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        outputs = readers.make_tf_record_dataset(
            file_pattern=file_pattern,
            num_epochs=num_epochs,
            batch_size=batch_size,
            parser_fn=fn,
            num_parallel_reads=num_parallel_reads,
            drop_final_batch=drop_final_batch,
            shuffle=False).make_one_shot_iterator().get_next()
        self._verify_records(
            sess, outputs, batch_size, file_index, num_epochs=num_epochs,
            interleave_cycle_length=num_parallel_reads,
            drop_final_batch=drop_final_batch, use_parser_fn=parser_fn)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(outputs)

  def testRead(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 3]:
        # Basic test: read from file 0.
        self._read_test(batch_size, num_epochs, 0)

        # Basic test: read from file 1.
        self._read_test(batch_size, num_epochs, 1)

        # Basic test: read from both files.
        self._read_test(batch_size, num_epochs)

        # Basic test: read from both files, with parallel reads.
        self._read_test(batch_size, num_epochs, num_parallel_reads=8)

  def testDropFinalBatch(self):
    for batch_size in [1, 2, 10]:
      for num_epochs in [1, 3]:
        # Read from file 0.
        self._read_test(batch_size, num_epochs, 0, drop_final_batch=True)

        # Read from both files.
        self._read_test(batch_size, num_epochs, drop_final_batch=True)

        # Read from both files, with parallel reads.
        self._read_test(batch_size, num_epochs, num_parallel_reads=8,
                        drop_final_batch=True)

  def testParserFn(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 3]:
        for drop_final_batch in [False, True]:
          self._read_test(batch_size, num_epochs, parser_fn=True,
                          drop_final_batch=drop_final_batch)
          self._read_test(batch_size, num_epochs, num_parallel_reads=8,
                          parser_fn=True, drop_final_batch=drop_final_batch)

  def _shuffle_test(self, batch_size, num_epochs, num_parallel_reads=1,
                    seed=None):
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = readers.make_tf_record_dataset(
            file_pattern=self.test_filenames,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_parallel_reads=num_parallel_reads,
            shuffle=True,
            shuffle_seed=seed)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        sess.run(iterator.initializer)
        first_batches = []
        try:
          while True:
            first_batches.append(sess.run(next_element))
        except errors.OutOfRangeError:
          pass

        sess.run(iterator.initializer)
        second_batches = []
        try:
          while True:
            second_batches.append(sess.run(next_element))
        except errors.OutOfRangeError:
          pass

        self.assertEqual(len(first_batches), len(second_batches))
        if seed is not None:
          # if you set a seed, should get the same results
          for i in range(len(first_batches)):
            self.assertAllEqual(first_batches[i], second_batches[i])

        expected = []
        for f in range(self._num_files):
          for r in range(self._num_records):
            expected.extend([self._record(f, r)] * num_epochs)

        for batches in (first_batches, second_batches):
          actual = []
          for b in batches:
            actual.extend(b)
          self.assertAllEqual(sorted(expected), sorted(actual))

  def testShuffle(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 3]:
        for num_parallel_reads in [1, 2]:
          # Test that all expected elements are produced
          self._shuffle_test(batch_size, num_epochs, num_parallel_reads)
          # Test that elements are produced in a consistent order if
          # you specify a seed.
          self._shuffle_test(batch_size, num_epochs, num_parallel_reads,
                             seed=21345)


if __name__ == "__main__":
  test.main()
