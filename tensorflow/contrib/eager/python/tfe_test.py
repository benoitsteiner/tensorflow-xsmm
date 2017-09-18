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
"""Tests for tfe.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.eager.python import tfe
from tensorflow.python.eager import test


class TFETest(test.TestCase):

  def testListDevices(self):
    # Expect at least one device.
    self.assertTrue(tfe.list_devices())

  def testNumGPUs(self):
    devices = tfe.list_devices()
    self.assertEqual(len(devices) - 1, tfe.num_gpus())

  def testCallingEnableEagerExecutionMoreThanOnce(self):
    # Note that eager.test.main() has already invoked enable_eager_exceution().
    with self.assertRaisesRegexp(
        ValueError,
        r"Do not call tfe\.%s more than once in the same process" %
        tfe.enable_eager_execution.__name__):
      tfe.enable_eager_execution()


if __name__ == "__main__":
  test.main()
