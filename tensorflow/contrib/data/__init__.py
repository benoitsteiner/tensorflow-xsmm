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
"""`tf.contrib.data.Dataset` API for input pipelines.

@@Dataset
@@Iterator
@@TFRecordDataset
@@FixedLengthRecordDataset
@@TextLineDataset

@@batch_and_drop_remainder
@@read_batch_features
@@rejection_resample
@@group_by_window
@@sloppy_interleave
@@sloppy_map
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.contrib.data.python.ops.dataset_ops import batch_and_drop_remainder
from tensorflow.contrib.data.python.ops.dataset_ops import Dataset
from tensorflow.contrib.data.python.ops.dataset_ops import FixedLengthRecordDataset
from tensorflow.contrib.data.python.ops.dataset_ops import group_by_window
from tensorflow.contrib.data.python.ops.dataset_ops import Iterator
from tensorflow.contrib.data.python.ops.dataset_ops import read_batch_features
from tensorflow.contrib.data.python.ops.dataset_ops import rejection_resample
from tensorflow.contrib.data.python.ops.dataset_ops import SqlDataset
from tensorflow.contrib.data.python.ops.dataset_ops import TextLineDataset
from tensorflow.contrib.data.python.ops.dataset_ops import TFRecordDataset
from tensorflow.contrib.data.python.ops.sloppy_ops import sloppy_interleave
# pylint: enable=unused-import

from tensorflow.python.util.all_util import remove_undocumented
remove_undocumented(__name__)
