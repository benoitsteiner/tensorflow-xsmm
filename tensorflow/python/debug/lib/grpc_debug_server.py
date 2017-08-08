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
"""gRPC debug server in Python."""
# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import threading
import time

from concurrent import futures
import grpc
from six.moves import queue

from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_service_pb2_grpc

DebugWatch = collections.namedtuple("DebugWatch",
                                    ["node_name", "output_slot", "debug_op"])


def _watch_key_event_reply(to_enable, node_name, output_slot, debug_op):
  """Make EventReply proto to represent a request to watch/unwatch a debug op.

  Args:
    to_enable: (`bool`) whether the request is to enable the watch key.
    node_name: (`str`) name of the node.
    output_slot: (`int`) output slot of the tensor.
    debug_op: (`str`) the debug op attached to node_name:output_slot tensor to
      watch or unwatch.

  Returns:
    An EventReply proto.
  """
  event_reply = debug_service_pb2.EventReply()
  state_change = event_reply.debug_op_state_changes.add()
  state_change.change = (
      debug_service_pb2.EventReply.DebugOpStateChange.ENABLE
      if to_enable else debug_service_pb2.EventReply.DebugOpStateChange.DISABLE)
  state_change.node_name = node_name
  state_change.output_slot = output_slot
  state_change.debug_op = debug_op
  return event_reply


class EventListenerBaseStreamHandler(object):
  """Per-stream handler of EventListener gRPC streams."""

  def __init__(self):
    """Constructor of EventListenerBaseStreamHandler."""

  def on_core_metadata_event(self, event):
    """Callback for core metadata.

    Args:
      event: The Event proto that carries a JSON string in its
        `log_message.message` field.
    """
    raise NotImplementedError(
        "on_core_metadata_event() is not implemented in the base servicer "
        "class")

  def on_graph_def(self, graph_def, device_name, wall_time):
    """Callback for Event proto received through the gRPC stream.

    This Event proto carries a GraphDef, encoded as bytes, in its graph_def
    field.

    Args:
      graph_def: A GraphDef object.
      device_name: Name of the device on which the graph was created.
      wall_time: An epoch timestamp (in microseconds) for the graph.
    """
    raise NotImplementedError(
        "on_graph_def() is not implemented in the base servicer class")

  def on_value_event(self, event):
    """Callback for Event proto received through the gRPC stream.

    This Event proto carries a Tensor in its summary.value[0] field.

    Args:
      event: The Event proto from the stream to be processed.
    """
    raise NotImplementedError(
        "on_value_event() is not implemented in the base servicer class")


class EventListenerBaseServicer(debug_service_pb2_grpc.EventListenerServicer):
  """Base Python class for gRPC debug server."""

  def __init__(self, server_port, stream_handler_class):
    """Constructor.

    Args:
      server_port: (int) Port number to bind to.
      stream_handler_class: A class of the base class
        `EventListenerBaseStreamHandler` that will be used to constructor
        stream handler objects during `SendEvents` calls.
    """

    self._server_port = server_port
    self._stream_handler_class = stream_handler_class

    self._server_lock = threading.Lock()
    self._server_started = False
    self._stop_requested = False

    self._event_reply_queue = queue.Queue()
    self._gated_grpc_debug_watches = set()

  def SendEvents(self, request_iterator, context):
    """Implementation of the SendEvents service method.

    This method receives streams of Event protos from the client, and processes
    them in ways specified in the on_event() callback. The stream is
    bi-directional, but currently only the client-to-server stream (i.e., the
    stream from the debug ops to the server) is used.

    Args:
      request_iterator: The incoming stream of Event protos.
      context: Server context.

    Raises:
      ValueError: If there are more than one core metadata events.

    Yields:
      An empty stream of responses.
    """
    core_metadata_count = 0

    # A map from GraphDef hash to a list of received chunks.
    graph_def_chunks = {}
    tensor_chunks = {}

    stream_handler = None
    for event in request_iterator:
      if not stream_handler:
        stream_handler = self._stream_handler_class()

      if event.graph_def:
        maybe_graph_def, maybe_device_name, maybe_wall_time = (
            self._process_encoded_graph_def_in_chunks(event, graph_def_chunks))
        if maybe_graph_def:
          stream_handler.on_graph_def(
              maybe_graph_def, maybe_device_name, maybe_wall_time)
      elif event.log_message.message:
        core_metadata_count += 1
        if core_metadata_count > 1:
          raise ValueError(
              "Expected one core metadata event; received multiple")
        stream_handler.on_core_metadata_event(event)
      elif event.summary and event.summary.value:
        maybe_tensor_event = self._process_tensor_event_in_chunks(
            event, tensor_chunks)
        if maybe_tensor_event:
          stream_handler.on_value_event(maybe_tensor_event)

    # The server writes EventReply messages, if any.
    while not self._event_reply_queue.empty():
      yield self._event_reply_queue.get()

  def _process_tensor_event_in_chunks(self, event, tensor_chunks):
    """Possibly reassemble event chunks.

    Due to gRPC's message size limit, a large tensor can be encapsulated in
    multiple Event proto chunks to be sent through the debugger stream. This
    method keeps track of the chunks that have arrived, reassemble all chunks
    corresponding to a tensor when they have arrived and return the reassembled
    Event proto.

    Args:
      event: The single Event proto that has arrived.
      tensor_chunks: A dict used to keep track of the Event protos that have
        arrived but haven't been reassembled.

    Returns:
      If all Event protos corresponding to a tensor have arrived, returns the
      reassembled Event proto. Otherwise, return None.
    """

    value = event.summary.value[0]
    debugger_plugin_metadata = json.loads(
        value.metadata.plugin_data[0].content)
    device_name = debugger_plugin_metadata["device"]
    num_chunks = debugger_plugin_metadata["numChunks"]
    chunk_index = debugger_plugin_metadata["chunkIndex"]

    if num_chunks <= 1:
      return event

    debug_node_name = value.node_name
    timestamp = int(event.wall_time)
    tensor_key = "%s_%s_%d" % (device_name, debug_node_name, timestamp)

    if tensor_key not in tensor_chunks:
      tensor_chunks[tensor_key] = [None] * num_chunks

    chunks = tensor_chunks[tensor_key]
    if value.tensor.tensor_content:
      chunks[chunk_index] = value.tensor
    elif value.tensor.string_val:
      chunks[chunk_index] = event

    if None not in chunks:
      if value.tensor.tensor_content:
        event.summary.value[0].tensor.tensor_content = b"".join(
            chunk.tensor_content for chunk in chunks)
        del tensor_chunks[tensor_key]
        return event
      elif value.tensor.string_val:
        merged_event = chunks[0]
        for chunk in chunks[1:]:
          merged_event.summary.value[0].tensor.string_val.extend(
              list(chunk.summary.value[0].tensor.string_val))
        return merged_event

  def _process_encoded_graph_def_in_chunks(self,
                                           event,
                                           graph_def_chunks):
    """Process an Event proto containing a chunk of encoded GraphDef.

    Args:
      event: the Event proto containing the chunk of encoded GraphDef.
      graph_def_chunks: A dict mapping keys for GraphDefs (i.e.,
      "<graph_def_hash>,<device_name>,<wall_time>") to a list of chunks of
      encoded GraphDefs.

    Returns:
      If all chunks of the GraphDef have arrived,
        return decoded GraphDef proto, device name, wall_time.
      Otherwise,
        return None, None, None.
    """
    graph_def = graph_pb2.GraphDef()
    index_bar_0 = event.graph_def.find(b"|")
    index_bar_1 = event.graph_def.find(b"|", index_bar_0 + 1)
    index_bar_2 = event.graph_def.find(b"|", index_bar_1 + 1)
    graph_def_hash_device_timestamp = event.graph_def[:index_bar_0]
    chunk_index = int(event.graph_def[index_bar_0 + 1 : index_bar_1])
    num_chunks = int(event.graph_def[index_bar_1 + 1 : index_bar_2])
    if graph_def_hash_device_timestamp not in graph_def_chunks:
      graph_def_chunks[graph_def_hash_device_timestamp] = [None] * num_chunks
    graph_def_chunks[graph_def_hash_device_timestamp][
        chunk_index] = event.graph_def[index_bar_2 + 1:]
    if all(graph_def_chunks[graph_def_hash_device_timestamp]):
      device_name = graph_def_hash_device_timestamp.split(b",")[1]
      wall_time = int(graph_def_hash_device_timestamp.split(b",")[2])
      graph_def.ParseFromString(
          b"".join(graph_def_chunks[graph_def_hash_device_timestamp]))
      del graph_def_chunks[graph_def_hash_device_timestamp]
      self._process_graph_def(graph_def)
      return graph_def, device_name, wall_time
    else:
      return None, None, None

  def _process_graph_def(self, graph_def):
    for node_def in graph_def.node:
      if (debug_data.is_debug_node(node_def.name) and
          node_def.attr["gated_grpc"].b):
        node_name, output_slot, _, debug_op = (
            debug_data.parse_debug_node_name(node_def.name))
        self._gated_grpc_debug_watches.add(
            DebugWatch(node_name, output_slot, debug_op))

  def run_server(self):
    """Start running the server.

    Blocks until `stop_server` is invoked.

    Raises:
      ValueError: If server stop has already been requested, or if the server
        has already started running.
    """
    self._server_lock.acquire()
    try:
      if self._stop_requested:
        raise ValueError("Server has already stopped")
      if self._server_started:
        raise ValueError("Server has already started running")

      self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
      debug_service_pb2_grpc.add_EventListenerServicer_to_server(self,
                                                                 self.server)
      self.server.add_insecure_port("[::]:%d" % self._server_port)
      self.server.start()
      self._server_started = True
    finally:
      self._server_lock.release()

    while not self._stop_requested:
      time.sleep(1.0)

  def stop_server(self, grace=1.0):
    """Request server stopping.

    Once stopped, server cannot be stopped or started again. This method is
    non-blocking. Call `wait()` on the returned event to block until the server
    has completely stopped.

    Args:
      grace: Grace period in seconds to be used when calling `server.stop()`.

    Raises:
      ValueError: If server stop has already been requested, or if the server
        has not started running yet.

    Returns:
      A threading.Event that will be set when the server has completely stopped.
    """
    self._server_lock.acquire()
    try:
      if not self._server_started:
        raise ValueError("Server has not started running")
      if self._stop_requested:
        raise ValueError("Server has already stopped")

      self._stop_requested = True
      return self.server.stop(grace=grace)
    finally:
      self._server_lock.release()

  def request_watch(self, node_name, output_slot, debug_op):
    """Request enabling a debug tensor watch.

    This will let the server send a EventReply to the client side
    (i.e., the debugged TensorFlow runtime process) to request adding a watch
    key (i.e., <node_name>:<output_slot>:<debug_op>) to the list of enabled
    watch keys. The list applies only to debug ops with the attribute
    gated_grpc=True.

    The request will take effect on the next debugged `Session.run()` call.

    To disable the watch, use `request_unwatch()`.

    Args:
      node_name: (`str`) name of the node that the to-be-watched tensor belongs
        to, e.g., "hidden/Weights".
      output_slot: (`int`) output slot index of the tensor to watch.
      debug_op: (`str`) name of the debug op to enable. This should not include
        any attribute substrings.
    """
    self._event_reply_queue.put(
        _watch_key_event_reply(True, node_name, output_slot, debug_op))

  def request_unwatch(self, node_name, output_slot, debug_op):
    """Request disabling a debug tensor watch.

    The request will take effect on the next debugged `Session.run()` call.

    This is the opposite of `request_watch()`.

    Args:
      node_name: (`str`) name of the node that the to-be-watched tensor belongs
        to, e.g., "hidden/Weights".
      output_slot: (`int`) output slot index of the tensor to watch.
      debug_op: (`str`) name of the debug op to enable. This should not include
        any attribute substrings.
    """
    self._event_reply_queue.put(
        _watch_key_event_reply(False, node_name, output_slot, debug_op))

  def gated_grpc_debug_watches(self):
    """Get the list of debug watches with attribute gated_grpc=True.

    Since the server receives `GraphDef` from the debugged runtime, it can only
    return such debug watches that it has received so far.

    Returns:
      A `list` of `DebugWatch` `namedtuples` representing the debug watches with
      gated_grpc=True. Each `namedtuple` element has the attributes:
        `node_name` as a `str`,
        `output_slot` as an `int`,
        `debug_op` as a `str`.
    """
    return list(self._gated_grpc_debug_watches)
