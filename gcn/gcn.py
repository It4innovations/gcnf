import tensorflow as tf
import numpy as np


class GCN:
    def __init__(self, graph_type, depth, hidden_layers):
        assert isinstance(hidden_layers, dict)
        assert set(hidden_layers.keys()) == set(graph_type.node_types)
        self.gt = graph_type
        self.nns = {}
        for nt in self.gt.node_types:
            assert isinstance(hidden_layers[nt], (list, tuple))
            layers = []
            if hidden_layers:
                layers += hidden_layers[nt] + [tf.layers.Dense(units=nt.vector_size, activation=None)]
            self.nns[nt] = layers
        self.inputs = self.create_input_placeholders()
        self.outputs = self.build_net(self.inputs, depth)

    def create_input_placeholders(self):
        placeholders = {}
        for at in self.gt.arc_types:
            placeholders[at] = tf.placeholder(tf.int32, (None, 2), 'ph-at-{}'.format(at.name))
        for nt in self.gt.node_types:
            placeholders[nt] = tf.placeholder(tf.float32, (None, nt.vector_size), 'ph-nt-{}'.format(nt.name))
        return placeholders

    def aggregate_neighborhood(self, state, read_from_ids, agregate_to_ids, n_count):
        values = tf.nn.embedding_lookup(state, read_from_ids)
        summary_sum = tf.math.unsorted_segment_sum(values, agregate_to_ids, n_count)
        summary_max = tf.math.unsorted_segment_max(values, agregate_to_ids, n_count)
        summary_max = tf.maximum(summary_max, -1.)
        summary_min = tf.math.unsorted_segment_min(values, agregate_to_ids, n_count)
        summary_min = tf.minimum(summary_min, 1.)
        return [summary_sum, summary_max, summary_min]

    def build_nt_nn(self, node_type, state, neighborhoods):
        layer_input = tf.concat([state] + neighborhoods, axis=1)
        for l in self.nns[node_type]:
            layer_input = l(layer_input)
        return tf.add(layer_input, state)

    def build_net(self, inputs, depth):
        state = inputs
        src_nodes = {}
        dst_nodes = {}
        for at in self.gt.arc_types:
            arcs = state[at]
            src_nodes[at] = arcs[:, 0]
            dst_nodes[at] = arcs[:, 1]
        n_count = {}
        for nt in self.gt.node_types:
            n_count[nt] = tf.shape(state[nt])[0]
        for _ in range(depth):
            new_state = {}
            for nt in self.gt.node_types:
                neighborhoods = []
                for at in self.gt.arc_types:
                    if at.from_nt == nt:
                        neighborhood = self.aggregate_neighborhood(
                            state=state[at.to_nt],
                            read_from_ids=dst_nodes[at],
                            agregate_to_ids=src_nodes[at],
                            n_count=n_count[nt])
                        neighborhoods += neighborhood
                new_state[nt] = self.build_nt_nn(nt, state[nt], neighborhoods)
            state = new_state
        return state
