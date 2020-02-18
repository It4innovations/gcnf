
import random
import tensorflow as tf
import numpy as np

from ..graph import ArcType, NodeType, Graph, GraphType, Node, Arc 
from ..gcn import GCN

tf.compat.v1.disable_eager_execution()


NT = NodeType(8)
AT = ArcType(NT, NT, bidirectional=True)

GT = GraphType([NT], [AT])


def gen_graph(n_nodes):
    g = Graph(GT)
    for _ in range(n_nodes):
        g.add_node(Node(NT))
    y = np.zeros((n_nodes))
    for _ in range(2*n_nodes):
        idx1 = random.randint(0, n_nodes-1)
        idx2 = random.randint(0, n_nodes-1)
        y[idx1] += 1
        y[idx2] += 1
        g.add_arc(Arc(AT, g.nodes[NT][idx1], g.nodes[NT][idx2]))
        g.add_arc(Arc(AT, g.nodes[NT][idx2], g.nodes[NT][idx1]))
    return g, y.reshape((-1, 1))


g, d = gen_graph(100)

print(g.get_nodes_np(NT), g.get_arcs_np(AT))
print(d)

net_structures = {NT: [tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)]}
gcn = GCN(GT, depth=5, hidden_layers=net_structures)
hl = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)
ol = tf.keras.layers.Dense(units=1, activation=None)
nn_output = ol(hl(gcn.outputs[NT]))


labels = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="labels")
loss = tf.compat.v1.losses.mean_squared_error(labels=labels, predictions=nn_output)

trainer = tf.compat.v1.train.AdamOptimizer().minimize(loss)


EPOCHS = 10000
with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run(session=sess)
    for epoch in range(EPOCHS):
        fd = {}
        for nt in g.graph_type.node_types:
            fd[gcn.inputs[nt]] = g.get_nodes_np(nt)
        for at in g.graph_type.arc_types:
            fd[gcn.inputs[at]] = g.get_arcs_np(at)
        fd[labels] = d
        l, new_repr, y_, _ = sess.run([loss, gcn.outputs[NT], nn_output, trainer], feed_dict=fd)
        print("loss: {}".format(l))
