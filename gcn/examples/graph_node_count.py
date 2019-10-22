
import random
import tensorflow as tf
import numpy as np

from ..graph import ArcType, NodeType, Graph, GraphType, Node, Arc 
from ..gcn import GCN

NT_GN = NodeType(4)
NT_GOD = NodeType(8)
AT_GN_GN = ArcType(NT_GN, NT_GN)
AT_GOD_GN = ArcType(NT_GOD, NT_GN)

GT = GraphType([NT_GN, NT_GOD], [AT_GN_GN, AT_GOD_GN])

def gen_graph(n_nodes):
    g = Graph(GT)
    god = Node(NT_GOD)
    g.add_node(god)
    for _ in range(n_nodes):
        node = Node(NT_GN)
        g.add_node(node)
        g.add_arc(Arc(AT_GOD_GN, god, node))
    for _ in range(2*n_nodes):
        idx1 = random.randint(0, n_nodes-1)
        idx2 = random.randint(0, n_nodes-1)
        g.add_arc(Arc(AT_GN_GN, g.nodes[NT_GN][idx1], g.nodes[NT_GN][idx2]))
        g.add_arc(Arc(AT_GN_GN, g.nodes[NT_GN][idx2], g.nodes[NT_GN][idx1]))
    y = np.array([n_nodes])
    return g, y.reshape((-1, 1))


net_structures = {NT_GN: [tf.layers.Dense(units=32, activation=tf.nn.leaky_relu)],
                  NT_GOD: [tf.layers.Dense(units=64, activation=tf.nn.leaky_relu)]}
gcn = GCN(GT, depth=5, hidden_layers=net_structures)
hl = tf.layers.Dense(units=32,activation=tf.nn.leaky_relu)
ol = tf.layers.Dense(units=1,activation=None)
nn_output = ol(hl(gcn.outputs[NT_GOD]))


labels = tf.placeholder(tf.float32, shape=(None, 1), name="labels")
loss = tf.losses.mean_squared_error(labels=labels, predictions=nn_output)

trainer = tf.compat.v1.train.AdamOptimizer().minimize(loss)


EPOCHS = 10000
with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    for epoch in range(EPOCHS):
            g, n_nodes = gen_graph(random.randint(10, 100))
            fd = {}
            for nt in g.graph_type.node_types:
                fd[gcn.inputs[nt]] = g.get_nodes_np(nt)
            for at in g.graph_type.arc_types:
                fd[gcn.inputs[at]] = g.get_arcs_np(at)
            fd[labels] = n_nodes
            l, new_repr, y_, _ = sess.run([loss, gcn.outputs[NT_GOD], nn_output, trainer], feed_dict=fd)
            print(new_repr)
            print(y_)
            print(n_nodes)
            print("loss: {}".format(l))
            #print("accuracy: {}".format(acc))
            print("---")