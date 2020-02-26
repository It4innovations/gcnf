
import random
import tensorflow as tf
import numpy as np

from ..graph import ArcType, NodeType, Graph, GraphType, Node, Arc, GraphSet
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
    return g, y


data = [gen_graph(100) for _ in range(100)]
X, y = zip(*data)
gs = GraphSet(X, y)

net_structures = {NT: [tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)]}
gcn = GCN(GT, depth=5, hidden_layers=net_structures)
hl = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)
ol = tf.keras.layers.Dense(units=1, activation=None)
nn_output = ol(hl(gcn.outputs[NT]))


labels = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="labels")
loss = tf.compat.v1.losses.mean_squared_error(labels=labels, predictions=nn_output)
tb_loss = tf.compat.v1.summary.scalar("mse", loss)

trainer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

EPOCHS = 10000
epoch = 0
with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run(session=sess)
    tb_writer = tf.compat.v1.summary.FileWriter("./tb/examples/node_degrees", session=sess)
    while epoch < EPOCHS:
        g, d, epoch_completed = gs.next_batch(10)
        fd = {}
        for nt in g.graph_type.node_types:
            fd[gcn.inputs[nt]] = g.get_nodes_np(nt)
        for at in g.graph_type.arc_types:
            fd[gcn.inputs[at]] = g.get_arcs_np(at)
        fd[labels] = np.array(d).reshape((len(g.nodes[NT]), 1))
        tbl, l, new_repr, y_, _ = sess.run([tb_loss, loss, gcn.outputs[NT], nn_output, trainer], feed_dict=fd)
        tb_writer.add_summary(tbl)
        print("loss: {}".format(l))
        if epoch_completed:
            epoch += 1
