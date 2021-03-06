
import random
import tensorflow as tf
import numpy as np

from ..graph import ArcType, NodeType, Graph, GraphType, Node, Arc, GraphSet
from ..gcn import GCN

tf.compat.v1.disable_eager_execution()


NT_GN = NodeType(4)
NT_GOD = NodeType(16)
AT_GN_GN = ArcType(NT_GN, NT_GN, bidirectional=True)
AT_GOD_GN = ArcType(NT_GOD, NT_GN, bidirectional=True)

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
    return g, y


net_structures = {NT_GN: [tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)],
                  NT_GOD: [tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu)]}
gcn = GCN(GT, depth=5, hidden_layers=net_structures)
hl = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)
ol = tf.keras.layers.Dense(units=1, activation=None)
nn_output = ol(hl(gcn.outputs[NT_GOD]))


labels = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="labels")
loss = tf.compat.v1.losses.mean_squared_error(labels=labels, predictions=nn_output)
tb_loss = tf.compat.v1.summary.scalar("mse", loss)

trainer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

data = [gen_graph(random.randint(10, 100)) for _ in range(100)]
X, y = zip(*data)

gs = GraphSet(X, y)

EPOCHS = 1000
epoch = 0
with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run(session=sess)
    tb_writer = tf.compat.v1.summary.FileWriter("./tb/examples/node_count", session=sess)
    while epoch < EPOCHS:
        g, n_nodes, epoch_completed = gs.next_batch(10)
        fd = {}
        for nt in g.graph_type.node_types:
            fd[gcn.inputs[nt]] = g.get_nodes_np(nt)
        for at in g.graph_type.arc_types:
            fd[gcn.inputs[at]] = g.get_arcs_np(at)
        fd[labels] = n_nodes
        tbl, l, new_repr, y_, _ = sess.run([tb_loss, loss, gcn.outputs[NT_GOD], nn_output, trainer], feed_dict=fd)
        tb_writer.add_summary(tbl)
        print("loss: {}".format(l))
        if epoch_completed:
            epoch += 1
