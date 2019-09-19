import tensorflow as tf

from keras.utils.np_utils import to_categorical   

from .graph import ArcType, NodeType, Graph, GraphType, Node, Arc 
from .gcn import GCN

NT = NodeType(4)
AT = ArcType(NT, NT)
GT = GraphType([NT], [AT])

def convert(image):
    g = Graph(GT)
    rows, cols = image.shape
    for r_idx in range(rows):
        for c_idx in range(cols):
            g.add_node(Node(NT, [image[r_idx, c_idx]]))

    for r_idx in range(rows):
        for c_idx in range(cols):
            from_idx = r_idx * cols + c_idx
            lc_idx = c_idx - 1
            if lc_idx >= 0:
                to_idx = r_idx * cols + lc_idx
                g.add_arc(Arc(AT, g.nodes[NT][from_idx], g.nodes[NT][to_idx]))

            rc_idx = c_idx + 1
            if rc_idx < cols:
                to_idx = rc_idx * cols + rc_idx
                g.add_arc(Arc(AT, g.nodes[NT][from_idx], g.nodes[NT][to_idx]))
            
            ur_idx = r_idx - 1
            if ur_idx >= 0:
                to_idx = ur_idx * cols + c_idx
                g.add_arc(Arc(AT, g.nodes[NT][from_idx], g.nodes[NT][to_idx]))

            dr_idx = r_idx + 1
            if dr_idx < rows:
                to_idx = dr_idx * cols + c_idx
                g.add_arc(Arc(AT, g.nodes[NT][from_idx], g.nodes[NT][to_idx]))                
    return g

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_graphs = []
for i in range(1):
    X_train_graphs.append(convert(X_train[i]))

gcn = GCN(GT)
hl = tf.layers.Dense(units=32,activation=tf.nn.relu)
ol = tf.layers.Dense(units=10,activation=None)

nn_output = ol(hl(gcn.outputs[NT]))

labels = tf.placeholder(tf.float32, shape=(None, 10), name="labels")
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_output, labels=labels))
trainer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

no_classes = 10
y = to_categorical(y_train, no_classes)

with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    for epoch in range(1000):
        for i in range(len(X_train_graphs)):
            g = X_train_graphs[i]
            fd = {}
            for nt in g.graph_type.node_types:
                fd[gcn.inputs[nt]] = g.get_nodes_np(nt)
            for at in g.graph_type.arc_types:
                fd[gcn.inputs[at]] = g.get_arcs_np(at)
            fd[labels] = y[i].reshape((1,10))
            l, _ = sess.run([loss, trainer], feed_dict=fd)
            print(l)
