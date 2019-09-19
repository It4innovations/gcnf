import numpy as np


class ArcType:
    def __init__(self, from_nt, to_nt, name=""):
        assert isinstance(from_nt, NodeType)
        assert isinstance(to_nt, NodeType)
        self.name = name
        self.from_nt = from_nt
        self.to_nt = to_nt
        self.counter = 0


class Arc:
    def __init__(self, arc_type, from_node, to_node):
        assert isinstance(arc_type, ArcType)
        assert isinstance(from_node, Node)
        assert isinstance(to_node, Node)
        assert from_node.node_type == arc_type.from_nt
        assert to_node.node_type == arc_type.to_nt
        self.arc_type = arc_type
        self.values = np.array([from_node.id, to_node.id])
        self.id = self.arc_type.counter
        self.arc_type.counter += 1


class NodeType:
    def __init__(self, vector_size, name=""):
        self.name = name
        self.vector_size = vector_size
        self.counter = 0


class Node:
    def __init__(self, node_type, values=[]):
        assert isinstance(node_type, NodeType)
        assert len(values) <= node_type.vector_size
        self.node_type = node_type
        self.values = np.concatenate((np.array(values), np.zeros(node_type.vector_size - len(values))))
        self.id = self.node_type.counter
        self.node_type.counter += 1


class GraphType:
    def __init__(self, node_types, arc_types):
        self.node_types = node_types
        self.arc_types = arc_types


class Graph:
    def __init__(self, graph_type):
        assert isinstance(graph_type, GraphType)
        self.nodes = {}
        self.arcs = {}
        self.graph_type = graph_type
        for at in graph_type.arc_types:
            self.arcs[at] = []
        for nt in graph_type.node_types:
            self.nodes[nt] = []

    def add_arc(self, arc):
        assert isinstance(arc, Arc)
        self.arcs[arc.arc_type].append(arc)

    def add_node(self, node):
        assert isinstance(node, Node)
        self.nodes[node.node_type].append(node)

    def get_nodes_np(self, node_type):
        assert isinstance(node_type, NodeType)
        nodes = []
        for n in self.nodes[node_type]:
            nodes.append(n.values)
        return np.array(nodes)

    def get_arcs_np(self, arc_type):
        assert isinstance(arc_type, ArcType)
        arcs = []
        for a in self.arcs[arc_type]:
            arcs.append(a.values)
        return np.array(arcs)
