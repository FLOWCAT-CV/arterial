import os
import networkx as nx

# path = "/Users/pere/opt/anaconda3/envs/gnnenv/gnn_arterial/database/DatasetANCD/undone"
path = "/Users/pere/opt/anaconda3/envs/vmtkenv/graphBranchModelLink"

patID = "example"

graph_path = os.path.join(path, patID, "graph.pickle")

G = 0
G = nx.readwrite.gpickle.read_gpickle(graph_path)

if os.path.isfile(os.path.join(path, patID, "graph_label.pickle")):
    print("Label is already made")
else:
    for node in G.nodes:
        print("Node", node, "connecting edges:")
        for edge in G.edges(node):
            print("     ", G.edges[edge]["CellID"])
        # input("This node's type is: ")
        if G.degree(node) == 1:
            G.nodes(data=True)[node]["nodetype"] = 1
            print("This node's type is: 1")
        else:
            G.nodes(data=True)[node]["nodetype"] = input("This node's type is: ")

    for n0, n1 in G.edges:
        print("Edge", G[n0][n1]["CellID"])
        G[n0][n1]["edgetype"] = input("This edge's type is: ")

    nx.readwrite.gpickle.write_gpickle(G, os.path.join(path, patID, "graph_label.pickle"))