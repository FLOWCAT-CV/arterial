import os
import copy

import numpy as np
import networkx as nx

import tensorflow as tf

from graph_nets.demos.models import EncodeProcessDecode
from graph_nets import utils_np, utils_tf

# Number of different bifurcation types
BIFTYPENUM = 19
# Number of different vessel types
VESTYPENUM = 16
# Number of processing steps of the GNN
numProcessingSteps = 10
# Define dicts
edgeTypes = {
    0: "other",
    1: "AA",
    2: "BT",
    3: "RCCA",
    4: "LCCA",
    5: "RSA",
    6: "LSA",
    7: "RVA",
    8: "LVA",
    9: "RICA",
    10: "LICA",
    11: "RECA",
    12: "LECA",
    13: "BA",
    14: "AA+BT",
    15: "RVA+LVA"  
}
nodeTypes = {
    0: "other",
    1: "endpoint",
    2: "AA_BT",
    3: "AA-LCCA",
    4: "AA-LSA",
    5: "AA-RSA",
    6: "BT-LCCA",
    7: "BT-RCCA/RSA",
    8: "RSA-RVA",
    9: "LSA-LVA",
    10: "RCCA-RICA/RECA",
    11: "LCCA-LICA/LECA",
    12: "RVA/LVA-BA",
    13: "BT-LSA",
    14: "AA-BT/LCCA",
    15: "AA-LVA",
    16: "BT-LSA",
    17: "AA-RCCA",
    18: "LCCA-LSA"
}

# Should be and environment variable
gnnModelPath = "/Users/pere/GitHub/arterial/arterial/vesselLabelling/graphNet/model/model.ckpt"
gnnModelMetaPath = "/Users/pere/GitHub/arterial/arterial/vesselLabelling/graphNet/model/model.ckpt.meta"

def gnnInference(caseDir):
    # Define path to graph
    graphPath = os.path.join(caseDir, "graph.pickle")
    print("Predicting graph", graphPath)

    # Define model
    model = EncodeProcessDecode(edge_output_size=VESTYPENUM, node_output_size=BIFTYPENUM)
    # Load input graph
    feedDict, inputPH, inputGraph = createFeedDictSingle(graphPath)
    # Define output operation
    outputOp = model(inputPH, numProcessingSteps)
    # Restore variables from disk
    sess = tf.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.import_meta_graph(gnnModelMetaPath)
    saver.restore(sess, gnnModelPath)

    # Perform inference
    inferenceValues = sess.run({"output": outputOp
                                }, feed_dict=feedDict)

    # Pass graph tuple to networkx graph
    output = utils_np.graphs_tuple_to_networkxs(inferenceValues["output"][0])[0]
    output = featuresToType(output, inputGraph)
    sess.close()

    # Save predicted graph
    nx.write_gpickle(output, os.path.join(caseDir, "graph_pred.pickle"))


def createFeedDictSingle(graphPath, graph=None):
    """Creates placeholders for the model training and evaluation.

    Args:
        graphPath: path to labeled input graph (networkx graph, .pickle format)
        input_ph: The input graph's placeholders, as a graph namedtuple.
        label_ph: The label graph's placeholders, as a graph namedtuple.
    Returns:
        feedDict: The feed `dict` of input and label placeholders and data.
        
    """
    def generateGraph(graphPath):
        # Read pickled graph
        graph = nx.read_gpickle(graphPath)
        graph.name = graphPath
        graph.graph["features"] = np.array([0.0])

        return graph

    # Generate single graph and create input placeholder
    if graph == None:
        graph = generateGraph(graphPath)
    inputGraph = utils_np.networkxs_to_graphs_tuple([graph])
    inputPH = utils_tf.placeholders_from_networkxs([graph])
    # Define feed dict
    feedDict = {inputPH: inputGraph}

    return feedDict, inputPH, graph
    

def featuresToType(predGraph, inputGraph):
    # Define new graph with same nodes and edges and information from the input graph
    outputGraph = nx.Graph()
    for node in predGraph.nodes:
        outputGraph.add_node(node)
        outputGraph.nodes(data=True)[node]["pos"] = inputGraph.nodes(data=True)[node]["pos"]
        outputGraph.nodes(data=True)[node]["nodetype"] = np.argmax(predGraph.nodes(data=True)[node]["features"])
        outputGraph.nodes(data=True)[node]["nodeTypeName"] = nodeTypes[np.argmax(predGraph.nodes(data=True)[node]["features"])]
        outputGraph.nodes(data=True)[node]["features"] = predGraph.nodes(data=True)[node]["features"]

    for n0, n1, _ in predGraph.edges:
        outputGraph.add_edge(n0, n1)
        outputGraph[n0][n1]["CellID"] = inputGraph[n0][n1]["CellID"]
        outputGraph[n0][n1]["edgetype"] = np.argmax(predGraph[n0][n1][0]["features"])
        outputGraph[n0][n1]["edgeTypeName"] = edgeTypes[np.argmax(predGraph[n0][n1][0]["features"])]
        outputGraph[n0][n1]["features"] = predGraph[n0][n1][0]["features"]

    return outputGraph