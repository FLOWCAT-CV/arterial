import os

import pickle5 as pickle

import torch

from .gnnUtils import nodeTransform, performInferenceGraphUNet, savePredictedGraph

def predictGraph(caseDir):
    ''' Inputs the graph derived from the segmentsArray in edge form (edges as centerline segments), 
    passes it to node form, performs the prediction by inference with the graph U-Net
    model and returns the original graph in edge form with the predicted vessel types 
    for each edge. Saves a pickle file with the predicted graph at 
    os.path.join(caseDir, "graph_pred.pickle").

    Arguments:
        - caseDir <str or path>: path to the caseDir of the case.
    
    Returns:

    '''

    # Load the edge form graph (graph.pickle) created at centerlineGraph.py
    # edgeFormGraphNx = nx.read_gpickle(os.path.join(caseDir, "graph.pickle")) # For some reason this stopped working
    with open(os.path.join(caseDir, "graph.pickle"), "rb") as pklFile:
        edgeFormGraphNx = pickle.load(pklFile)

    # Pass the graph to node form
    graphNx = nodeTransform(edgeFormGraphNx)

    # Load the trained graph U-Net model for inference
    modelPath = os.path.join(os.environ["arterialDir"], "vesselLabelling/graphUNet/model/model.pth")
    model = torch.load(modelPath)

    # Perform inference with the trained model
    predictedVessels = performInferenceGraphUNet(model, graphNx)

    # Save the predicted graph in edge form as graph_pred.pickle
    savePredictedGraph(caseDir, edgeFormGraphNx, predictedVessels)
