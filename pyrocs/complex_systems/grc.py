import networkx as nx
import numpy as np


def grc(A : np.ndarray, directed : bool):
    """
    Global reaching centrality (GRC) measures the level of hierarchy within a network based on flow. The equation within the package follows the formulations from [Mones et al., 2012] (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0033799), who quantify GRC as the difference between the maximum and the average value of the local reach centralities of nodes within the network:
    
    GRC = ∑[(C_R^max-C_R(i)]/(N-1),
    
    where C_R is the local reach centrality that reflects the proportion of nodes that can be reached from a particular node i (C_R(i)) or reflects the maximum value of local reach centrality within the network (C_R^max) and N is the number of nodes present within the network. GRC values can range from 0 to 1, with lower values indicating lower hierarchy and vice versa [Lakkaraju et al., 2019] (https://www.osti.gov/servlets/purl/1639730).

    Args:
        A: Square matrix of adjacencies in the network
        directed (bool): If true, assume A represents a directed graph (row -> column).
            If false, assume A represents an undirected graph.
    Returns:
        Global reaching centrality of the graph 
    """

    if directed:
        G = nx.from_numpy_matrix(A, nx.DiGraph)
    else:
        G = nx.from_numpy_matrix(A)
        
    if G.number_of_edges() == 0:
        print("WARNING: Social network to compute GRC over has no edges!")
        return 0.0
    else:
        return nx.global_reaching_centrality(G)
