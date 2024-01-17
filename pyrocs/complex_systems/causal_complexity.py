import numpy as np
import networkx as nx


def cyclomatic_complexity(A : np.ndarray):
    '''
    Cyclomatic complexity reflects the number of linearly 
    independent paths within a system of interest 
    and can be calculated using the number of edges 
    (E), nodes (N), and connected components (P)
    :cite:p:`ebert_cyclomatic_2016`.
    The equation within the package follows the formulations from
    :cite:p:`naugle_feedback_2023` as follows:
    
    .. math::
        M = E - N + 2P,
    
    where :math:`M` is the cyclomatic complexity, :math:`E` is the total number 
    of edges, :math:`N` is the total number of nodes, and :math:`P` is 
    the number of connected components. Generally, lower cyclomatic 
    complexity values indicate that the content has fewer interconnections 
    and thus, can be understood more linearly (relative to 
    higher cyclomatic complexity values).     
    
    Args:
        A: array
    Returns:
        cyclomatic complexity of the graph   
    '''

    G = nx.from_numpy_matrix(A)

    P = nx.number_connected_components(G)
    E = nx.number_of_edges(G)
    N = nx.number_of_nodes(G)

    return E - N + 2 * P 


def feedback_density(A : np.ndarray):
    '''
    Feedback density captures the fraction of edges :math:`(E_{loop})` 
    and nodes (:math:`N_{loop}`) that are involved in at least one feedback loop.
    As such, it reflects the potential for cyclic behaviors. 
    The equation within the package follows the formulations 
    from :cite:p:`naugle_feedback_2023`
    as follows:
    
    .. math ::
        D = (E_{loop}+N_{loop})/(E+N),
    
    where D is the feedback density, :math:`E_{loop}` is the fraction of edges, 
    :math:`N_{loop}` is the fraction of nodes, :math:`E` is the total number of edges, 
    and :math:`N` is the total number of nodes. Feedback density values 
    are normalized between 0 and 1, where 0 indicates that no
    feedback loops (i.e., paths that begin and ened at the same node) 
    are present in the system while 1 indicates all nodes and 
    edges are included in one or more feedback loops.
    
    Args:
        A: array
    Returns:
        feedback density of the graph   
    '''

    G = nx.from_numpy_matrix(A, parallel_edges=False, create_using=nx.MultiDiGraph)

    Etot = nx.number_of_edges(G)
    Ntot = nx.number_of_nodes(G)

    Eloop = 0
    for edge in G.edges:
        try:
            if nx.has_path(G, edge[1], edge[0]):
                Eloop = Eloop + 1
        except nx.NetworkXNoPath:
            pass

    Nloop = 0
    for node in G.nodes:
        try:
            if nx.find_cycle(G, node) != None:
                Nloop = Nloop + 1
        except nx.NetworkXNoCycle:
            pass

    return (Eloop + Nloop) / (Etot + Ntot)


def causal_complexity(A: np.ndarray):
    '''
    Causal complexity measures the underlying causal structure 
    of a system by considering both the system’s intricacy as
    well as interconnectedness. The equation within the package 
    follows the formulations from
    :cite:p:`naugle_feedback_2023`,
    who generate a non-normalized measure of causal complexity as a 
    product of cyclomatic complexity and 1 + feedback density as follows:
    
    .. math::
    
        C = M*(1+D) = (E - N + 2P) * (1+(E_{loop}+N_{loop})/(E+N)),

    where :math:`C` is the causal complexity, :math:`M` is the cyclomatic 
    complexity, and :math:`D` is the feedback density. Cyclomatic complexity 
    reflects the number of linearly independent paths within a system 
    of interest and can be calculated using the number of edges 
    (:math:`E`), nodes (:math:`N`), and connected components (:math:`P`) 
    :cite:p:`ebert_cyclomatic_2016`. 
    In contrast, feedback density captures the fraction of edges 
    (:math:`E_{loop}`) and nodes (:math:`N_loop`) that are involved in at least one 
    feedback loop. As such, it reflects the potential for cyclic behaviors. 
    Jointly, the measure of causal complexity reflects the number of paths 
    through a system weighted to reflect those with feedback loops. 
    Thus, systems with more feedback density will have higher values 
    of causal complexity than those systems with lower feedback density.
    
    Args:
        A: array
    Returns:
        causal complexity of the graph
    '''
    M = cyclomatic_complexity(A)
    D = feedback_density(A)

    return (M * (1. + D))
