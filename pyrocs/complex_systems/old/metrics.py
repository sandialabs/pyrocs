import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import networkx as nx
import lzma as lzma
import gzip
import sys as sys
import dataIO.cdfIO as cdfio
import os.path
import math

from LZMAEstimator import LZMAEstimator
from DataSplitter import FullSplitter, QSplitter, QSubsetter

from networkx.exception import NetworkXNoCycle, NetworkXNoPath

sns.set(style="whitegrid", color_codes=True)

"""
metrics file.
"""
# Complexity


def cyclomatic_complexity(G):
    """
    Calculates the cyclomatic complexity of a networkx graph. G should be the ground truth. Calculate the cyclomatic complexity per Agent, Group or System graph, not as a whole (since cyclomatic complexity takes into account the # of connected components)

    The definition is: M = E - N + 2P, where E = edges, N = nodes and P = number of connected components.
    :param G:
    :return:
    """
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()

    g_undire = G.to_undirected()
    n_connected_components = nx.number_connected_components(g_undire)

    # print(str(n_edges) + " " + str(n_nodes))
    return (n_edges - n_nodes + 2 * n_connected_components)


def feedback_density(G):
    """
    Calculates the feedback density of a networkx graph. G should be the ground truth.

    The definition is: D = (E_loop + N_loop)/(E + N), where E_loop = number of edges included in at least one feedback loop, N_loop = number of nodes included in at least one feedback loop, E = number of edges and N = number of nodes.
    :param G:
    :return:
    """

    #TODO: there might be a more efficient way to implement these loops
    n_feedback_edges = 0
    for edge in G.edges:
        try:
            if nx.has_path(G, edge[1], edge[0]):
                n_feedback_edges = n_feedback_edges + 1
        except NetworkXNoPath:
            pass
    n_feedback_nodes = 0
    for node in G.nodes:
        try:
            if nx.find_cycle(G, node) != None:
                n_feedback_nodes = n_feedback_nodes + 1
        except NetworkXNoCycle:
            pass

    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()

    # print(str(n_feedback_edges) + " " + str(n_feedback_nodes))
    # print(str(n_edges) + " " + str(n_nodes))
    return (n_feedback_edges + n_feedback_nodes) / (n_edges + n_nodes), n_feedback_edges, n_feedback_nodes


def causal_complexity(G):
    """
    Calculates the causal complexity of a networkx graph. G should be the ground truth.

    The definition is: C = M * (1 + D), where M = cyclomatic complexity and D = feedback density.
    :param G:
    :return:
    """

    M = cyclomatic_complexity(G)
    D = feedback_density(G)

    # print(str(M))
    # print(str(D))
    CC = (M * (1. + D[0]))

    di_intricacy = {"nodes":G.number_of_nodes(),
                    "edges":G.number_of_edges(),
                    "cyclomatic":M,
                    "feedback":D[0],
                    "feedbackedges":D[1],
                    "feedbacknodes":D[2],
                    "causal":CC
                    }
    return di_intricacy

def hierarchy(G):
    """
    Calculates the hierarchiness of a social network.
    :param G:
    :return:
    """

    if G.number_of_edges() == 0:
        print("WARNING: Social network to compute GRC over has no edges!")
        return 0.0
    else:

        nu_grc = nx.global_reaching_centrality(G)

        if nu_grc == 0.0:
            # It may be because the graph is supposed to be undirected (i.e., reciprocal edges). So check that.
            if nx.is_directed(G):
                nu_num_edges = nx.number_of_edges(G)
                G_un = G.to_undirected(reciprocal=True)
                if nx.number_of_edges(G_un) == .5*nu_num_edges:
                    # This seems likely that the graph was supposed to be undirected.
                    print("WARNING: GRC was calculated on the undirected version of the graph!")
                    return nx.global_reaching_centrality(G_un)

        return nu_grc

def information_theoretic_complexity_w_LZMA_too_sensitive(ts_data, time_axis=0):
    """
    Compute normaized information distance between past and future for
    each separation point in the time-series data (which is a numpy array).
    : return nid_average: average normalized information distance
    """
        
    # handle NaNs
    if np.isnan(ts_data).any():
        ts_data = ts_data.astype('float')

    lzc = lzma.LZMACompressor()
    out = lzc.flush()
    base = sys.getsizeof(out)
    NS = ts_data.shape[time_axis] - 1
    nid_values = np.zeros((NS))
    for s in range(NS):
        lzc = lzma.LZMACompressor()
        outA = lzc.compress(np.take(ts_data, s, axis=time_axis))
        outB = lzc.flush()
        out = b"".join([outA, outB])
        past = sys.getsizeof(out) - base
        lzc = lzma.LZMACompressor()
        outA = lzc.compress(np.take(ts_data, s+1, axis=time_axis))
        outB = lzc.flush()
        out = b"".join([outA, outB])
        future = sys.getsizeof(out) - base
        lzc = lzma.LZMACompressor()
        outA = lzc.compress(np.take(ts_data, s, axis=time_axis))
        outB = lzc.compress(np.take(ts_data, s+1, axis=time_axis))
        outC = lzc.flush()
        out = b"".join([outA, outB, outC])
        future_given_past = sys.getsizeof(out) - past
        lzc = lzma.LZMACompressor()
        outA = lzc.compress(np.take(ts_data, s+1, axis=time_axis))
        outB = lzc.compress(np.take(ts_data, s, axis=time_axis))
        outC = lzc.flush()
        out = b"".join([outA, outB, outC])
        past_given_future = sys.getsizeof(out) - future
        nid_values[s] = np.maximum(past_given_future, future_given_past) / np.maximum(past, future)

    nid_average = np.mean(nid_values)

    return nid_average


def information_theoretic_complexity_phase1(ts_data, time_axis=0):
    """
    Compute normaized compression distance between past and future for
    each separation point in the time-series data (which is a numpy array).
    : return ncd_average: average normalized compression distance
    """
    print("information_theoretic_complexity_phase1")

    # convert to consistent format
    ts_data = ts_data.astype('float')

    lzc = lzma.LZMACompressor()
    out = lzc.flush()
    base = sys.getsizeof(out)
    lzc = lzma.LZMACompressor()
    outA = lzc.compress(ts_data)
    outB = lzc.flush()
    out = b"".join([outA, outB])
    past_and_future = sys.getsizeof(out) - base
    NS = ts_data.shape[time_axis] - 1
    ncd_values = np.zeros((NS))
    for s in range(NS):
        lzc = lzma.LZMACompressor()
        outA = lzc.compress(np.take(ts_data, np.arange(0, s+1), axis=time_axis))
        outB = lzc.flush()
        out = b"".join([outA, outB])
        past = sys.getsizeof(out) - base
        lzc = lzma.LZMACompressor()
        outA = lzc.compress(np.take(ts_data, np.arange(s+1, NS), axis=time_axis))
        outB = lzc.flush()
        out = b"".join([outA, outB])
        future = sys.getsizeof(out) - base
        ncd_values[s] = (past_and_future - np.minimum(past, future)) / np.maximum(past, future)

    # to match the performer's guide, we are using L2-norm instead of mean here
    #ncd_average = np.mean(ncd_values)
    ncd_average = np.linalg.norm(ncd_values) / np.sqrt(NS)

    return ncd_average

def compress(ts_data):
    lzc = lzma.LZMACompressor()
    outA = lzc.compress(ts_data)
    outB = lzc.flush()
    out = b"".join([outA, outB])
    return out

def get_compressed_size(ts_data, base):
    out = compress(ts_data)
    return len(out) - base

# plot NCD vs split point for multiple time series and corresponding 'baseline'
# (baseline is NCD values on all-0s of same size as original time series)
def plot_NCD_pairs(ls_ncd_data, ls_ncd_baseline):
    if len(ls_ncd_data) != len(ls_ncd_baseline):
        print("ERROR:  #baseline series not the same as #data series")
        return
    n = len(ls_ncd_data)
    color=iter(plt.cm.rainbow(np.linspace(0,1,n)))
    plt.figure()
    plt.title('NCDs for {} instances and baselines'.format(n))
    for i in range(n):
        c=next(color)
        plt.plot(ls_ncd_data[i], c=c,  label=i)
        plt.plot(ls_ncd_baseline[i], '.', c=c)
    plt.legend()
    plt.show

# plot NCD vs. split point for multiple time series
def plot_NCD_series(ls_ncd_values):
    n=len(ls_ncd_values)
    plt.figure()
    plt.title('NCD values vs split points for {} instances'.format(n))
    color=iter(plt.cm.rainbow(np.linspace(0,1,n)))
    for i in range(n):
        c=next(color)
        plt.plot(ls_ncd_values[i], c=c, label=i)
    plt.legend()
    plt.show

def plot_performer_data(ts_data):
    # data in this case is assumed to have time as first axis
    # performer data given as strings
    ts_data = ts_data.astype('float')
    numD = ts_data.shape[1]
    print("plot_data numD: {}".format(numD))
    numRows = math.ceil(numD/2)
    plt.figure()
    for i in range(numD):
        plt.subplot(numRows, 2, i+1)
        plt.plot(ts_data[:,i], '.')
    plt.show()

def get_all_NCD_values(ts_data, time_axis=0):
    # similar to information_theoretic_complexity_phase3 but
    # returns all NCD values rather than average
    # mostly used for testing
    # can't assume that this method will stay in sync with the
    # current IT metric method
    print("get_all_NCD_values")
    # convert to consistent format
    ts_data = ts_data.astype('float')

    splitter = FullSplitter(ts_data)
    lzme = LZMAEstimator(splitter)
    lzme.estimate()
    return lzme.ncd_values

def get_estimator(ts_data, time_axis=0,
                  splitClass=QSubsetter,
                  estClass=LZMAEstimator):
    splitter = splitClass(ts_data, time_axis)
    estimator = estClass(splitter)
    return estimator.estimate

def information_theoretic_complexity_phase3(ts_data, time_axis=0,
                                            splitClass=QSubsetter,
                                            estClass=LZMAEstimator):
    """
    Compute normalized compression distance between past and future for
    select separation points in the time-series data (which is a numpy array).
    : return ncd_average: average normalized compression distance
    Compared to phase2 version, this:
        1) uses Splitter class to slice the data in different ways;
            we can use each separation point as before,
            or use quartiles or other schemes
        2) uses Estimator class to actually do the main calculation;
            LZMAEstimator does what phase2 code did;
            other Estimator implementations are possible
    """
    print("information_theoretic_complexity_phase3")
    # convert to consistent format
    ts_data = ts_data.astype('float')

    estimate = get_estimator(ts_data, time_axis, splitClass, estClass)
    return estimate()

def information_theoretic_complexity_phase2(ts_data, time_axis=0):
    """
    Compute normalized compression distance between past and future for
    each separation point in the time-series data (which is a numpy array).
    : return ncd_average: average normalized compression distance
    Compared to phase1 version, this:
        1) fixes an off-by-one error with NS
        2) uses mean instead of L2 norm
    Compared to original phase2 version, this uses a separate method to get compressed sizes
    """
    print("information_theoretic_complexity_phase2_decomposed")
    # convert to consistent format
    ts_data = ts_data.astype('float')

    base = get_compressed_size(b"", 0)

    past_and_future = get_compressed_size(ts_data, base)
    nu_samples = ts_data.shape[time_axis]
    nu_splits = nu_samples - 1
    ncd_values = np.zeros((nu_splits))
    for s in range(nu_splits):
        past = get_compressed_size(np.take(ts_data, np.arange(0, s+1), axis=time_axis), base)
        future = get_compressed_size(np.take(ts_data, np.arange(s+1, nu_samples), axis=time_axis), base)
        ncd_values[s] = (past_and_future - np.minimum(past, future)) / np.maximum(past, future)

    ncd_average = np.mean(ncd_values)
    return ncd_average

def information_theoretic_complexity(ts_data, time_axis=0):
    """
    Trying to make it easier to change which method to use.
    """
    print("\nstarting IT complexity calculation; data size:")
    print(ts_data.shape)
    return information_theoretic_complexity_phase3(ts_data, time_axis)


def information_theoretic_complexity_w_GZIP(ts_data, time_axis=0):
    """
    Compute normaized compression distance between past and future for
    each separation point in the time-series data (which is a numpy array).
    : return ncd_average: average normalized compression distance
    """
        
    # handle NaNs
    ts_data = ts_data.astype(float)

    # if np.isnan(ts_data).any():
    #     ts_data = ts_data.astype('float')

    past_and_future = sys.getsizeof(gzip.compress(ts_data))
    NS = ts_data.shape[time_axis] - 1
    ncd_values = np.zeros((NS))
    for s in range(NS):
        past = sys.getsizeof(gzip.compress(np.take(ts_data, np.arange(0, s+1), axis=time_axis)))
        future = sys.getsizeof(gzip.compress(np.take(ts_data, np.arange(s+1, NS), axis=time_axis)))
        ncd_values[s] = (past_and_future - np.minimum(past, future)) / np.maximum(past, future)

    # to match the performer's guide, we are using L2-norm instead of mean here
    #ncd_average = np.mean(ncd_values)
    ncd_average = np.linalg.norm(ncd_values) / np.sqrt(NS)

    return ncd_average


#
# def flexibility_range(ch_cdf_location,
#                              ch_catalog_location,
#                              ch_parameter_mapping_file,
#                              ls_agents,
#                              ls_timesteps,
#                              ls_vars):
#     """
#     This method calculates the "range" part of the flexbility metric.
#     :param ch_cdf_location:
#     :param ch_catalog_location:
#     :param ch_parameter_mapping_file: Maps each parameter that can significantly increase complexity to a set of instances.
#     :param ls_agents:
#     :param ls_timesteps:
#     :param ls_vars:
#     :return:
#     """
#
#     df_parameter_mapping = pd.read_csv(ch_parameter_mapping_file, sep="\t")
#
#     ls_final_dataset = []
#     for row in df_parameter_mapping.itertuples():
#
#         # Loop through all the rows.
#         ch_param_name = row.ParameterName
#         ch_instance = row.Instance
#
#         # Load and extract the ts.
#         np_array_data = cdfio.extract_ts(ch_cdf_path=ch_cdf_location,
#                                      ch_catalog_loc=ch_catalog_location,
#                                      ls_agents=ls_agents,
#                                      ls_timesteps=ls_timesteps,
#                                      di_vars=ls_vars,
#                                          nu_instance=int(ch_instance))
#
#         # Compute complexity.
#         nu_info_comp = information_theoretic_complexity(np_array_data)
#
#         # Generate the social network
#
#         nx_soc_net = cdfio.generate_social_network(ch_cdf_path=ch_cdf_location,
#                                                    ch_catalog_loc=ch_catalog_location,
#                                                    ls_timesteps=ls_timesteps)
#         # Compute hierarchy complexity.
#         nu_hierarchy_comp = hierarchy(nx_soc_net)
#
#         ls_final_dataset.append({'ParameterName':ch_param_name,
#                                  'Instance':ch_instance,
#                                  'IT_Complexity':nu_info_comp,
#                                  'SO_Complexity':nu_hierarchy_comp})
#
#     df_final_data = pd.DataFrame(ls_final_dataset)
#
#     # Now let's calculate the ranges.
#
#     ls_return_data = []
#     for name, grp in df_final_data.groupby('ParameterName'):
#
#         di_final = {
#             'ParameterName': name,
#             # 'NumInstances':str(grp.size()),
#             'IT_Complexity_Range':grp['IT_Complexity'].max() -grp['IT_Complexity'].min(),
#             'SO_Complexity_Range': grp['SO_Complexity'].max() - grp['SO_Complexity'].min()
#         }
#         ls_return_data.append(di_final)
#
#
#
#     return(pd.DataFrame(ls_return_data))


def plausibility2(ch_cdf_location,
                             ch_catalog_location,
                             ls_instances,
                             ls_agents,
                             ls_timesteps,
                             di_vars):
    """
    Plausibility re-done addressing multiple variables and multiple entities per variable.

    We calculate the empirical range based on the range of a variable across all entities and runs.
    :param ch_cdf_location:
    :param ch_catalog_location:
    :param ls_instances:
    :param ls_agents:
    :param ls_timesteps:
    :param di_vars:
    :return:
    """

    # process plausibility CDF if it hasn't been processed before.
    ch_cdf_basename = os.path.basename(ch_cdf_location)
    o_path_to_catalog_dir = os.path.join(ch_catalog_location, ch_cdf_basename)

    ls_all_results = []
    di_variable_info = {}

    # Loop through all variable/entity pairs.
    for (ch_variable_name, o_var_values) in di_vars.items():

        di_var_loc = json.load(open(os.path.join(o_path_to_catalog_dir, "varLocations.txt")))
        ch_variable_file = ""

        # We need to identify the file that this variable is from.
        if ch_variable_name in di_var_loc['SST']:
            ch_variable_file = "SummaryStatisticsDataTable.tsv"
        else:
            ch_variable_file = "RunDataTable.tsv"

        # Get the range of the variable first.
        # We need to pass the entities the variable could be subset on because that will determine the
        # bin size (we need to find the smallest range of the variable across all pairs of variables/entities.
        di_emp_min_max = cdfio.find_empirical_min_max_range_and_bin(ch_cdf_location,
                                                                    ch_catalog_location=ch_catalog_location,
                                                                    ls_instances=ls_instances,
                                                                    o_entities = o_var_values,
                                                                    ch_plausibility_variable=ch_variable_name,
                                                                    nu_bins=None)
        di_one_var = {
            "minQOI": di_emp_min_max["min"],
            "maxQOI": di_emp_min_max["max"],
            "numBins": di_emp_min_max["num_bins"]}

        di_variable_info[ch_variable_name] = di_one_var

        for nu_instance_id in ls_instances:
            # Load mapping of variables to files.


            # Load the data from the variable
            ch_path = os.path.join(ch_cdf_location,
                      "Instances", "Instance"+str(nu_instance_id), "Runs", "run-0", ch_variable_file)
            df_run_data = pd.read_csv(ch_path, sep="\t")
            df_subset = df_run_data[df_run_data["VariableName"] == ch_variable_name]

            # Now check if we have multiple entities or not.
            if isinstance(o_var_values, str):
                # only single value. Should be global.
                if o_var_values == "global":
                    di_results = calculate_plausibility_for_one_var_one_entity(df_subset,
                                                                               ch_variable_name,
                                                                               di_emp_min_max['bins'])
                    di_results["Instance"] = nu_instance_id

                    ls_all_results.append(dict(di_results))
                else:
                    raise(RuntimeError("Plausibility variable: "+ch_variable_name+" is specified as: "+o_var_values))
            elif isinstance(o_var_values, list):

                # Specifying a list of entities.
                for ch_entity in o_var_values:
                    di_results = calculate_plausibility_for_one_var_one_entity(df_data=df_subset,
                                                                               ch_var=ch_variable_name,
                                                                               np_bins=di_emp_min_max['bins'],
                                                                               ch_entity=ch_entity)
                    di_results["Instance"] = nu_instance_id
                    ls_all_results.append(dict(di_results))

            df_foo = pd.DataFrame(ls_all_results)

    print(df_foo)
    return {"data":df_foo,
            "varinfo":di_variable_info}


def calculate_plausibility_for_one_var_one_entity(df_data,
                                                  ch_var,
                                                  np_bins,
                                                  ch_entity = None):
    """
    Calculate plausibility values for a single variable and a single entity.
    """

    di_return = {}
    df_data_subset = df_data[df_data["VariableName"]==ch_var]

    # Subset if an entity was specified. Otherwise just use all the values.
    if ch_entity:
        df_data_subset = df_data_subset[df_data_subset["EntityIdx"] == ch_entity]

    np_vec = df_data_subset["Value"].astype('float')
    print(np_vec)
    # np_vec = s_subset_value.values()
    np_dist = generate_distribtion_from_continous_vector(np_vec, bins=np_bins)
    print(np_dist)
    # print("Distribution: "+str(np_dist))
    # And now calculate the entropy
    nu_entropy = sp.stats.entropy(np_dist, base=2)
    nu_variance = np.var(np_vec)

    di_return = {'Variable':ch_var,
                 'Entity':ch_entity,
                 'Entropy': nu_entropy,
                 'Variance': nu_variance}

    return di_return


#
# def plausibility(ch_cdf_location,
#                              ch_catalog_location,
#                              ls_instances,
#                              ls_agents,
#                              ls_timesteps,
#                              di_vars):
#     """
#     Calculates the metrics for plausibility. Assume only 1 run per instance.
#
#     We need a method to extract a summary statistics ts.
#     :return:
#     """
#
#     # Check if CDF has been processed.
#
#     cdfio.check_process_load_cdf(ch_cdf_location, ch_catalog_location)
#     # We need to identify the range for the plausibility variable.
#     # First get the variable name. There should only be one variable.
#     ch_variable_name = list(di_vars.keys())[0]
#
#     # Then identify which file the variable is from (either RunDataTable.tsv or SummaryStatisticsTable.tsv)
#
#     # Load the VarDefTable within the CDF.
#     pd_var = pd.read_csv(os.path.join(ch_cdf_location, "SimulationDefinition","VariableDefTable.tsv"), sep="\t")
# #    ch_range = pd_var[pd_var.Name == ch_variable_name].Values[0]
#
#     di_emp_min_max = cdfio.find_empirical_min_max_range_and_bin(ch_cdf_location,
#                                                          ch_catalog_location=ch_catalog_location,
#                                                          ls_instances=ls_instances,
#                                                          ch_plausibility_variable=ch_variable_name,
#                                                                 nu_bins = 10)
#     np_bins = di_emp_min_max["bins"]
#
#     # Now parse and create a set of bins.
#     # np_bins = cdfio.parse_variable_range_and_divide(ch_range,10)
#
#     # For a set of instances calculate the entropy and the variance for a QOI.
#     ls_out_data = []
#     for nu_one_instance in ls_instances:
#         # Load and extract the QOI ts.
#         np_array_data = cdfio.extract_ts(ch_cdf_path=ch_cdf_location,
#                                          ch_catalog_loc=ch_catalog_location,
#                                          ls_agents=ls_agents,
#                                          ls_timesteps=ls_timesteps,
#                                          di_vars=di_vars, nu_instance=int(nu_one_instance))
#
#         # print("Data for plausibility")
#         # print(np_array_data)
#
#         # Get the column. The timeseries should have only a single column.
#         np_vec = np_array_data[0]
#         np_vec = np_vec.astype(float)
#
#         # Generate a distribution
#         np_dist = generate_distribtion_from_continous_vector(np_vec, bins=np_bins)
#
#         # print("Distribution: "+str(np_dist))
#         # And now calculate the entropy
#         nu_entropy = sp.stats.entropy(np_dist, base=2)
#         nu_variance = np.var(np_vec)
#         ls_out_data.append({'Instance': nu_one_instance,
#                             'Entropy': nu_entropy,
#                             'Variance': nu_variance})
#
#
#     return {"data":pd.DataFrame(ls_out_data),
#             "minQOI":di_emp_min_max["min"],
#             "maxQOI":di_emp_min_max["max"],
#             "numBins":di_emp_min_max["num_bins"]}


def generate_distribtion_from_continous_vector(np_ar, bins):
    """
    Calculates a discrete distribution from a continous vector.
    :param np_ar:
    :param bins: Be default not specified, but if specified then use these bins. otherwise cut into 10.
    :return: Series or numpy array of the probability values.
    """

    np_copy = np_ar.copy()
    np_copy = np_copy.astype(float)
    np_copy = np_copy[~np.isnan(np_copy)]
    s_0 = pd.Series(pd.cut(np_copy, bins=bins, labels=False))

    # Drop the na values.
    s_1 = s_0.value_counts()
    df_counts = pd.DataFrame({'bins':s_1.index, 'counts':s_1.values})
    df_empty = pd.DataFrame({'bins':range(0,len(bins)), 'counts': 0})
    df_dist = df_empty.merge(df_counts, on='bins', how='left').fillna(0)
    df_dist['Prob'] = df_dist.counts_y / sum(df_dist.counts_y)

    return df_dist['Prob'].values

if __name__ == '__main__':
    print("Hollow world")

