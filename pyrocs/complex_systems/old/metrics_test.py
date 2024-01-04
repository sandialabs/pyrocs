import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import itertools

import metrics
import dataIO.cdfIO as cdfio

from LZMAEstimator import LZMAEstimator
from DataSplitter import QSplitter, QSubsetter


def entropy_est_1D(data):
    # create histogram
    # TODO use metrics.generate_distribtion_from_continous_vector
    # scipy entropy
    di_values = {}
    NS = data.shape[0]
    for i in range(NS):
        value = data[i]
        if value in di_values.keys():
            di_values[value] = di_values[value] + 1
        else:
            di_values[value] = 1
            
    entropy = 0
    for value in di_values:
        freq = di_values[value]/NS
        entropy += -freq*math.log(freq,2) 

    return entropy
    

def get_cdf_entropy(ch_cdf_path, ch_filename, ls_instances, ls_runs):
    for nu_instance in ls_instances:
        for nu_run in ls_runs:
            df = cdfio.load_run_data(ch_cdf_path, ch_filename, nu_instance, nu_run)
            data = df.Value
            # TODO - convert format?
            entropy = entropy_est_1D(data)
            print("Instance {}, Run {}; entropy: {}".format(nu_instance, nu_run, entropy))
    
def plot_data(ts_data, ch_name=""):
    # TODO duplicates metrics.py plot_performer_data except for name/title
    # may not need next line?
    ts_data = ts_data.astype('float')
    numD = ts_data.shape[1]
    numRows = math.ceil(numD/2)
    plt.figure()

    for i in range(ts_data.shape[1]):
        plt.subplot(numRows, 2, i+1)
        plt.plot(ts_data[:,i])
    plt.suptitle(ch_name)
    plt.show()
    
def create_periodic_sequence(nu_phases, nu_len):
    # just a repeating sequence of nu_phases numbers
    # p=2: 01010101
    # p=3: 012012
    # etc.
    seq = np.zeros(nu_len)
    for i in range(nu_len):
        seq[i] = i % nu_phases
    return seq

def create_test_suite(NS=1000, numD=1):
    di_tests = {}
    di_tests['zeros'] = np.zeros((NS, numD))
    di_tests['ones'] = np.ones((NS, numD))

    period2 = create_periodic_sequence(2, NS)
    ts_data = np.zeros((NS, numD))
    for j in range(numD):
        ts_data[:,j] = period2
    di_tests['period2'] = ts_data

    period3 = create_periodic_sequence(3, NS)
    ts_data = np.zeros((NS, numD))
    for j in range(numD):
        ts_data[:,j] = period3
    di_tests['period3'] = ts_data

    # linear values
    ts_data = np.zeros((NS, numD))
    for j in range(numD):
        ts_data[:,j] = range(NS)
    di_tests['linear'] = ts_data

    # reverse linear values
    ts_data = np.zeros((NS, numD))
    for j in range(numD):
        if j%2==0:
            ts_data[:,j] = range(NS)
        else:
            ts_data[:,j] = range(NS, 0, -1)
    di_tests['reverse linear'] = ts_data

    # random values
    random.seed(29411492)
    ts_data = np.zeros((NS, numD))
    for i in range(NS):
        for j in range(numD):
            ts_data[i,j] = random.randrange(0, NS)
    di_tests['random'] = ts_data

    return di_tests


def get_compressed_size(ts_data):
    lzme = LZMAEstimator.LZMAEstimator()
    return lzme.get_compressed_size(ts_data)


def idempotency(x, f=get_compressed_size):
    # for normal compressor, should have
    # C(xx) = C(x)
    x = x.astype('float')   # first thing IT metric code does
    tol = 1200*math.log(len(x.tobytes()))
    cx = f(x)
    cxx = f(np.concatenate([x, x]))
    print("expect C(xx)=C(x), within tolerance {}".format(tol))
    print("C(xx): {}; C(x): {}; C(xx)~=C(x): {}".format(cxx, cx, abs(cxx-cx)<tol))
    return [tol, cx, cxx, abs(cxx-cx)]

def check_idempotency_by_size(f=get_compressed_size):
    print('\n================================================')
    print("Idempotency tests\n")
    sizes = [2, 10, 100, 200, 500, 1000]
    ls_results = []
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            NS = sizes[i]
            numD = sizes[j]
            total = NS * numD
            x = np.zeros((NS, numD))
            results = idempotency(x, f)
            di_results = {"NS":NS, 'numD':numD, 'total':total, 'tol':results[0],
                          "cx":results[1], 'cxx':results[2], 'diff':results[3]}
            ls_results.append(dict(di_results))

    df = pd.DataFrame(ls_results)
    print(df)

    plt.figure()
    plt.title('abs(cxx-cx) vs NS')
    plt.plot(df['NS'], df['diff'], 'bo')
    plt.plot(df['NS'], df['tol'], 'ro')
    plt.show()

    plt.figure()
    plt.title('abs(cxx-cx) vs numD')
    plt.plot(df['numD'], df['diff'], 'bo')
    plt.plot(df['numD'], df['tol'], 'ro')
    plt.show()

    plt.figure()
    plt.title('abs(cxx-cx) vs total size')
    plt.plot(df['total'], df['diff'], 'bo')
    plt.plot(df['total'], df['tol'], 'ro')
    plt.show()

def normal_compressor_tests(x, y, z, f=get_compressed_size):
    # tests for a "normal compressor" from Cilibrasi & Vitanyi 05
    # supposed to hold *up to an additive O(log n) term*
    # C(empty string) = 0
    # C(xx) = C(x)
    # C(xy) >= C(x)
    # C(xy) = C(yx)
    # C(xy) + C(z) <= C(xz) + C(yz)
    print('\n================================================')
    print('Normal Compressor Tests\n')
    print("size of compressed empty 2D array: {}".format(f(np.ndarray((0,0),dtype=object))))

    n = max(len(x), len(y), len(z))
    print("normal compressor max input length:  {}".format(n))
    nb = max(len(x.tobytes()), len(y.tobytes()), len(z.tobytes()))
    print("but what matters is max length of bytes object: {}".format(nb))
    tol = math.log(nb)
    print("tolerance O({})".format(tol))
    
    cx = f(x)

    idempotency(x, f)
    
    cxy = f(np.concatenate([x, y]))
    print("expect C(xy)>=C(x)")
    print("C(xy): {}; C(x): {}; C(xy)>C(x)?: {}".format(cxy, cx, cxy>cx))
    
    cyx = f(np.concatenate([y, x]))
    print("expect C(xy)=C(yx), within tolerance")
    print("C(xy): {}; C(yx): {}; C(xy)-C(yx): {}".format(cxy, cyx, abs(cxy-cyx)))
    
    cz = f(z)
    cxz = f(np.concatenate([x, z]))
    cyz = f(np.concatenate([y, z]))
    print("expect C(xy) + C(z) <= C(xz) + C(yz)")
    print("C(xy): {}; C(z): {}; C(xz): {}; C(yz): {}; [C(xy)+C(z)]<=[C(xz)+C(yz)]: {}".format(
            cxy, cz, cxz, cyz, (cxy+cz)<=(cxz+cyz)))

def check_P3_IT_trends(di_tests, splitter, estimator):
    print('\n================================================')
    print("P3 IT complexity values for various input data\n")
    di_results = {}
    for key in di_tests:
        result = metrics.information_theoretic_complexity_phase3(di_tests[key], 
                                                                 splitClass=splitter, 
                                                                 estClass=estimator)
        di_results[key] = result
        print('{}: {}'.format(key, result))
    return di_results

def check_IT_trends(di_tests, f=metrics.information_theoretic_complexity):
    # "tests" of IT metric - looking for "reasonable" trends
    print('\n================================================')
    print("IT complexity values for various input data\n")
    di_results = {}
    for key in di_tests:
        result = f(di_tests[key])
        di_results[key] = result
        print('{}: {}'.format(key, result))
    return di_results

def compare_IT_trends():
    # test sizes of interest to use in creating test suite:
    # first # is NS (#timesteps/samples); second is numD (#dimensions/variables)
    # (100,1) start small when changing code...
    # (1000,1) kind of a standard 1D set; takes a few minutes for phase2 code
    # (100,2) - close to USC P1 data size (smallest P1 data set)
    # (350,6) - close to GMU P1 size; took a few minutes for phase2 code
    # haven't tried these:
    # (1000,500) - largest P1 dataset
    # (150,1225) - largest #variables from P1
    ls_data_sizes = [(100,1),
                     (100,2),
                     (1000,1),
                     (350,6)
                     ]

    for sizes in ls_data_sizes:
        di_tests = create_test_suite(sizes[0], sizes[1])
        di_results = {}

        # plots help make sure we're using the inputs we think we are,
        # but comment out plots when using >10 variables
        for key in di_tests:
            plot_data(di_tests[key], key)

#        di_results['P2'] = check_IT_trends(di_tests, f=metrics.information_theoretic_complexity_phase2)
        di_results['P3'] = check_IT_trends(di_tests, f=metrics.information_theoretic_complexity)
        di_results['P3QSplit'] = check_P3_IT_trends(di_tests, QSplitter, LZMAEstimator)
        di_results['P3QSubset'] = check_P3_IT_trends(di_tests, QSubsetter, LZMAEstimator)

        plt.figure()
        marker = itertools.cycle(('o', 's', 'v', '^', '<', '>'))
        for key in di_results:
            plt.plot(di_results[key].values(), label = key, marker = next(marker), linestyle='')
        plt.xticks(np.arange(len(di_tests.keys())), di_tests.keys())
        plt.legend()
        plt.title('{} timesteps; {} variables'.format(sizes[0], sizes[1]))
        plt.show()

if __name__ == '__main__':

#    check_idempotency_by_size()

#    NS = 100
#    numD = 1
#    di_tests = create_test_suite(NS, numD)
#    normal_compressor_tests(di_tests['allones'], di_tests['period2'], di_tests['period3'])
#    normal_compressor_tests(di_tests['multiDallones'], di_tests['multiDperiod2'], di_tests['multiDperiod3'])

    compare_IT_trends()
