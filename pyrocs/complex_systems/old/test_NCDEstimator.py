# test_NCDEstimator

import numpy as np

from DataSplitter import DataSplitter, FullSplitter, QSplitter, QSubsetter
from NCDEstimator import NCDEstimator

NS = 10
numD = 1
ts_data = np.zeros((NS, numD))
for j in range(numD):
    ts_data[:,j] = range(NS)
ts_data = ts_data.astype('float')

def test_default_splitter():
    splitter = DataSplitter(ts_data)
    assert splitter.nu_splits == 1
    past, future = splitter.nextData()
    assert len(past) == len(future)
    past, future = splitter.nextData()
    assert future == None

def test_full_splitter():
    splitter = FullSplitter(ts_data)
    assert splitter.nu_splits == NS-1
    for i in range(splitter.nu_splits):
        print('full test, loop {}'.format(i))
        past, future = splitter.nextData()
        assert len(past) == i+1
        assert len(future) == NS-(i+1)
    past, future = splitter.nextData()
    assert future == None

def test_qsplitter():
    splitter = QSplitter(ts_data)
    assert splitter.nu_splits == 3
    # for data length of 10
    exp_past_len = [2, 5, 7]
    exp_future_len = [8, 5, 3]
    for i in range(splitter.nu_splits):
        past, future = splitter.nextData()
        assert len(past) == exp_past_len[i]
        assert len(future) == exp_future_len[i]
    past, future = splitter.nextData()
    assert future == None

def test_qsubsetter():
    splitter = QSubsetter(ts_data)
    assert splitter.nu_splits == 3
    # for data length of 10
    # same split points used as QSplitter, but past/future only cover
    # two subsets rather than entire sequence
    exp_past_len = [2, 3, 2]
    exp_future_len = [3, 2, 3]
    for i in range(splitter.nu_splits):
        past, future = splitter.nextData()
        assert len(past) == exp_past_len[i]
        assert len(future) == exp_future_len[i]
    past, future = splitter.nextData()
    assert future == None

def test_estimator():
    # NCDEstimator uses a default get_compressed_size method that just uses 
    # len(data)  (for real NCD estimates, need to use subsets of NCDEstimate
    # that use proper compressors)
    # if data subsets always cover the whole data, NCD should always be 1
    splitter = DataSplitter(ts_data)
    estimator = NCDEstimator(splitter)
    assert estimator.estimate() == 1
    splitter = FullSplitter(ts_data)
    estimator = NCDEstimator(splitter)
    assert estimator.estimate() == 1
    splitter = QSplitter(ts_data)
    estimator = NCDEstimator(splitter)
    assert estimator.estimate() == 1
    # and even with QSubsetter, where any 2 subsets don't cover all of ts_data,
    # NCDEstimator should be correctly comparing "compressed" sizes of 
    # the individual subsets to their concatenation, so we should still get 1
    splitter = QSubsetter(ts_data)
    estimator = NCDEstimator(splitter)
    assert estimator.estimate() == 1
