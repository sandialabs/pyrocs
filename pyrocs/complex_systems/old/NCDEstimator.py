# -*- coding: utf-8 -*-
"""
Base class for using NCD to estimate forecast complexity.
Requires a DataSplitter to provide data subsets to compare.

This implementation doesn't actually use a compressor; it just returns
the length of the data as the 'compressed' size.  It should be subclassed with 
a specific compressor implementation.
This version is nice for testing though, because the 'NCD' calculated by 
this implementation should always be 1.
"""

import numpy as np
from DataSplitter import QSubsetter

class NCDEstimator():

    def __init__(self, splitter):
        self.splitter = splitter
        self.ncd_values = np.zeros(splitter.nu_splits)

    def get_compressed_size(self, data):
        return len(data)

    def NCDfromLen(self, len1, len2, len12):
        return (len12 - np.minimum(len1, len2)) / np.maximum(len1, len2)

    def NCDfromData(self, data1, data2, len12=None):
        if len12 is None:
            len12=self.get_compressed_size(np.concatenate([data1, data2]))
        return self.NCDfromLen(self.get_compressed_size(data1), 
                               self.get_compressed_size(data2), 
                               len12)

    def getFullSize(self):
        # if past+future always = ts_data, we can cache the compressed size of
        # ts_data and reuse it
        # but if we're using subsets that don't cover the whole time series,
        # then the size of subset1+subset2 has to be recalculated each time
        if isinstance(self.splitter, QSubsetter):
            self.len_full_data = None
        else:
            self.len_full_data = self.get_compressed_size(self.splitter.ts_data)
        
    def estimate(self):
        len_full_data = self.getFullSize()
        past, future = self.splitter.nextData()
        index = 0
        while future is not None:
            ncd = self.NCDfromData(past, future, len_full_data)
            if ncd>1:
                print("NCD > 1 at split point {}".format(index))
            self.ncd_values[index] = ncd
#            print('estimate {}: {}'.format(index, self.ncd_values[index]))
            past, future = self.splitter.nextData()
            index = index+1
        return np.mean(self.ncd_values)
