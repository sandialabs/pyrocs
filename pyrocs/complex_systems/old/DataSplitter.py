"""
Classes for various ways of slicing our time series.
They are instantiated with the full time series; each call to 
nextData should get the next "past" and "future" parts of the data, 
until the data has been exhausted.
Definitions of where the split between "past" and "future" is, and how big
each subset is, vary between subclasses.
"""

import numpy as np
import math

class DataSplitter():
    """
    Default implementation mostly used for testing
    Just splits data in half
    """
    def __init__(self, ts_data, time_axis=0):
        self.ts_data = ts_data
        self.time_axis = time_axis
        self.index = 0
        self.nu_samples = self.ts_data.shape[self.time_axis]
        self.nu_splits = 1
        
    def nextSplitPoint(self):
        return math.floor(self.nu_samples/2)
    
    def nextData(self):
        if self.index>=self.nu_splits:
            return None, None
        k = self.nextSplitPoint()
        item1 = np.take(self.ts_data, np.arange(0, k), axis=self.time_axis)
        item2 = np.take(self.ts_data, np.arange(k, self.nu_samples), axis=self.time_axis)
        self.index = self.index+1
        return item1, item2

class FullSplitter(DataSplitter):
    """
    Splits data at every possible split point
    """
    def __init__(self, ts_data, time_axis=0):
        super().__init__(ts_data, time_axis)
        self.nu_splits = self.nu_samples - 1
        
    def nextSplitPoint(self):
        return self.index + 1
        
        
class QSplitter(DataSplitter):
    """
    Chooses split points roughly at every quarter
    Past+future subsets cover the entire time series (unlike QSubsetter)
    """
    def __init__(self, ts_data, time_axis=0):
        super().__init__(ts_data, time_axis)
        self.nu_splits = 3
        self.ls_splits = self.get_quartile_splits()
        
    def get_quartile_splits(self):
        half = math.floor(self.nu_samples/2)
        quarter = math.floor(half/2)
        thirdq = math.floor((self.nu_samples-half)/2)

        ls_splits = []
        ls_splits.append(quarter)
        ls_splits.append(half)
        ls_splits.append(half+thirdq)
        return ls_splits

    def nextSplitPoint(self):
        return self.ls_splits[self.index]
    
class QSubsetter(QSplitter):
    """
    Chooses split points roughly at every quarter
    Past+future subsets cover only the data between split points
    (unlike QSplitter)
    """
    def __init__(self, ts_data, time_axis=0):
        super().__init__(ts_data, time_axis)
        self.ls_splits = [0] + self.ls_splits + [self.nu_samples]
        
    def nextData(self):
        if self.index>=self.nu_splits:
            return None, None
        start = self.ls_splits[self.index]
        k = self.ls_splits[self.index+1]
        stop = self.ls_splits[self.index+2]
        item1 = np.take(self.ts_data, np.arange(start, k), axis=self.time_axis)
        item2 = np.take(self.ts_data, np.arange(k, stop), axis=self.time_axis)
        self.index = self.index+1
        return item1, item2
           
    