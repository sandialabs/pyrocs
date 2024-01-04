# -*- coding: utf-8 -*-

from NCDEstimator import NCDEstimator
import lzma

class LZMAEstimator(NCDEstimator):
    
    def __init__(self, splitter):
        super().__init__(splitter)
        self.base = len(self.compress(b''))
        
    def compress(self, ts_data):
        lzc = lzma.LZMACompressor()
        outA = lzc.compress(ts_data)
        outB = lzc.flush()
        out = b"".join([outA, outB])
        return out
    
    def get_compressed_size(self, ts_data):
        out = self.compress(ts_data)
        return len(out) - self.base


        
