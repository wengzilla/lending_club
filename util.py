from __future__ import division
import pandas as pd
import numpy as np
import bisect

def gen_categorical(category, df, max_categories=10, quantile=False):
    max_val = df[category].max()
    min_val = df[category].min()

    if not quantile:
        bucket_size = (max_val - min_val) / max_categories
        buckets = [min_val + i*bucket_size for i in range(1, max_categories)]
    else:
        percentiles = np.arange(0, 1, max_categories)
        buckets = np.percentiles(df[category], percentiles)

    return df[category].apply(lambda x: bisect.bisect_left(buckets, x))
