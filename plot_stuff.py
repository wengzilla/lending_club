import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import util

def setup_df(df):
    df['total_recovered'] = df['total_pymnt'] + df['recoveries'] - df['collection_recovery_fee']
    df['proportion_recovered'] = df['total_recovered']/df['funded_amnt']

def plot_categorical(category, df, max_categories=10, quantile=False):
    # is categorical?
    filtered = df.dropna(subset=[category])
    if filtered[category].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        # convert to categorical
        filtered[category] = util.gen_categorical(category, filtered, max_categories, quantile)
    grouped = filtered.groupby(category)
    num_groups = len(grouped.groups.keys())
    for ind, k in enumerate(grouped.groups.keys()):
        axs = plt.subplot(num_groups, 1, ind+1)
        plt.hist(grouped.get_group(k)['proportion_recovered'], bins=np.arange(0, 1.5, 0.1))
        plt.title('Group: %s' %k)
        axs.set_xlim(0, 1.5)
    plt.show()

def import_data(year):
    fname = '../LoanStats%d.csv' %year
    df = pd.read_csv(fname, skiprows=1, nrows=10000) 
    return df

if __name__ == '__main__':
    df = import_data(2015)
    setup_df(df)
    filtered = df[df['loan_status'] != 'Current']
    plot_categorical('grade', filtered)
