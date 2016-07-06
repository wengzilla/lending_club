import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_categorical(category, df):
    grouped = df.groupby(category)
    num_groups = len(grouped.groups.keys())
    plt.figure(figsize=(num_groups*150,200))
    for ind, k in enumerate(grouped.groups.keys()):
        plt.subplot(num_groups, 1, ind+1)
        plt.hist(grouped.get_group(k)['proportion_recovered'], bins=20)
        plt.title('Group: %s' %k)
    plt.show()

def import_data(year):
    fname = 'LoanStats%d.csv' %year
    df = pd.read_csv(fname, skiprows=1, nrows=10000) 
    return df

if __name__ == '__main__':
    df = import_data(2015)
    df['total_recovered'] = df['total_pymnt'] + df['recoveries'] - df['collection_recovery_fee']
    df['proportion_recovered'] = df['total_recovered']/df['funded_amnt']
    plot_categorical('loan_status', df)
