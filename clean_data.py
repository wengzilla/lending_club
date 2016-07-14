import pdb
import util
import pandas as pd
import numpy as np

# TODO: Maybe don't need all these globals?
YEAR_MAP = {2007: 'a', 2012: 'b', 2014: 'c', 2015: 'd', 2016: 'd'}
EXCLUDE_STATUSES = [
    'Does not meet the credit policy. Status:Charged Off',
    'Does not meet the credit policy. Status:Fully Paid',
]
PERCENT_COLS = ['int_rate', 'revol_util']
TERM = 'term'
EMP_LENGTH = 'emp_length'
DATE_COLS = ['issue_d', 'earliest_cr_line']
DROP_COLS = ['fico_range_high', 'fico_range_low']

def import_all_data(nrows):
    dfs = []
    for year in YEAR_MAP.keys():
        dfs.append(import_data(year, nrows))

    return pd.concat(dfs)

def import_data(year, nrows=None):
    fname = 'data/LoanStats3%s.csv' %YEAR_MAP[year]
    df = pd.read_csv(fname, skiprows=1, nrows=nrows)

    df_feats = df[get_features()].copy()
    df_feats = df_feats.select(lambda x: x not in EXCLUDE_STATUSES)

    # Deal with n/a values
    df_feats.replace('n/a', np.nan, inplace=True)
    df_feats['mths_since_last_delinq'].fillna(-1, inplace=True)
    df_feats['mths_since_last_record'].fillna(-1, inplace=True)
    df_feats.dropna(axis=0, inplace=True)

    # clean columns
    util.df_clean_percent(df_feats, PERCENT_COLS, True)
    util.df_clean_month(df_feats, TERM, True)
    util.df_clean_emp_length(df_feats, EMP_LENGTH, True)
    for d in DATE_COLS: 
        util.df_clean_date(df_feats, d, True)

    # Add some useful columns
    credit_delta =  df_feats['issue_d'] - df_feats['earliest_cr_line']
    df_feats['months_since_fst_credit'] = (credit_delta / np.timedelta64(1, 'M')).astype(int)
    df_feats['fico'] = (df_feats['fico_range_high'] + df_feats['fico_range_low']) / 2
    df_feats['payout_prop'] = (df_feats['total_pymnt'] - df_feats['collection_recovery_fee']) / df_feats['loan_amnt']
    
    # Filter out currents
    df_feats=  df_feats[df_feats['loan_status'] != 'Does not meet the credit policy. Status:Charged Off']
    df_feats = df_feats[df_feats['loan_status'] != 'Does not meet the credit policy. Status:Fully Paid']
    df_feats = df_feats[df_feats['term'] == 36]
    df_feats = df_feats[df_feats['loan_status'] != 'Current']

    # Some manual memory management
    df_feats.drop(DROP_COLS, axis=1, inplace=True)
    df_feats.set_index('id', inplace=True)
    return df_feats

def prep_for_classification(df):
    util.df_binarize_column(df, 'home_ownership', drop_col=True)
    util.df_binarize_column(df, 'purpose', drop_col=True)
    util.df_binarize_column(df, 'verification_status', drop_col=True)
    df['loan_status'] = (df['loan_status'] == 'Fully Paid').astype(int)

    drop_feats = drop_features()
    df_dropped_cols = df.drop(drop_feats, axis=1)
    return df_dropped_cols

def feature_target_split(df):
    y = df['loan_status']
    X = df.drop('loan_status', axis=1)
    return (X, y)

def get_features():
    features = [
        'id',
        'loan_amnt',
        'term',
        'int_rate', # percent clean
        'installment',
        'grade', # binarize? 
        'sub_grade',
        'emp_length', # convert to int
        'home_ownership', # binarize
        'annual_inc',
        'verification_status',
        'purpose', #binarize
        'dti',
        'delinq_2yrs',
        'fico_range_low',
        'fico_range_high', # take avg?
        'mths_since_last_delinq', # set threshold?
        'mths_since_last_record',
        'open_acc', # bucket?
        'pub_rec', # binarize? 
        'revol_util', # clean percent
        'revol_bal',
        'total_acc',
        'zip_code', # ???
        'addr_state', # binarize?!
        'issue_d',
        'earliest_cr_line', # turn to num months since?
        'inq_last_6mths',
        'chargeoff_within_12_mths', # binarize
        'loan_status',
        'total_pymnt',
        'collection_recovery_fee'
    ]

    return features

def drop_features():
    to_drop_features = [
        'term',
        'addr_state',
        'zip_code',
        'sub_grade',
        'earliest_cr_line',
        'issue_d',
        'chargeoff_within_12_mths',
        'mths_since_last_record',
        'mths_since_last_delinq',
        'pub_rec',
        'collection_recovery_fee',
        'total_pymnt',
        'payout_prop'
    ]
    return to_drop_features
