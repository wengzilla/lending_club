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

def import_all_data():
    pass

def import_data(year, nrows=10000):
    fname = '../LoanStats3%s.csv' %YEAR_MAP[year]
    df = pd.read_csv(fname, skiprows=1, nrows=nrows)
    df_feats = df[get_features()].copy()
    if year == 2007:
        df_feats = df_feats.select(lambda x: x not in EXCLUDE_STATUSES)

    df_feats.replace('n/a', np.nan, inplace=True)
    util.df_clean_percent(df_feats, PERCENT_COLS, True)
    util.df_clean_month(df_feats, TERM, True)
    util.df_clean_emp_length(df_feats, EMP_LENGTH, True)
    for d in DATE_COLS: 
        util.df_clean_date(df_feats, d, True)

    # Fill na
    df_feats['mths_since_last_delinq'].fillna(-1, inplace=True)
    df_feats['mths_since_last_record'].fillna(-1, inplace=True)

    # Add columns
    credit_delta = df_feats['earliest_cr_line'] - df_feats['issue_d']
    df_feats['months_since_fst_credit'] = (credit_delta / np.timedelta64(1, 'M')).astype(int)
    df_feats['fico'] = (df_feats['fico_range_high'] - df_feats['fico_range_low']) / 2

    df_feats.drop(DROP_COLS, axis=1, inplace=True)
    df_feats.dropna(axis=0, inplace=True)
    return df_feats

def get_features():
    features = [
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
        'loan_status'
    ]

    return features
