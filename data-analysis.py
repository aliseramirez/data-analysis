# Initial Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import shapiro
import warnings
warnings.filterwarnings("ignore")


def initial_analysis(df):
    """
    Given a dataframe produces a simple report on initial data analytics

    Params:
        - df 

    Returns:
        - Shape of dataframe records and columns
        - Columns and data types

    """
    print('Report of Initial Data Analysis:\n')
    print(f'Shape of dataframe: {df.shape}')
    print(f'Features and Data Types: \n {df.dtypes}')

def percent_missing(df):
    """
    Given a dataframe it calculates the percentage of missing records per column

    Params:
        - df

    Returns:
        - Dictionary of column name and percentage of missing records
    
    """
    col=list(df.columns)
    perc=[round(df[c].isna().mean()*100,2) for c in col]
    miss_dict=dict(zip(col,perc))
    return miss_dict

num_feat=[]
def numerical_features(df):
    for c in list(df.columns):
        if (df[c].dtypes) == 'int' or (df[c].dtypes) == 'float':
            num_feat.append(c)
    return num_feat

def normality(df,col_list):
    """
    Given a dataframe determines whether each numerical column is Gaussian 

    Ho = Assumes distribution is not Gaussian
    Ha = Assumes distribution is Gaussian

    Params:
        - df

    Returns:
        - W Statistic
        - p-value
        - Acceptance or Rejection of null hypothesis (Ho)
        - List of columns that do not have gaussian distribution

    """
    non_gauss=[]
    w_stat=[]
    # Determine if each sample of numerical feature is gaussian
    alpha = 0.05
    for f in num_feat:
        stat,p=shapiro(df[f])
        if p <= alpha: # Reject Ho -- Distribution is not normal
            non_gauss.append(f)
            w_stat.append(stat)
    # Dictionary of numerical features not gaussian and W-Statistic        
    norm_dict=dict(zip(non_gauss,w_stat))
    return norm_dict

def skew_kurtosis(df,norm_dict):
    """
    Calculates the skewness and kurtosis of columns that were 
    identified to be non-gaussian
    
    Params: 
        - df
        - norm_dict, dictionary with keys representing non-gaussian columns

    Returns:
        - Skewness
        - Kurtosis
        
    """
    for k in list(norm_dict.keys()):
        sk_tup=tuple((skew(df[k]),kurtosis(df[k])))
        print(k)
        print(sk_tup)

