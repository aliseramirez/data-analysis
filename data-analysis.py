# Initial Data Analysis
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import shapiro

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
initial_analysis(df)


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
percent_missing(df)


non_gauss=[]
def sample_normality(df,list_var):
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
        - List of columns that are not Gaussian 

    """
    alpha = 0.05
    # Subset features of dataset that are numerical 
    for c in list(df.columns):
        if (df[c].dtypes) == 'int' or (df[c].dtypes) == 'float':
            stat,p=shapiro(df)
            if p > alpha:
                print('Sample distribution is Gaussian (Fail to reject Ho)')
            else:
                print('Sample distribution is not Gaussian (Reject Ho)')
                non_gauss.append(df[c])
        return stat,p
    return non_gauss
sample_normality(df,non_gauss)