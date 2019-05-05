import pandas as pd
import numpy as np
import datetime as dt
import functools
import dask.dataframe as dd

def month(df):
    df['TrvlMo'] = df['TrvlDt'].dt.month
    return df

def year(df):
    df['TrvlYr'] = df['TrvlDt'].dt.year
    return df

def DOW(df):
    df['DOW'] = df['TrvlDt'].dt.dayofweek
    return df

def time(df):
    df['TrvlTime'] = df['DeptDt'].dt.time
    df['TrvlDt'] = df['DeptDt'].dt.date
    df['TrvlDt'] = pd.to_datetime(df['TrvlDt'])
    return df

def chnl(x):
    other = ['Company Travel','Group desk','PackagingDesk','Unknown']
    if x == 'Kiosk':
        return 'AirportAgent'
    elif x == 'Direct Connect':
        return 'GDS'
    elif x in other:
        return 'Other'
    return x

def diff(df):
    df['DTD'] = (df['TrvlDt'] - df['BkDt']).dt.days
    df = df.loc[df['DTD'] >= 0]
    return df

def connect(df):
    df['Connect'] = np.where((df['Org'] == df['Dept']) & (df['Dest'] == df['Arr']),0,1)
    return df

def duration(df):
    trip = (df.groupby(['PNR'])['DTD'].max() - df.groupby(['PNR'])['DTD'].min()).reset_index()
    trip = trip.rename(columns={'DTD':'Duration'})
    return trip

def journey(df):
    df['OrgDom'] = df['OrgDom'].astype(int)
    df['DestDom'] = df['DestDom'].astype(int)
    df['Domestic'] = (df['OrgDom'] + df['DestDom']) - 1
    return df

def leg(df):
    df['DeptDom'] = df['DeptDom'].astype(int)
    df['ArrDom'] = df['ArrDom'].astype(int)
    df['Domestic'] = (df['DeptDom'] + df['ArrDom']) - 1
    return df

def domestic(x):
    if x <= 0:
        return False
    else:
        return True
    return x

def cumsum(df):
    df['CumPax'] = df.groupby(['Domestic','TrvlYr','TrvlMo','Product'],as_index=False)['Pax'].transform(pd.Series.cumsum)
    df['CumBag'] = df.groupby(['Domestic','TrvlYr','TrvlMo','Product'],as_index=False)['Bag'].transform(pd.Series.cumsum)
    df['CumBagRev'] = df.groupby(['Domestic','TrvlYr','TrvlMo','Product'],as_index=False)['Rev'].transform(pd.Series.cumsum)
    return df

def cumsum_seat(df):
    df['CumPax'] = df.groupby(['Domestic','TrvlYr','TrvlMo','Category'],as_index=False)['Pax'].transform(pd.Series.cumsum)
    df['CumSeat'] = df.groupby(['Domestic','TrvlYr','TrvlMo','Category'],as_index=False)['Seat'].transform(pd.Series.cumsum)
    df['CumSeatRev'] = df.groupby(['Domestic','TrvlYr','TrvlMo','Category'],as_index=False)['SeatRev'].transform(pd.Series.cumsum)
    return df


def cumsum_chnl(df):
    df['CumPax'] = df.groupby(['TrvlYr','TrvlMo','Product','Chnl'],as_index=False)['Pax'].transform(pd.Series.cumsum)
    df['CumBag'] = df.groupby(['TrvlYr','TrvlMo','Product','Chnl'],as_index=False)['Bag'].transform(pd.Series.cumsum)
    df['CumBagRev'] = df.groupby(['TrvlYr','TrvlMo','Product','Chnl'],as_index=False)['Rev'].transform(pd.Series.cumsum)
    return df

def metrics(df):
    df['RPP'] = df['CumBagRev']/df['CumPax']
    df['TR'] = df['CumBag']/df['CumPax']
    df['BR'] = df['CumBagRev']/df['CumBag']
    return df

def metrics_seat(df):
    df['RPP'] = df['CumSeatRev']/df['CumPax']
    df['TR'] = df['CumSeat']/df['CumPax']
    df['BR'] = df['CumSeatRev']/df['CumSeat']
    return df

def metricYoY(df):
    df['RPPYoY'] = (df['RPP'] - df['RPPly'])/df['RPPly']
    df['TRYoY'] = (df['TR'] - df['TRly'])/df['TRly']
    df['BRYoY'] = df['BR'] - df['BRly']
    df.fillna(0)
    return df

def miles(x):
    if x <= 750:
        return 'Short'
    elif x > 750 and x <= 1500:
        return 'Medium'
    elif x > 1500:
        return 'Long'
    else:
        return 'NaN'
    return df

def dtd(x):
    if x <= 21:
        return '0-21'
    elif x > 21 and x <= 45:
        return '22-45'
    elif x > 45 and x <= 60:
        return '46-60'
    elif x > 60:
        return '60+'
    else:
        return 'NaN'
    return df

def pax(x):
    if x == 1:
        return 'Single'
    elif x == 2:
        return 'Couple'
    elif x > 2:
        return 'Group'
    else:
        return 'NaN'
    return df

def cluster(df):
    df['Cluster'] = df['Miles_Group'] + '_' + df['DTD_Group'] + '_' + df['Pax_Group']
    return df

def ratios(df):
    df['CarryOn/Bag1'] = df['CarryOn']/df['Bag1']
    df['Bag2/Bag1'] = df['Bag2']/df['Bag1']
    df['%PNR'] = df['PNR']/df['PNRTotal']
    df['%Connect'] = df['Connect']/df['ConnectTotal']
    return df

def bag1(df):
    df['Bag1TR'] = df['Bag1']/df['Pax']
    df['Bag1RPP'] = df['Bag1Rev']/df['Pax']
    df['Bag1BR'] = df['Bag1Rev']/df['Bag1']
    return df

def carryon(df):
    df['CarryOnTR'] = df['CarryOn']/df['Pax']
    df['CarryOnRPP'] = df['CarryOnRev']/df['Pax']
    df['CarryOnBR'] = df['CarryOnRev']/df['CarryOn']
    return df

def bag2(df):
    df['Bag2TR'] = df['Bag2']/df['Pax']
    df['Bag2RPP'] = df['Bag2Rev']/df['Pax']
    df['Bag2BR'] = df['Bag2Rev']/df['Bag2']
    return df

def checked(df):
    df['CheckedTR'] = df['Checked']/df['Pax']
    df['CheckedRPP'] = df['CheckedRev']/df['Pax']
    df['CheckedBR'] = df['CheckedRev']/df['Checked']
    return df

def cum(df):
    df['CumTR'] = df['Cum']/df['Pax']
    df['CumRPP'] = df['CumRev']/df['Pax']
    df['CumBR'] = df['CumRev']/df['Cum']
    return df

def yoy(df):
    df['%ConnectYoY'] = df['%Connect'] - df['%ConnectLY']
    df['%PNRYoY'] = df['%PNR'] - df['%PNRLY']
    df['CarryOn/BagYoY'] = df['CarryOn/Bag1'] - df['CarryOn/Bag1LY']
    df['Bag2/Bag1YoY'] = df['Bag2/Bag1'] - df['Bag2/Bag1LY']
    df['FareYoY'] = df['Rev'] - df['RevLY']
    df['DurationYoY'] = df['Duration'] - df['DurationLY']

    df['CumRPPYoY'] = df['CumRPP'] - df['CumRPPLY']
    df['CarryOnRPPYoY'] = df['CarryOnRPP'] - df['CarryOnRPPLY']
    df['Bag1RPPYoY'] = df['Bag1RPP'] - df['Bag1RPPLY']
    df['Bag2RPPYoY'] = df['Bag2RPP'] - df['Bag2RPPLY']

    df['CumTRYoY'] = df['CumTR'] - df['CumTRLY']
    df['CarryOnTRYoY'] = df['CarryOnTR'] - df['CarryOnTRLY']
    df['Bag1TRYoY'] = df['Bag1TR'] - df['Bag1TRLY']
    df['Bag2TRYoY'] = df['Bag2TR'] - df['Bag2TRLY']

    df['CumBRYoY'] = df['CumBR'] - df['CumBRLY']
    df['CarryOnBRYoY'] = df['CarryOnBR'] - df['CarryOnBRLY']
    df['Bag1BRYoY'] = df['Bag1BR'] - df['Bag1BRLY']
    df['Bag2BRYoY'] = df['Bag2BR'] - df['Bag2BRLY']
    return df

def wow(df):
    df['CumRPPWoW'] =  df['CumRPPLW'] - df['CumRPPLYW'] # YoY -- Last week
    df['CarryOnRPPWoW'] = df['CarryOnRPPLW'] - df['CarryOnRPPLYW']
    df['Bag1RPPWoW'] = df['Bag1RPPLW'] - df['Bag1RPPLYW']
    df['Bag2RPPWoW'] = df['Bag2RPPLW'] - df['Bag2RPPLYW']

    df['CumTRWoW'] = df['CumTRLW'] - df['CumTRLYW']
    df['CarryOnTRWoW'] = df['CarryOnTRLW'] - df['CarryOnTRLYW']
    df['Bag1TRWoW'] = df['Bag1TRLW'] - df['Bag1TRLYW']
    df['Bag2TRWoW'] = df['Bag2TRLW'] - df['Bag2TRLYW']

    df['CumBRWoW'] = df['CumBRLW'] - df['CumBRLYW']
    df['CarryOnBRWoW'] = df['CarryOnBRLW'] - df['CarryOnBRLYW']
    df['Bag1BRWoW'] = df['Bag1BRLW'] - df['Bag1BRLYW']
    df['Bag2BRWoW'] = df['Bag2BRLW'] - df['Bag2BRLYW']
    return df

def wo2w(df):
    df['CumRPPWo2W'] = df['CumRPPLTW'] - df['CumRPPLYTW'] # YoY -- last two weeks
    df['CarryOnRPPWo2W'] = df['CarryOnRPPLTW'] - df['CarryOnRPPLYTW']
    df['Bag1RPPWo2W'] = df['Bag1RPPLTW'] - df['Bag1RPPLYTW']
    df['Bag2RPPWo2W'] = df['Bag2RPPLTW'] - df['Bag2RPPLYTW']

    df['CumTRWo2W'] = df['CumTRLTW'] - df['CumTRLYTW']
    df['CarryOnTRWo2W'] = df['CarryOnTRLTW'] - df['CarryOnTRLYTW']
    df['Bag1TRWo2W'] = df['Bag1TRLTW'] - df['Bag1TRLYTW']
    df['Bag2TRWo2W'] = df['Bag2TRLTW'] - df['Bag2TRLYTW']

    df['CumBRWo2W'] = df['CumBRLTW'] - df['CumBRLYTW']
    df['CarryOnBRWo2W'] = df['CarryOnBRLTW'] - df['CarryOnBRLYTW']
    df['Bag1BRWo2W'] = df['Bag1BRLTW'] - df['Bag1BRLYTW']
    df['Bag2BRWo2W'] = df['Bag2BRLTW'] - df['Bag2BRLYTW']
    return df

def group(dataframes):
    df = functools.reduce(lambda left,right: dd.merge(left,right,on=['TrvlMo'],how='outer'),dataframes)
    return df
