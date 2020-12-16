# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 21:50:11 2020

@author: krajula
"""

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.fftpack import rfft
from tsfresh.feature_extraction.feature_calculators import autocorrelation


def getSVMModel():
    svmModel = pickle.load(open('svm.pickle', 'rb'))
    return svmModel

# Read Data
#read data from csv file
def read_data(filePath):
    r = pd.read_csv(filePath, header=None, sep='\n')
    df = r[0].str.split(',', expand=True)
    df.fillna(value=pd.np.nan, inplace=True)
    df = df.transform(lambda x: pd.to_numeric(x, errors='coerce'))
    return df

#preprocess data 
def preprocess_data(data,x):
    pData = pd.DataFrame(data)
    noColumns = pData.shape[1]
    noOfColumnsToBeAdded = x-noColumns
    if(noColumns<x):
        for x in range(noColumns,noColumns+noOfColumnsToBeAdded):
            pData[x] = float("NaN")
    if(noColumns>x):
        pData = pData.iloc[:,:x]
    # Drop rows having more than 15% empty values
    rIToDrop=[]
    naValuesCountInRows = pData.isna().sum(axis=1)
    for rowIndex in range(len(naValuesCountInRows.values)):
        if naValuesCountInRows.values[rowIndex]>0.15*pData.shape[1]:
            rIToDrop.append(rowIndex)
    pData= pData.drop(rIToDrop)
    pData.reset_index(inplace=True,drop=True)
    # Fill missing values using polynomial interpolation    
    pData.interpolate(method='polynomial',order=3,limit_direction='both',inplace=True)
    # Backward fill
    pData.bfill(inplace=True)
    # Forward fill
    pData.ffill(inplace=True)
    return pData

# Feature 1 MSD
def moving_standard_deviation(dataframe):
    wCount = 0
    wSize = 5
    msd = pd.DataFrame({})
    while(wCount<= dataframe.shape[1]//wSize):
        msd = pd.concat([msd,dataframe.iloc[:, (wSize * wCount):((wSize * wCount)+wSize-1)].std(axis=1)], axis=1,ignore_index=True)
        wCount = wCount+1
    return msd.iloc[:,:5]

# Feature 2 FFT
def fft_feature(df):
    fft =  rfft(df, n=8,axis=1)
    fftFrame=pd.DataFrame(data=fft)
    return fftFrame


# Feature 3 AC
def auto_correlation(dataframe):
    aCFeature = []
    for i in range(0,len(dataframe)):
        aCFeature.append(autocorrelation(dataframe.iloc[i], 1))
    return pd.DataFrame(aCFeature)

# Feature 4 EM 
def expanding_mean(series):
    wSize = 5
    series.expanding(wSize, axis=1).mean()
    eMResult = series.drop(series.iloc[:,0:wSize-1],axis=1)
    return eMResult

# Feature 5 MD
def maximum_deviation(data):
    cols = data.shape[1]
    mDeviations = []
    for i in range(data.shape[0]):
        s = []
        for j in range(data.shape[1]-1):
            d = [(cols-1)-i for i in range(0,cols)]
            s.append(find_slope(data.values[i][j], data.values[i][j+1],d[j], d[j+1]))
        mDeviations.append(find_max_diff(s))
    return pd.DataFrame(mDeviations)
        

def find_slope(x1,x2, y1,y2):
    return y2-y1/x2-x1
    

def find_max_diff(sArray):
    d = -9999
    for i in range(len(sArray)-1):
        if(sArray[i+1]-sArray[i])> d:
            d= sArray[i+1]-sArray[i]
    return d


#constructing matrix
def feature_matrix(data):
    # Feature 1 MSD
    msd_features = moving_standard_deviation(data)
    msd_titles = ['f1_msd'+str(i) for i in range(msd_features.shape[1])]
    msd_features.columns = msd_titles
    # feature 2 ac
    ac_features = auto_correlation(data)
    ac_titles = ['ac'+str(i) for i in range(ac_features.shape[1])]
    ac_features.columns = ac_titles
    # feature 3 fft
    fft_features = fft_feature(data)
    fft_titles = ['f2_fft'+str(i) for i in range(fft_features.shape[1])]
    fft_features.columns= fft_titles
    # feature 4 mvd
    mvd_features = maximum_deviation(data)
    md_titles = ['md'+str(i) for i in range(mvd_features.shape[1])]
    mvd_features.columns = md_titles
    # feature 5 em
    em_features = expanding_mean(data)
    em_titles = ['f3_em'+str(i) for i in range(em_features.shape[1])]
    em_features.columns= em_titles
    
 
    return pd.concat([
            msd_features,
            mvd_features,
            fft_features,
            ac_features,
            em_features,
            
            
            ],axis=1)
    
    
filepath = str(input("give the input file with path:"))
test_Data = read_data(filepath)
test_Data = preprocess_data(test_Data,24)
fMTestData = feature_matrix(test_Data)

nFMTestData = StandardScaler().fit_transform(fMTestData)
pcaModel = pickle.load(open('pca.pickle', 'rb'))
updatedFMTestData = pd.DataFrame(pcaModel.transform(nFMTestData))
svmModel = getSVMModel()
ResultSVM = svmModel.predict(updatedFMTestData)

r = pd.DataFrame(ResultSVM)
print(r)
r.to_csv('Result.csv',header=False, index=False)


