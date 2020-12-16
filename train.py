# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 23:02:00 2020

@author: krajula
"""

import pandas as pd
import numpy as np
import time
from datetime import timedelta
from datetime import datetime
from scipy.fftpack import rfft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import pickle
from sklearn.metrics import accuracy_score,classification_report
from sklearn import svm
from tsfresh.feature_extraction.feature_calculators import autocorrelation

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

def secondsPerDay(tme):
    hours, minutes, seconds = tme.split(':')
    return (int(hours)*60*60)+(int(minutes)*60)+int(seconds)

#def checkTime(tme, tmeRange):
    #return secondsPerDay(tmeRange[0]) < secondsPerDay(tme) <= secondsPerDay(tmeRange[1])


CGM_Patient1_File='CGMData.csv'
CGM_data_P1 = pd.read_csv(CGM_Patient1_File,low_memory=False,parse_dates=[['Date','Time']])
#print(CGM_data_P1.info())
#CGM_data_P1["dateTime"]= pd.to_datetime( CGM_data_P1['Date'] + " " + CGM_data_P1['Time'])
#print(CGM_data_P1.info())
#df["date"]=pd.to_datetime(df["date"])
#print(CGM_data_P1[CGM_data_P1["Date_Time"].between('2018-02-12 13:02:27','2018-02-12 13:02:27')])
CGM_data_P1['Date']=CGM_data_P1.Date_Time.dt.date

CGM_data_P1['Time']=CGM_data_P1.Date_Time.dt.time
#print(pd.to_datetime('2018-02-11 00:02:27')-timedelta(hours=1,minutes=30))

#print(CGM_data_P1.info())
#imp_info=CGM_data_P1[CGM_data_P1.columns[1:3]]
#imp_info['cgvalue']=CGM_data_P1[CGM_data_P1.columns[30:31]]
#print(imp_info.head())
#CGM_data_P1.reindex(CGM_data_P1['dateTime'])
#print(CGM_data_P1["dateTime"].head())
#date_range('2018-04-09', periods=4, freq='1D20min')between_time('2018-02-12 13:22:27','2018-02-12 13:02:27')
#print(CGM_data_P1.between_time('2018-02-12 13:22:27','2018-02-12 13:02:27'))
#print(CGM_data_P1.loc[(CGM_data_P1['dateTime'] >'2018-02-12 13:02:27') & (CGM_data_P1['dateTime'] < '2018-02-12 13:22:27')])



CGM_Patient2_File='CGMData670GPatient3.xlsx'
CGM_data_P2_xsel=pd.read_excel(CGM_Patient2_File)
CGM_data_P2_xsel.to_csv('CGM_Patient2.csv')
CGM_data_P2 = pd.read_csv('CGM_Patient2.csv',low_memory=False,parse_dates=[['Date','Time']])
CGM_data_P2['Date']=CGM_data_P1.Date_Time.dt.date

CGM_data_P2['Time']=CGM_data_P1.Date_Time.dt.time

#print(CGM_data_P2.info())
              
#print(CGM_data_P2['Time'].count())

Insulin_Patient1_File='InsulinData.csv'
Insulin_data_P1 = pd.read_csv(Insulin_Patient1_File,low_memory=False,parse_dates=[['Date','Time']])
Insulin_data_P1['Date']=Insulin_data_P1.Date_Time.dt.date
Insulin_data_P1['Time']=Insulin_data_P1.Date_Time.dt.time

MealTime_P1 = Insulin_data_P1[Insulin_data_P1['BWZ Carb Input (grams)'].notnull()]

MealmodeOn=MealTime_P1.tail(1)
min=CGM_data_P1.head(1)
max=CGM_data_P1.tail(1)
min_value=min['Date_Time']
max_value=max['Date_Time']
datetime_t1=pd.to_datetime(min['Date_Time'])
#print(datetime_t1)


Previous_time_stamp=max_value


mealdata=[]
df = pd.DataFrame() 
dfs={}
d=[]
l_Itime=[]
l_CTime=[]
check=[]
nomeal_d=[]
d_p1_nm=[]
for i, row in enumerate(MealTime_P1[::-1].iterrows()):
    
    temp_I=[row[1][0]]
    l_Itime.append(temp_I)
    temp_C=[CGM_data_P1['Date_Time']]
    l_CTime.append(temp_C)
    Compared_rows=CGM_data_P1[((CGM_data_P1['Date_Time'])>=(row[1][0]))]
    Compared_row=Compared_rows.tail(1)
   
    #print(datetime_t2)
    if i==0:
        t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
   
        start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
        end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
        temp=[start.values[0],t1.values[0],end.values[0]]
        check.append(temp)
    
        nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
        S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
        rows=CGM_data_P1.loc[mask]
        temp=rows['Sensor Glucose (mg/dL)'].to_frame()
        t=list(temp.iloc[: ,0].values)
        dfs[i]=t
        d.append(t)
        
    else:
        t2=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
        #print(t1.values[0],t2.values[0],end.values[0])
        right=t1.values[0]<=t2.values[0]
        #print(right)
        left=t2.values[0]<end.values[0]
        #print(left)
        eq=(t2.values[0]==end.values[0])
        
        St_time = nm['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        #print(St_time)
        Et_time= t2['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        #print(St_time-Et_time)
        mask = (CGM_data_P2['Date_Time'] >= St_time.values[0]) & (CGM_data_P2['Date_Time'] < Et_time.values[0])
        rows_t=CGM_data_P2.loc[mask]
        temp=rows_t['Sensor Glucose (mg/dL)'].to_frame()
        t_p2=list(temp.iloc[: ,0].values)
        
        d_p1_nm.append(t_p2)
        #print(eq)
        if (right and left):
            #print('kavya in between ')
            d.pop()
            check.pop()
            d_p1_nm.pop()
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            temp=[start.values[0],t1.values[0],end.values[0]]
            check.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
            rows=CGM_data_P1.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs[i]=t
            d.append(t)
            
        elif (eq):
            #print('kavya in equal')
            d.pop()
            check.pop()
            d_p1_nm.pop()
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            temp=[start.values[0],t1.values[0],end.values[0]]
            check.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
            rows=CGM_data_P1.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs[i]=t
            d.append(t)
        else:
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            temp=[start.values[0],t1.values[0],end.values[0]]
            check.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
            rows=CGM_data_P1.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs[i]=t
            d.append(t)
            
            
print(len(d))
meal_P1=pd.DataFrame(d)
print(len(l_Itime))
print(len(l_CTime)) 
print(len(check)) 
nomeal_ref_p1=pd.DataFrame(check)
print(nomeal_ref_p1.info())
#nomeal_ref_p1.to_csv("nomeal_reference_time_p1.csv")
#print(meal_P1.info())
meal_P1 = meal_P1.drop(meal_P1.columns[-1],axis=1)
meal_P1=meal_P1.dropna()
#print(meal_P1.info())  
meal_P1.to_csv('mealData1.csv',header=False,index=False)  
     
meal_P1_nm=pd.DataFrame(d_p1_nm)
print(d_p1_nm) 
#meal_P2_nm.drop(meal_P2_nm.columns[:,24:], axis=1, inplace=True)
#meal_P2_nm=meal_P2_nm.dropna()
meal_P1_nm.to_csv('Nomeal1.csv',header=False,index=False)            


Insulin_Patient2_File= pd.read_excel ('InsulinAndMealIntake670GPatient3.xlsx')
Insulin_Patient2_File.to_csv ('Insulin_patient2.csv')
Insulin_data_P2 = pd.read_csv('Insulin_patient2.csv',low_memory=False,parse_dates=[['Date','Time']])
Insulin_data_P2['Date']=Insulin_data_P1.Date_Time.dt.date
Insulin_data_P2['Time']=Insulin_data_P1.Date_Time.dt.time
print(Insulin_data_P2['Time'].count())
#print(Insulin_data_P2.info())
MealTime_P2 = Insulin_data_P2[Insulin_data_P2['BWZ Carb Input (grams)'].notnull()]
mealdata_p2=[]
df = pd.DataFrame()
time_P2=pd.DataFrame()
dfs_p2={}
d_p2=[]
l_Itime_p2=[]
l_CTime_p2=[]
check_p2=[]
nomeal_d_p2=[]
l=[]
d_p2_nm=[]
x=0
for i, row in enumerate(MealTime_P2[::-1].iterrows()):
    
    temp_I=[row[1][0]]
    l_Itime.append(temp_I)
    temp_C=[CGM_data_P2['Date_Time']]
    l_CTime_p2.append(temp_C)
    Compared_rows=CGM_data_P2[((CGM_data_P2['Date_Time'])>=(row[1][0]))]
    Compared_row=Compared_rows.tail(1)
   
    #print(datetime_t2)
    if i==0:
        t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
        nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
        start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
        end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
        
    
        time_P2.append(t1)
        print(time_P2)
        #DF = DF[:-1]
        S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        mask = (CGM_data_P2['Date_Time'] >= S_time.values[0]) & (CGM_data_P2['Date_Time'] < E_time.values[0])
        rows=CGM_data_P2.loc[mask]
        temp=rows['Sensor Glucose (mg/dL)'].to_frame()
        t=list(temp.iloc[: ,0].values)
        dfs_p2[i]=t
        d_p2.append(t)
        temp=[t1.values[0]]
        check_p2.append(temp)
        l.append(t1)
        
    else:
        #print(t1['Date_Time'])
        t2=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
        
        #print(t1.values[0],t2.values[0],end.values[0])
        right=t1.values[0]<=t2.values[0]
        #print(right)
        left=t2.values[0]<end.values[0]
        #print(left)
        eq=(t2.values[0]==end.values[0])
        
        
        
        
        
        #start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
        #end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            #temp=[start.values[0],t1.values[0],end.values[0]]
            #check_p2.append(temp)
        St_time = nm['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        #print(St_time)
        Et_time= t2['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        #print(St_time-Et_time)
        mask = (CGM_data_P2['Date_Time'] >= St_time.values[0]) & (CGM_data_P2['Date_Time'] < Et_time.values[0])
        rows_t=CGM_data_P2.loc[mask]
        temp=rows_t['Sensor Glucose (mg/dL)'].to_frame()
        t_p2=list(temp.iloc[: ,0].values)
        
        d_p2_nm.append(t_p2)
        #print(d_p2_nm)
        
        
        
        
        
        
        
        #print(eq)
        if (right and left):
            #print('kavya in between ')
            d_p2.pop()
            check_p2.pop()
            l.pop()
            d_p2_nm.pop()
            time_P2.drop(time_P2.tail(1).index,inplace=True)
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            #temp=[start.values[0],t1.values[0],end.values[0]]
            #check_p2.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P2['Date_Time'] >= S_time.values[0]) & (CGM_data_P2['Date_Time'] < E_time.values[0])
            rows=CGM_data_P2.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs_p2[i]=t
            d_p2.append(t)
            temp=[t1.values[0]]
            check_p2.append(temp)
            time_P2.append(t1)
            
            l.append(t1)
                
                
                
            
        elif (eq):
            #print('kavya in equal')
            d_p2.pop()
            check_p2.pop()
            l.pop()
            d_p2_nm.pop()
            time_P2.drop(time_P2.tail(1).index,inplace=True)
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            #temp=[start.values[0],t1.values[0],end.values[0]]
            #check_p2.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P2['Date_Time'] >= S_time.values[0]) & (CGM_data_P2['Date_Time'] < E_time.values[0])
            rows=CGM_data_P2.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs_p2[i]=t
            d_p2.append(t)
            temp=[t1.values[0]]
            check_p2.append(temp)
            time_P2.append(t1)
            l.append(t1)
        else:
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            #temp=[start.values[0],t1.values[0],end.values[0]]
            #check_p2.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P2['Date_Time'] >= S_time.values[0]) & (CGM_data_P2['Date_Time'] < E_time.values[0])
            rows=CGM_data_P2.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs_p2[i]=t
            d_p2.append(t)
            temp=[t1.values[0]]
            check_p2.append(temp)
            time_P2.append(t1)
            l.append(t1)
            
            
print(len(d_p2))
meal_P2=pd.DataFrame(d_p2)
print(len(l_Itime_p2))
print(len(l_CTime_p2))
#print(check_p2) 
print(len(check_p2)) 
#nomeal_time=pd.DataFrame(check_p2)
#print(time_P2.info())
#print(time_P2)
#print(len(d_p2_nm))
#print(d_p2_nm)
meal_P2_nm=pd.DataFrame(d_p2_nm)
#meal_P2_nm.drop(meal_P2_nm.columns[:,24:], axis=1, inplace=True)
#meal_P2_nm=meal_P2_nm.dropna()
meal_P2_nm.to_csv('Nomeal2.csv',header=False,index=False)
#print(nomeal_time.info())
#nomeal_time.to_csv('reference_time.csv')
#print(meal_P2.info())
#meal_P2 = meal_P2.drop(meal_P2.columns[-1],axis=1)
meal_P2=meal_P2.dropna()
#print(meal_P2.info())
meal_P2.to_csv('mealData2.csv',header=False,index=False)
#print(MealTime_P1['Time'].count())


mealData_1 = read_data(r"mealData1.csv")
mealData_2 = read_data(r"mealData2.csv")
noMealData_1 = read_data(r"Nomeal1.csv")
noMealData_2 = read_data(r"Nomeal2.csv")
mealData_1 = preprocess_data(mealData_1,24)
mealData_2 = preprocess_data(mealData_2,24)
noMealData_1 = preprocess_data(noMealData_1,24)
noMealData_2 = preprocess_data(noMealData_2,24)
mealData_1.to_csv("m1.csv",header=False,index=False)
mealData_2.to_csv("m2.csv",header=False,index=False)
noMealData_1.to_csv("nm1.csv",header=False,index=False)
noMealData_2.to_csv("nm2.csv",header=False,index=False)



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
    
#model
def svm_classfier(X_train, X_test, y_train, y_test):
    c = svm.SVC(kernel='rbf',gamma="auto")
    c.fit(X_train, y_train)
    y_pred = c.predict(X_test)
    print("Accuracy",accuracy_score(y_pred, y_test) * 100)
    print(classification_report(y_test, y_pred))
    print('_______________________________________________________________________________________________________________________')
    filename = 'svm.pickle'
    pickle.dump(c, open(filename, 'wb'))   
    

tMData = pd.concat([mealData_1,mealData_2],ignore_index=True)
tMData.fillna(tMData.mean(), inplace=True)
fMOfMealData = feature_matrix(tMData)

tNMData=pd.concat([noMealData_1,noMealData_2],ignore_index=True)
tNMData.fillna(tNMData.mean(),inplace=True)
fMOfNoMealData = feature_matrix(tNMData)

normalizedFMMealData = pd.DataFrame(StandardScaler().fit_transform(fMOfMealData))
normalizedFMNoMealData = pd.DataFrame(StandardScaler().fit_transform(fMOfNoMealData))

totalData = normalizedFMMealData.append([normalizedFMNoMealData])

pca = PCA(n_components=5)
pca.fit(totalData)
filename = 'pca.pickle'
pickle.dump(pca, open(filename, 'wb'))

updatedFMMealData = pd.DataFrame(pca.transform(normalizedFMMealData))
updatedFMNoMealData = pd.DataFrame(pca.transform(normalizedFMNoMealData))

updatedFMMealData['label']=1
updatedFMNoMealData['label']=0

Training_Data = updatedFMMealData.append(updatedFMNoMealData)

# K Fold cross validation
kfold = KFold(5, True, 1)
for train_I, test_I in kfold.split(Training_Data):
    training_Data = Training_Data.iloc[train_I]
    test_Data = Training_Data.iloc[test_I]
    X_train, y_train, X_test, y_test = training_Data.loc[:, training_Data.columns != 'label'], training_Data['label'], test_Data.loc[:, test_Data.columns != 'label'], test_Data['label']
    svm_classfier(X_train, X_test, y_train, y_test)