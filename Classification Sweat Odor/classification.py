#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:10:49 2021

@author: basuki
"""
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  
import numpy as np
import pywt
from statsmodels.robust import mad
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import cohen_kappa_score
from sklearn.neural_network import MLPClassifier
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    RAW = "*.csv"    
    dfRAW = pd.DataFrame([])
    for f in glob.glob(RAW):        
        print("Proses : %s"%f)
        tmpData = pd.read_csv(f)           
        nmFile = f[0:]                      
        tmpData['idFile']=nmFile.replace(".csv","")  
        tmpData['kelas']=f[5:7]
        dfRAW = dfRAW.append(tmpData)
    IDFile = dfRAW.idFile.unique()    
    KELAS  = dfRAW.kelas.unique() 
    KELAS.sort()
    jumlahKolom = dfRAW.shape[1]-2  
    
    dfRAW =  dfRAW.drop(columns=['time(s)','Temp', 'Humid'])
    dfAvg = dfRAW.groupby(['idFile']).agg([np.mean])
    plt.show() 
    
    N_fitur=dfAvg.shape[1] ## jumlah kolom
    for i in dfAvg.index.values:      
        kelas = list(KELAS).index(i[5:7])
        dfAvg.loc[i,'Kelas']=kelas
        
    dataValue = dfAvg.values
    dataX   = dataValue[:,:N_fitur]    
    yLabel  = dataValue[:,N_fitur]
    
    #data split
    X_train, X_test, y_train, y_test = train_test_split(dataX, yLabel, test_size=0.20)
    
    
    # Standardize features by removing mean and scaling to unit variance:
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    # Use the KNN classifier to fit data:
    classifier = KNeighborsClassifier(n_neighbors=4)
    classifier.fit(X_train, y_train) 
    
    # Predict y data with classifier: 
    y_predict = classifier.predict(X_test)
    
    # Print results: 
    print(confusion_matrix(y_test, y_predict))
    print(classification_report(y_test, y_predict))
