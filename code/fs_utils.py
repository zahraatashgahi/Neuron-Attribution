
###########################################################################################
###########  Import packages
import arff
import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import datasets
import torchvision
import scipy
from scipy.io import loadmat
import urllib.request as urllib2 

def load_data(dataset_name, path = "./data/"):
    #========================== hand-written image datasets ==========================
    if dataset_name == "mnist":
        return load_mnist(path)
    elif dataset_name == "usps":
        return load_usps(path)
    elif dataset_name == "gisette":
        return load_gisette(path)
    #========================== image datasets ==========================
    elif dataset_name == "coil20":
        return load_coil20(path)
    elif dataset_name == "ORL":
        return load_ORL(path)
    elif dataset_name == "Yale":
        return load_Yale(path)
    #========================== text datasets ==========================
    elif dataset_name == "BASEHOCK":
        return load_BASEHOCK(path)
    elif dataset_name == "PCMAC":
        return load_PCMAC(path)
    elif dataset_name == "RELATHE":
        return load_RELATHE(path)
    #========================== biological datasets ==========================
    elif dataset_name == "Prostate_GE":
        return load_Prostate_GE(path)
    elif dataset_name == "SMK":
        return load_SMK(path)
    elif dataset_name == "gla":
        return load_GLA(path)
    elif dataset_name == "CLL":
        return load_CLL(path)
    elif dataset_name == "Carcinom":
        return load_Carcinom(path)
    elif dataset_name == "lymphoma":
        return load_lymphoma(path)
    #========================== other datasets ==========================
    elif dataset_name == "Arcene":
        return load_Arcene(path)
    #========================== time series datasets ==========================
    elif dataset_name == "isolet":
        return load_isolet(path)
    elif dataset_name == "har":
        return load_har(path)
    #========================== noisy datasets ==========================
    elif dataset_name == "madelon":
        return load_madelon(path)
    #========================== synthetic datasets ==========================
    elif dataset_name == "syn_100_200":
        return load_syn_100_200(path)
    elif dataset_name == "syn_500_200":
        return load_syn_500_200(path)
    elif dataset_name == "syn_1000_200":
        return load_syn_1000_200(path)
    elif dataset_name == "syn_10000_200":
        return load_syn_10000_200(path)
    elif dataset_name == "syn_100_500":
        return load_syn_100_500(path)
    elif dataset_name == "syn_500_500":
        return load_syn_500_500(path)
    elif dataset_name == "syn_1000_500":
        return load_syn_1000_500(path)
    elif dataset_name == "syn_10000_500":
        return load_syn_10000_500(path)




#========================== hand-written image datasets ==========================
def load_mnist(path, train_prop=0.8):
    train_data = np.loadtxt(path  + "mnist_train.csv", 
                            delimiter=",")
    test_data = np.loadtxt(path  + "mnist_test.csv", 
                           delimiter=",") 

    X_train = np.asarray(train_data[:, 1:]) 
    X_test = np.asarray(test_data[:, 1:]) 

    y_train = np.asarray(train_data[:, :1])
    y_test = np.asarray(test_data[:, :1])
    
    y_train = np.asarray([int(y_train[i]) for i in range(len(y_train))])
    y_test = np.asarray([int(y_test[i]) for i in range(len(y_test))])
    #print(y_train.shape)
    #print(y_train)
    #print(X_train)
    from sklearn.utils import shuffle
    X, y = shuffle(X_train, y_train, random_state=42)

    X_scaler = MinMaxScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_usps(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'USPS.mat', squeeze_me=True)
    #mat = scipy.io.loadmat('./datasets/COIL20.mat')
    #print(mat.keys())
    X = mat['X']
    y = mat['Y'] 
    # print(y)
    # # print minimum and maximum values of y
    # print("min = ", np.min(y))
    # print("max = ", np.max(y))
    y=y-1
    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_gisette(path, train_prop=0.8):
    X_train = np.loadtxt(path + 'gisette/gisette_train.data')
    y_train = np.loadtxt(path + 'gisette/gisette_train.labels')
    X_test =  np.loadtxt(path + 'gisette/gisette_valid.data')
    y_test =  np.loadtxt(path + 'gisette/gisette_valid.labels')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')
    y_train[y_train==-1] = 0
    y_test[y_test==-1] = 0
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


#========================== image datasets ==========================
def load_coil20(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'COIL20.mat', squeeze_me=True)
    #mat = scipy.io.loadmat('./datasets/COIL20.mat')
    X = mat['fea']
    y = mat['gnd'] 
    y=y-1
    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_ORL(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'ORL.mat', squeeze_me=True)
    #mat = scipy.io.loadmat('./datasets/COIL20.mat')
    X = mat["X"]
    y = mat["Y"]
    y=y-1
    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_Yale(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'Yale.mat', squeeze_me=True)
    #mat = scipy.io.loadmat('./datasets/COIL20.mat')
    X = mat["X"]
    y = mat["Y"]
    y=y-1
    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


###########################################################################################
########### Text 
def load_BASEHOCK(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'BASEHOCK.mat', squeeze_me=True)
    X = mat["X"]
    y = mat["Y"]
    y = y-1
    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_PCMAC(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'PCMAC.mat', squeeze_me=True)
    X = mat["X"]
    y = mat["Y"]
    y = y-1
    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, 
                                                        shuffle=True, random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_RELATHE(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'RELATHE.mat', squeeze_me=True)
    X = mat["X"]
    y = mat["Y"]
    y = y-1
    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

###########################################################################################
########### Biological  
def load_Prostate_GE(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'Prostate_GE.mat', squeeze_me=True)
    X = mat["X"]
    y = mat["Y"]
    y = y-1

    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True,
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_SMK(path, train_prop=0.8):
    mat = scipy.io.loadmat(path +'SMK_CAN_187.mat', squeeze_me=True)
    X = mat["X"]
    #X = X[:, :2000]
    y = mat["Y"]
    y = y-1
    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop,
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test
   
def load_GLA(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'GLA-BRA-180.mat', squeeze_me=True)
    X = mat["X"]
    y = mat["Y"]
    y = y-1

    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_CLL(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'CLL-SUB-111.mat', squeeze_me=True)
    X = mat["X"]
    y = mat["Y"]
    y = y - 1

    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_Carcinom(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'Carcinom.mat', squeeze_me=True)
    X = mat["X"]
    y = mat["Y"]
    y = y - 1

    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_lymphoma(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'lymphoma.mat', squeeze_me=True)
    X = mat["X"]
    y = mat["Y"]
    y = y - 1

    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


###########################################################################################
########### Other
def load_Arcene(path, train_prop=0.8):
    mat = scipy.io.loadmat(path + 'arcene.mat', squeeze_me=True)
    X = mat["X"]
    y = mat["Y"]
    y[y==-1]  = 0

    
    y = np.asarray(y, dtype = 'int')
    X = np.asarray(X, dtype = 'float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_prop, 
                                                        shuffle=True, 
                                                        random_state=42) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


###########################################################################################
###########  time series
def load_isolet(path, train_prop=0.8):
    import pandas as pd 
    data= pd.read_csv(path + 'isolet.csv')
    data = data.values 
    X = data[:,:-1]
    X = X.astype("float")
    y = data[:,-1]
    # y = y - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
    for i in range(len(y_train)):
        if len(y_train[i])==4:
            y_train[i] = int(y_train[i][1])*10 + int(y_train[i][2])
        elif len(y_train[i])==3:
            y_train[i] = int(y_train[i][1])
    for i in range(len(y_test)):
        if len(y_test[i])==4:
            y_test[i] = int(y_test[i][1])*10 + int(y_test[i][2])
        elif len(y_test[i])==3:
            y_test[i] = int(y_test[i][1])
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')
    y_test = y_test - 1
    y_train = y_train - 1
    # turn y to float
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_har(path, train_prop=0.8):
    X_train = np.loadtxt(path + 'UCI HAR Dataset/train/X_train.txt')
    y_train = np.loadtxt(path + 'UCI HAR Dataset/train/y_train.txt')
    X_test =  np.loadtxt(path + 'UCI HAR Dataset/test/X_test.txt')
    y_test =  np.loadtxt(path + 'UCI HAR Dataset/test/y_test.txt')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    y_test = y_test - 1
    y_train = y_train - 1

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


###########################################################################################
###########  noisy
def load_madelon(path, train_prop=0.8):
    X_train = np.loadtxt(path + 'MADELON/madelon_train.data')
    y_train = np.loadtxt(path + 'MADELON/madelon_train.labels')
    X_test =  np.loadtxt(path + 'MADELON/madelon_valid.data')
    y_test =  np.loadtxt(path + 'MADELON/madelon_valid.labels')
    y_train[y_train==-1] = 0
    y_test[y_test==-1] = 0
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


###########################################################################################
###########  synthetic
def load_syn_100_200(path, train_prop=0.8):
    X_train = np.load(path + 'syn_100_200/X_train.npy')
    y_train = np.load(path + 'syn_100_200/y_train.npy')
    X_test =  np.load(path + 'syn_100_200/X_test.npy')
    y_test =  np.load(path + 'syn_100_200/y_test.npy')

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_syn_500_200(path, train_prop=0.8):
    X_train = np.load(path + 'syn_500_200/X_train.npy')
    y_train = np.load(path + 'syn_500_200/y_train.npy')
    X_test =  np.load(path + 'syn_500_200/X_test.npy')
    y_test =  np.load(path + 'syn_500_200/y_test.npy')

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_syn_1000_200(path, train_prop=0.8):
    X_train = np.load(path + 'syn_1000_200/X_train.npy')
    y_train = np.load(path + 'syn_1000_200/y_train.npy')
    X_test =  np.load(path + 'syn_1000_200/X_test.npy')
    y_test =  np.load(path + 'syn_1000_200/y_test.npy')

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_syn_10000_200(path, train_prop=0.8):
    X_train = np.load(path + 'syn_10000_200/X_train.npy')
    y_train = np.load(path + 'syn_10000_200/y_train.npy')
    X_test =  np.load(path + 'syn_10000_200/X_test.npy')
    y_test =  np.load(path + 'syn_10000_200/y_test.npy')

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test




def load_syn_100_500(path, train_prop=0.8):
    X_train = np.load(path + 'syn_100_500/X_train.npy')
    y_train = np.load(path + 'syn_100_500/y_train.npy')
    X_test =  np.load(path + 'syn_100_500/X_test.npy')
    y_test =  np.load(path + 'syn_100_500/y_test.npy')

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_syn_500_500(path, train_prop=0.8):
    X_train = np.load(path + 'syn_500_500/X_train.npy')
    y_train = np.load(path + 'syn_500_500/y_train.npy')
    X_test =  np.load(path + 'syn_500_500/X_test.npy')
    y_test =  np.load(path + 'syn_500_500/y_test.npy')

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_syn_1000_500(path, train_prop=0.8):
    X_train = np.load(path + 'syn_1000_500/X_train.npy')
    y_train = np.load(path + 'syn_1000_500/y_train.npy')
    X_test =  np.load(path + 'syn_1000_500/X_test.npy')
    y_test =  np.load(path + 'syn_1000_500/y_test.npy')

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_syn_10000_500(path, train_prop=0.8):
    X_train = np.load(path + 'syn_10000_500/X_train.npy')
    y_train = np.load(path + 'syn_10000_500/y_train.npy')
    X_test =  np.load(path + 'syn_10000_500/X_test.npy')
    y_test =  np.load(path + 'syn_10000_500/y_test.npy')

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

