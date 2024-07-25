
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
import pickle
import urllib.request as urllib2 

DEVICE = 'cuda:0'
def get_path(device):
    if re.match("cuda:[0-9]", device):
        return './data/'
    elif device == 'cpu':
        return os.path.dirname(__file__) + './data/'
    else:
        raise ValueError(f'Device {device} not matched to known devices')


#========================== hand-written image datasets ==========================
def load_mnist(seed, train_prop=0.8, batch_size=100):
    train_data = np.loadtxt(get_path(DEVICE)  + "mnist_train.csv", 
                            delimiter=",")
    test_data = np.loadtxt(get_path(DEVICE)  + "mnist_test.csv", 
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    
    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    
    
    return loaders, batch_size

def load_usps(seed, train_prop=0.8, batch_size=100):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'USPS.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_gisette(seed, train_prop=0.8, batch_size=100):
    X_train = np.loadtxt(get_path(DEVICE)  + 'gisette/gisette_train.data')
    y_train = np.loadtxt(get_path(DEVICE)  + 'gisette/gisette_train.labels')
    X_test =  np.loadtxt(get_path(DEVICE)  + 'gisette/gisette_valid.data')
    y_test =  np.loadtxt(get_path(DEVICE)  + 'gisette/gisette_valid.labels')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')
    y_train[y_train==-1] = 0
    y_test[y_test==-1] = 0
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size


###########################################################################################
########### Text 
def load_BASEHOCK(seed, train_prop=0.8, batch_size=32):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'BASEHOCK.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_PCMAC(seed, train_prop=0.8, batch_size=32):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'PCMAC.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_RELATHE(seed, train_prop=0.8, batch_size=32):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'RELATHE.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size


###########################################################################################
###########  Load image datasets

def load_coil20(seed, train_prop=0.8, batch_size=100):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'COIL20.mat', squeeze_me=True)
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
    X_scaler = MinMaxScaler()   #
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_ORL(seed, train_prop=0.8, batch_size=100):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'ORL.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_Yale(seed, train_prop=0.8, batch_size=32):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'Yale.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size


###########################################################################################
########### Biological  
def load_Prostate_GE(seed, train_prop=0.8, batch_size=32):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'Prostate_GE.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_SMK(seed, train_prop=0.8, batch_size=32):
   
    mat = scipy.io.loadmat(get_path(DEVICE) +'SMK_CAN_187.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size
    
def load_GLA(seed, train_prop=0.8, batch_size=32):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'GLA-BRA-180.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_CLL(seed, train_prop=0.8, batch_size=32):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'CLL-SUB-111.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_Carcinom(seed, train_prop=0.8, batch_size=32):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'Carcinom.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_lymphoma(seed, train_prop=0.8, batch_size=32):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'lymphoma.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

###########################################################################################
########### Other
def load_Arcene(seed, train_prop=0.8, batch_size=32):
    mat = scipy.io.loadmat(get_path(DEVICE) + 'arcene.mat', squeeze_me=True)
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

###########################################################################################
###########  time series
def load_isolet(seed, train_prop=0.8, batch_size=100):
    import pandas as pd 
    data= pd.read_csv(get_path(DEVICE) + 'isolet.csv')
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

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_har(seed, train_prop=0.8, batch_size=100):
    X_train = np.loadtxt(get_path(DEVICE) + 'UCI HAR Dataset/train/X_train.txt')
    y_train = np.loadtxt(get_path(DEVICE) + 'UCI HAR Dataset/train/y_train.txt')
    X_test =  np.loadtxt(get_path(DEVICE) + 'UCI HAR Dataset/test/X_test.txt')
    y_test =  np.loadtxt(get_path(DEVICE) + 'UCI HAR Dataset/test/y_test.txt')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    y_test = y_test - 1
    y_train = y_train - 1

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size


###########################################################################################
###########  noisy
def load_madelon(seed, train_prop=0.8, batch_size=64):
    X_train = np.loadtxt(get_path(DEVICE) + 'MADELON/madelon_train.data')
    y_train = np.loadtxt(get_path(DEVICE) + 'MADELON/madelon_train.labels')
    X_test =  np.loadtxt(get_path(DEVICE) + 'MADELON/madelon_valid.data')
    y_test =  np.loadtxt(get_path(DEVICE) + 'MADELON/madelon_valid.labels')
    y_train[y_train==-1] = 0
    y_test[y_test==-1] = 0
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size


###########################################################################################
###########  synthetic
def load_syn_100_200(seed, train_prop=0.8, batch_size=32):
    X_train = np.load(get_path(DEVICE)  + 'syn_100_200/X_train.npy')
    y_train = np.load(get_path(DEVICE)  + 'syn_100_200/y_train.npy')
    X_test =  np.load(get_path(DEVICE)  + 'syn_100_200/X_test.npy')
    y_test =  np.load(get_path(DEVICE)  + 'syn_100_200/y_test.npy')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_syn_100_500(seed, train_prop=0.8, batch_size=32):
    X_train = np.load(get_path(DEVICE)  + 'syn_100_500/X_train.npy')
    y_train = np.load(get_path(DEVICE)  + 'syn_100_500/y_train.npy')
    X_test =  np.load(get_path(DEVICE)  + 'syn_100_500/X_test.npy')
    y_test =  np.load(get_path(DEVICE)  + 'syn_100_500/y_test.npy')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size


def load_syn_500_200(seed, train_prop=0.8, batch_size=100):
    X_train = np.load(get_path(DEVICE)  + 'syn_500_200/X_train.npy')
    y_train = np.load(get_path(DEVICE)  + 'syn_500_200/y_train.npy')
    X_test =  np.load(get_path(DEVICE)  + 'syn_500_200/X_test.npy')
    y_test =  np.load(get_path(DEVICE)  + 'syn_500_200/y_test.npy')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size


def load_syn_500_500(seed, train_prop=0.8, batch_size=100):
    X_train = np.load(get_path(DEVICE)  + 'syn_500_500/X_train.npy')
    y_train = np.load(get_path(DEVICE)  + 'syn_500_500/y_train.npy')
    X_test =  np.load(get_path(DEVICE)  + 'syn_500_500/X_test.npy')
    y_test =  np.load(get_path(DEVICE)  + 'syn_500_500/y_test.npy')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_syn_1000_200(seed, train_prop=0.8, batch_size=100):
    X_train = np.load(get_path(DEVICE)  + 'syn_1000_200/X_train.npy')
    y_train = np.load(get_path(DEVICE)  + 'syn_1000_200/y_train.npy')
    X_test =  np.load(get_path(DEVICE)  + 'syn_1000_200/X_test.npy')
    y_test =  np.load(get_path(DEVICE)  + 'syn_1000_200/y_test.npy')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_syn_1000_500(seed, train_prop=0.8, batch_size=100):
    X_train = np.load(get_path(DEVICE)  + 'syn_1000_500/X_train.npy')
    y_train = np.load(get_path(DEVICE)  + 'syn_1000_500/y_train.npy')
    X_test =  np.load(get_path(DEVICE)  + 'syn_1000_500/X_test.npy')
    y_test =  np.load(get_path(DEVICE)  + 'syn_1000_500/y_test.npy')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size



def load_syn_10000_200(seed, train_prop=0.8, batch_size=100):
    X_train = np.load(get_path(DEVICE)  + 'syn_10000_200/X_train.npy')
    y_train = np.load(get_path(DEVICE)  + 'syn_10000_200/y_train.npy')
    X_test =  np.load(get_path(DEVICE)  + 'syn_10000_200/X_test.npy')
    y_test =  np.load(get_path(DEVICE)  + 'syn_10000_200/y_test.npy')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size

def load_syn_10000_500(seed, train_prop=0.8, batch_size=100):
    X_train = np.load(get_path(DEVICE)  + 'syn_10000_500/X_train.npy')
    y_train = np.load(get_path(DEVICE)  + 'syn_10000_500/y_train.npy')
    X_test =  np.load(get_path(DEVICE)  + 'syn_10000_500/X_test.npy')
    y_test =  np.load(get_path(DEVICE)  + 'syn_10000_500/y_test.npy')
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders, batch_size








