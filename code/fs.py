# Feature Selection

# ===========================================================
# =================        Load libraries    ================
import os
import numpy as np
import random
import seaborn as sns
from natsort import natsorted
import re
import sys
import pickle

sys.path.insert(0,'..')
from code.fs_utils import load_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ===========================================================
# ===============     Initialization    =====================

path_results = "./results/"
exp = "Experiment1_feature_selection"
reg = "l2"
results = {}
# results_fs = {"itr":{}, "epoch":{}, "sum_epoch":{}}
# set random seed to 0
np.random.seed(0)
random.seed(0)


# ===========================================================
# =================     Read results    =====================
# read datasets in the results folder
datasets = next(os.walk(path_results))[1]
datasets = natsorted(datasets)
print("datasets = ", datasets, flush=True)


all_methods = []
current_datasets = []
for dataset in datasets:
    current_datasets.append(dataset)
    results[dataset] = {}

    # ------------------------------------------- read a new dataset
    print("\n\n")
    print("=="*50)
    print(dataset)

    # load data
    X_train, X_test, y_train, y_test = load_data(dataset, path = "./data/")
    print("X_train = ", X_train.shape)
    print("X_test = ", X_test.shape)
    print("y_train = ", y_train.shape)
    print("y_test = ", y_test.shape, flush=True)
    
    # set the path to the dataset results
    path_dataset = path_results + dataset + "/" + exp + "/"

    # get method directories
    dir_methods = next(os.walk(path_dataset))[1]
    dir_methods = natsorted(dir_methods)
    for dir_method in dir_methods:
        if dir_method not in all_methods: 
            all_methods.append(dir_method)
        if dir_method not in results[dataset]:
            results[dataset][dir_method] = {}
        # ------------------------------------------- read a new method results
        path_method = path_dataset + dir_method + "/"
        for path_param in ["param_0", "param_1", "param_2"]:
            results[dataset][dir_method][path_param] = {}
            path_reg = path_method + "/" + reg + "/"+ path_param +"/"

            # get dir seeds
            dir_seeds = next(os.walk(path_reg))[1]
            dir_seeds = natsorted(dir_seeds)
            results[dataset][dir_method][path_param]= {"valid_loss":[],
                                            "test_loss":[],
                                            "best_epoch":[],
                                            "test_acc":[],  
                                            "fs":{"itr":{}, "epoch":{}, "sum_epoch":{}}}
            for dir_seed in dir_seeds:
 
                # ------------------------------------------------ for each seed, get the results
                path = path_reg + dir_seed + "/"
                flag_do_fs = False
                # check if the results.txt exist
                if not os.path.isfile(path + "results.txt"): continue
                if os.path.isfile(path + "results_fs_sum_epoch.txt"):
                    with open(path + "results_fs_sum_epoch.txt", "r") as f:
                        lines = f.readlines()
                        if dataset == "madelon":
                            if len(lines) < 1: flag_do_fs = True
                        else:
                            if len(lines) < 5: flag_do_fs = True
                else:
                    flag_do_fs = True
                if flag_do_fs == True:
                    if os.path.isfile(path + "results_fs_itr.txt"):
                        os.remove(path + "results_fs_itr.txt")
                    if os.path.isfile(path + "results_fs_epoch.txt"):
                        os.remove(path + "results_fs_epoch.txt")
                    if os.path.isfile(path + "results_fs_sum_epoch.txt"):
                        os.remove(path + "results_fs_sum_epoch.txt")
                
                # read the results.txt
                with open(path + "results.txt", "r") as f:
                    lines = f.readlines()
                    #print(lines)
                    results[dataset][dir_method][path_param]["valid_loss"].append( 
                        float(re.search('best_valid_loss = (.*) \n', lines[0]).group(1)))
                    results[dataset][dir_method][path_param]["test_loss"].append( 
                        float(re.search('test_loss = (.*)\n', lines[1]).group(1)))
                    results[dataset][dir_method][path_param]["test_acc"].append(
                        float(re.search('test_acc = (.*)\n', lines[2]).group(1))*100.0)
                    results[dataset][dir_method][path_param]["best_epoch"].append(
                        int(re.search('best_epoch = (.*)\n', lines[3]).group(1)))    
                    e = re.search('best_epoch =(.*)\n', lines[3]).group(1)
                if not flag_do_fs:
                    # ------------------------------------------- no need to do feature selection
                    # extract results for different K values
                    for key in ["epoch", "itr", "sum_epoch"]:
                        with open(path + "results_fs_{}.txt".format(key), "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                K = int(re.search('K =(.*), Accuracy', line).group(1))
                                acc = float(re.search('Accuracy =(.*)%\n', line).group(1))
                                if K not in results[dataset][dir_method][path_param]["fs"][key]:
                                    results[dataset][dir_method][path_param]["fs"][key][K] = []
                                results[dataset][dir_method][path_param]["fs"][key][K].append(acc)
                else:
                    # ------------------------------------------- do feature selection
                    # load the importance of features
                    imp_in_epoch = np.loadtxt(path + "imp_in_epoch.txt")
                    imp_in_itr = np.loadtxt(path + "imp_in_itr.txt")
                    # sum importance importance_arr[:,:int(e)]
                    sum_epoch = np.sum(imp_in_epoch[:,:int(e)], axis=1)
                    # print("sum_epoch = ", sum_epoch.shape)
                    if dataset == "madelon":
                        Ks = [20]
                    else:
                        Ks = [25, 50, 75, 100, 200]
                    for K in Ks:
                        for importance_arr, key in zip([imp_in_epoch[:,int(e)], imp_in_itr[:,int(e)], sum_epoch], 
                                                    ["epoch", "itr", "sum_epoch"]):
                            # get the indices of the top k important features
                            # print(np.argsort(importance_arr)[::-1].shape)
                            idx = np.argsort(importance_arr)[::-1][:K]

                            X_train_fs = X_train[:, idx]
                            X_test_fs = X_test[:, idx]
                            # train a svm with the top k features
                            clf = SVC()
                            clf.fit(X_train_fs, y_train)
                            y_pred = clf.predict(X_test_fs)
                            acc = accuracy_score(y_test, y_pred)
                            # print K and accuracy
                            # print('K = %d, Accuracy = %.3f%%' % (K, acc*100.0))
                            # write results to file 
                            with open(path + "results_fs_{}.txt".format(key), "a") as f:
                                f.write("K = %d, Accuracy = %.3f%%\n" % (K, acc*100.0))
                            if K not in results[dataset][dir_method][path_param]["fs"][key]:
                                results[dataset][dir_method][path_param]["fs"][key][K] = []
                            results[dataset][dir_method][path_param]["fs"][key][K].append(acc*100.0)
            


        with open("./results_fs.pkl", "wb") as f:
            pickle.dump(results, f)
        # save a text list (all_methods)
        with open("./all_methods.txt", "w") as f:
            for method in all_methods:
                f.write(method + "\n")

