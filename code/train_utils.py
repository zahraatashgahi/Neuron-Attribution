import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from code.regularizers import l1, add_input_noise, mixup_data, mixup_criterion

from code.importance import *


def train_epoch(args, model, train_loader, optimiser, loss_func, params, regulariser,
                device=None, mask = None, epoch = 0):
    

    reg_loss = 0
    train_loss = 0
    train_pred_loss = 0
    train_accuracy = 0
    print("="*50)
    print("Epoch = ", epoch, flush=True)
    # measure the time of each epoch
    start_time = time.time()
    # initialize input neuron importance with the size of input
    imp_in_epoch_i = None
    t_neuron_importance = 0
    for i, (data, label) in enumerate(train_loader):
        model.train()
        data, label = data.to(device), label.to(device)
        if regulariser == 'input_noise':
            data = add_input_noise(data, params['std'])
        optimiser.zero_grad()
        output, _, _ = model(data)
        pred_loss = loss_func(output, label)
        predicted_labels = torch.argmax(output, dim=1)
        
        
        # ----------------- Compute input neuron importance -----------------
        start_t_importance = time.time()
        imp_in_itr_i = compute_neuron_importance(args, model, 
                                                 mask, data, label, device=device, epoch=epoch, itr=i)
        if imp_in_epoch_i is None:
            imp_in_itr_i = imp_in_itr_i.reshape(imp_in_itr_i.shape[0], -1)
            imp_in_epoch_i = imp_in_itr_i.detach()
        else:
            imp_in_itr_i = imp_in_itr_i.reshape(imp_in_itr_i.shape[0], -1)
            imp_in_epoch_i += imp_in_itr_i.detach()
        t_neuron_importance += time.time() - start_t_importance
        
        # -------------------------------------------------------------------
        # ---- add regularisation loss (other than l2)
        if regulariser == 'TANGOS' and (params['lambda_1_curr'] > 0 or params['lambda_2_curr'] > 0):
            sparsity_loss, correlation_loss = attr_loss(model, data, device=device, subsample=params['subsample'])
            reg_loss = params['lambda_1_curr'] * sparsity_loss + params['lambda_2_curr'] * correlation_loss

        elif regulariser == 'l1':
            reg_loss = params['weight'] * l1(model)

        elif regulariser == 'mixup':
            X_mixup, y_a, y_b, lam = mixup_data(data, label, alpha=params['alpha'], device=device)
            output_mixup, _, _ = model(X_mixup)
            reg_loss = mixup_criterion(loss_func, output_mixup, y_a, y_b, lam)
        
        #------------------------ backprop ------------------------
        loss = pred_loss + reg_loss
        loss.backward()
        if mask is not None: mask.step()
        else: optimiser.step()

        train_loss += loss.item()
        train_pred_loss += pred_loss.item()
        
        #--------------------- Compute accuracy -------------------
        correct = (predicted_labels == label).sum().item()
        total = len(predicted_labels)
        train_accuracy += correct / total
        torch.cuda.empty_cache()


    # ------------------ print the time for each epoch ------------------
    print("--- [t] Epoch: %s seconds ---" % round(time.time() - start_time, 2))       
    print("--- [t] Neuron Importance: %s seconds ---" % round(t_neuron_importance, 2))   
    return model, mask, train_loss/len(train_loader), \
            train_pred_loss/len(train_loader), train_accuracy/len(train_loader),\
            imp_in_itr_i, imp_in_epoch_i  


def evaluate(model, test_loader, loss_func, device=None):
    running_loss, running_pred_loss, running_accuracy = 0, 0, 0
    for epoch, (data, label) in enumerate(test_loader):
        model.eval()
        data, label = data.to(device), label.to(device)

        output, _, _ = model(data)
        predicted_labels = torch.argmax(output, dim=1)
        
        
        # compute metric
        pred_loss = loss_func(output, label)
        loss = pred_loss

        running_loss += loss.item()
        running_pred_loss += pred_loss.item()
        
        
        # Compute accuracy
        correct = (predicted_labels == label).sum().item()
        total = len(predicted_labels)
        running_accuracy += correct / total

    return running_pred_loss/(epoch + 1), running_loss/(epoch + 1), running_accuracy/(epoch + 1)


def plt_logs(args, fold, imp_in_epoch, imp_in_itr,
             train_loss_all, train_pred_loss_all, 
             val_loss_all, train_acc_all, val_acc_all, 
             epoch, dataset, imp_in_itr_i):
    # save input neuron importance
        np.savetxt(args.save_dir+ "/" + 'imp_in_epoch.txt', imp_in_epoch.detach().cpu().numpy())
        np.savetxt(args.save_dir+ "/" + 'imp_in_itr.txt', imp_in_itr.detach().cpu().numpy()) 

        # save loss and accuracy
        np.savetxt(args.save_dir+ "/loss/" + fold + 'train_loss.txt', train_loss_all) 
        np.savetxt(args.save_dir+ "/loss/" + fold + 'train_pred_loss.txt', train_pred_loss_all) 
        np.savetxt(args.save_dir+ "/loss/" + fold + 'val_loss.txt', val_loss_all) 
        np.savetxt(args.save_dir+ "/acc/" + fold + 'train_acc.txt', train_acc_all) 
        np.savetxt(args.save_dir+ "/acc/" + fold + 'val_acc.txt', val_acc_all) 
        

        plt.figure()
        # Plot the lines -- 
        plt.plot(np.arange(epoch), train_loss_all[:epoch], label='Train Loss')
        plt.plot(np.arange(epoch), train_pred_loss_all[:epoch], label='Train Prediction Loss')
        plt.plot(np.arange(epoch), val_loss_all[:epoch], label='Valid ')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(dataset)
        plt.legend()
        plt.savefig(args.save_dir+ "/loss/" + fold + "_loss.pdf", bbox_inches='tight')
        
        
        plt.figure()
        # Plot the lines -- 
        plt.plot(np.arange(epoch), train_acc_all[:epoch], label='Train Accuracy')
        plt.plot(np.arange(epoch), val_acc_all[:epoch], label='Valid Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(dataset)
        plt.legend()
        plt.savefig(args.save_dir+ "/acc/" + fold + "_acc.pdf", bbox_inches='tight')
        
        if dataset == "mnist":
            # plot strength vector as a 2d image
            # convert stregth to cpu numpy
            strength =  imp_in_itr_i.cpu().detach().numpy()
            strength = strength.reshape(28,28)
            plt.figure()
            sns.heatmap(strength, cmap='viridis')
            plt.savefig(args.save_dir+ "/attribution/" + str(epoch) + "_imp_in.pdf", bbox_inches='tight')
            plt.close()

















