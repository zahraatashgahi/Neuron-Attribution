
##########################################################################
####################          Imports                  ###################
import os, shutil, sys  
import torch
import itertools
import json
import copy
import numpy as np
import torch.nn as nn
from torch import optim
import pickle 
sys.path.insert(0,'..')

from code.losses import parameter_schedule, MSE
from code.utils import *
from code.models import MLP
import code.load_data
from code.sparselearning.core import Masking, CosineDecay
from code.utils import load_args
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np    

from code.utils import ids_to_dataloader_split, count_parameters
from code.train_utils import train_epoch, evaluate, plt_logs
##########################################################################
####################             Parameters            ###################
EPOCHS = 200
TRAINING_PATIENCE =  50
#DEVICE = 'cuda:0'



##########################################################################
################    run for a method and a dataset       #################

def run_fold(args, fold_name, model, trainloader, valloader, 
             config, config_dataset, seed, device = None):
    tag, dataset, regulariser, params, fold = fold_name.split(':')
    # loss_func = MSE if config_dataset['type'] == 'regression' else 
    loss_func = nn.CrossEntropyLoss()
    print("loss_func", loss_func)
    print("regulariser", regulariser)
    l2_weight = config['weight'] if regulariser == 'l2' else 0
    print("l2_weight = ", l2_weight)
    optimiser = optim.Adam(model.parameters(), lr=config_dataset['lr'], weight_decay=l2_weight)
    args.lr_scheduler = None
    # if args.nolr_scheduler:
    #     args.lr_scheduler = None
    # else:
    #     args.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[int(args.epochs / 4) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier], last_epoch=-1)


    #------------------------------- Initialize mask sparse==True
    mask = None
    if args.sparse:
        decay = CosineDecay(args.prune_rate, len(trainloader)*(EPOCHS*args.multiplier))
        mask = Masking(optimiser, prune_rate=args.prune_rate, death_mode=args.prune, prune_rate_decay=decay, growth_mode=args.growth,
                       redistribution_mode=args.redistribution, args=args, 
                       train_loader=trainloader, dense_params =count_parameters(model))
        mask.add_module(model, sparse_init=args.sparse_init)
        
    if 'TANGOS' in regulariser:
        parameter_scheduler = parameter_schedule(config['lambda_1'], config['lambda_2'], config['param_schedule'])
    else:
        parameter_scheduler = None
    best_val_loss = np.inf; last_update = 0
    train_loss_all = np.zeros(EPOCHS)
    train_pred_loss_all = np.zeros(EPOCHS)
    val_loss_all = np.zeros(EPOCHS)
    val_acc_all = np.zeros(EPOCHS)
    train_acc_all = np.zeros(EPOCHS)
    imp_in_itr, imp_in_epoch  = None, None
    for epoch in range(EPOCHS):
        if 'TANGOS' in regulariser:
            lambda_1, lambda_2 = parameter_scheduler.get_reg(epoch)
            config['lambda_1_curr'] = lambda_1
            config['lambda_2_curr'] = lambda_2

        model, mask, train_loss, train_pred_loss, train_acc,\
             imp_in_itr_i, imp_in_epoch_i = \
            train_epoch(args, model, trainloader, optimiser, loss_func, config,
                            regulariser, device=device, mask = mask, epoch =epoch)
        # args.lr_scheduler.step()
        val_loss, _, val_acc = evaluate(model, valloader, loss_func, device=device)

        #----------------------------- save the best model 
        if (val_loss < best_val_loss) or (epoch < 5):
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            last_update = epoch
            if mask is not None:
                best_mask = copy.deepcopy(mask.masks)
            else:
                best_mask = None
        
        if imp_in_epoch is None:
            imp_in_epoch = imp_in_epoch_i.detach()
            imp_in_itr = imp_in_itr_i.detach()
        else:
            #--------------------------- append as a new row
            imp_in_epoch = torch.cat((imp_in_epoch, imp_in_epoch_i.detach()), dim=1)
            imp_in_itr = torch.cat((imp_in_itr, imp_in_itr_i.detach()), dim=1)

        train_loss_all[epoch] = train_loss
        train_pred_loss_all[epoch] = train_pred_loss
        val_loss_all[epoch] = val_loss
        val_acc_all[epoch] = val_acc
        train_acc_all[epoch] = train_acc
        # early stopping criteria
        if epoch - last_update == TRAINING_PATIENCE:
            # ToDo save early_stop epoch
            break
        
        # print and plot logs
        plt_logs(args, fold, imp_in_epoch, imp_in_itr,
             train_loss_all, train_pred_loss_all, 
             val_loss_all, train_acc_all, val_acc_all, 
             epoch, dataset, imp_in_itr_i)
        print("Epoch {}, train_loss={:.3f}, train_pred_loss={:.3f}, val_loss={:.3f}"\
                            .format(epoch, train_loss, train_pred_loss, val_loss))
        # print train and test accuracy
        print("train_acc={:.3f}, val_acc={:.3f}\n".format(train_acc, val_acc))

        
    return best_val_loss, best_model, last_update, best_mask


##########################################################################
##################        run cross validation         ###################
def run_cv(args, config_dataset: dict, regulariser: str, params: dict,
            run_name: str, seed: int, device = None):
    
    #------------------------------------------------ load data
    tag, dataset, _, idx_params = run_name.split(':')
    data_fetcher = getattr(code.load_data, config_dataset['loader'])
    if "synthetic" in dataset:
        loaders = data_fetcher(seed=0, name=dataset)
    else:
        loaders, batch_size = data_fetcher(seed=0)
    dropout = params['p'] if regulariser == 'dropout' else 0
    batch_norm = True if regulariser == 'batch_norm' else False

    #------------------------------------------------ Split the data into train and val
    torch.manual_seed(seed); np.random.seed(seed)
    # split train data into train and validation (80% train, 20% validation)
    # I get this erro "IndexError: tensors used as indices must be long, byte or bool tensors" 
    # when I use the following code
    #trainloader, valloader = torch.utils.data.random_split(loaders['train'], [int(0.8*len(loaders['train'])), int(0.2*len(loaders['train']))])
    # So I use the following code instead
    print("len(loaders['train']) = ", len(loaders['train']))
    train_ids, val_ids = torch.utils.data.random_split(range(len(loaders['train'])),
                                                        [int(0.8*len(loaders['train'])),
                                                        int(len(loaders['train'])) - int(0.8*len(loaders['train']))])
    print("len(train_ids) = ", len(train_ids))
    print("len(val_ids) = ", len(val_ids))
    # train_ids, val_ids = torch.utils.data.random_split(loaders['train'], 
    #                     [int(0.8*len(loaders['train'])), 
    #                      int(0.2*len(loaders['train']))])
    best_loss = np.inf
    fold = 0
    
    trainloader, valloader = ids_to_dataloader_split(loaders['train'], train_ids, 
                                                     val_ids, seed=seed, batch_size=batch_size)
    fold_name = run_name + f':{fold}'
    #------------------------------------------------ choose model
    if args.model == "MLP":
        model = MLP(num_features=config_dataset['num_features'], num_outputs=config_dataset['num_outputs'],
                    dropout=dropout, batch_norm=batch_norm).to(device)
    print(model)
    print("model device", device, flush= "True")

    #------------------------------------------------ run
    fold_loss, fold_model, fold_epoch, fold_mask = run_fold(args, fold_name, model, 
                                                trainloader, valloader, params,
                                                config_dataset, seed=seed, device=device)
    if fold_loss < best_loss:
        best_loss = fold_loss
        best_model = copy.deepcopy(fold_model)
        best_mask= copy.deepcopy(fold_mask)
        best_epoch = fold_epoch


    #------------------------------------------------ evalutate best performing model on held out test set
    # loss_func = MSE if config_dataset['type'] == 'regression' else 
    loss_func = nn.CrossEntropyLoss()
    test_loss, _, test_acc = evaluate(best_model, loaders['test'], loss_func, device=device)
    tag, dataset, regulariser, p = run_name.split(':')

    print("test_loss = ", test_loss)
    print("test_acc = ", test_acc)
    #------------------------------------------------ save model + mask
    with open(args.save_dir +"results.txt", "w+") as f:
        f.write("best_valid_loss = {} \n".format(float(best_loss)));  
        f.write("test_loss = {}\n".format(test_loss)); 
        f.write("test_acc = {}\n".format(test_acc));         
        f.write("best_epoch = {}\n".format(best_epoch));   
    if args.seed == 0:
        torch.save(model, args.save_dir + "best_model")
    if args.seed == 0:
        with open(args.save_dir + 'best_mask.pickle', 'wb') as f:
            pickle.dump(best_mask, f)
    with open(args.save_dir + 'params.txt', 'w') as file:
        file.write(json.dumps(params)) # use `json.loads` to do the reverse


def grid_search_iterable(parameter_dict: dict) -> list:
    """Generate an iterable list of hyperparameters from a dictionary containing the values to be considered"""
    keys, values = zip(*parameter_dict.items())
    parameter_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return parameter_grid

def load_config(args, name):
    curr_dir = os.path.dirname(__file__)
    config_dir = os.path.join(curr_dir, "experiments/stand-alone-regularization/", f'{args.dir_configs}/{name}.json')
    with open(config_dir) as f:
        config_dict = json.load(f)
    config_keys = config_dict.keys()
    return config_dict, config_keys


##########################################################################
##################         Run the experiments          ##################
def run_experiment(args, seed, tag: str, device = None):
    # load config files
    config_regs, regularisers = load_config(args, 'regularizers')
    config_data, datasets = load_config(args, 'datasets')
    original_stdout = sys.stdout


    for dataset in datasets:
        for regulariser in regularisers:
            parmaeter_iterable = grid_search_iterable(config_regs[regulariser])
            for idx, param_set in enumerate(parmaeter_iterable):
                print( idx, param_set)
                args.name = dataset
                save_dir = "./results/{}/{}/exp_{}/{}/param_{}/seed_{}/".format(dataset, 
                                                            args.exp,  args.tag,
                                                            regulariser, 
                                                            idx, args.seed)
                args.save_dir = save_dir
                if os.path.isfile(save_dir+ 'params.txt'):
                    print("Path already exists")
                    continue
                else:
                    if os.path.exists(args.save_dir ) and os.path.isdir(args.save_dir ):
                        shutil.rmtree(args.save_dir )
                    if not os.path.exists(args.save_dir ):
                        os.makedirs(args.save_dir )
                        os.mkdir(args.save_dir+ "/attribution")
                        os.mkdir(args.save_dir+ "/loss")
                        os.mkdir(args.save_dir+ "/acc")
                        print(dataset, flush=True)
                        sys.stdout = open(args.save_dir+  "logs.out", "w")
                        print("="*80)
                        
                run_name = f'{tag}:{dataset}:{regulariser}:{idx}'
                
                # run CV on this combination
                print(run_name)
                run_cv(args, config_data[dataset], regulariser, 
                       param_set, run_name, seed, device = device)
                sys.stdout = original_stdout
                #sys.stdout.close()

##########################################################################
####################             Main                  ###################
if __name__ == '__main__':
    args= load_args()
    print(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    print(device, flush= "True")
    run_experiment(args, seed=args.seed, tag=args.tag, device = device)