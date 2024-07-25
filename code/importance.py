
##########################################################################
####################          Imports                  ###################
import sys  
import torch
sys.path.insert(0,'..')
from code.utils import *
import warnings
warnings.filterwarnings("ignore")
from functorch import jacrev
from functorch import vmap
from torch.autograd.functional import jacobian



##########################################################################
####################             Functions             ###################

def compute_neuron_importance(args, model,
        mask, data_input, label, device='cpu', epoch=0, itr = 0):
    """
    Compute input neuron importance 
    """
    # print(("Device = ", device), flush=True)
    if args.fs_method == "QuickSelection":
        imp_in = QuickSelection(args, model, mask, data_input, device=device)
    elif args.fs_method == "Random":
        imp_in = Random(args, model, mask, data_input, device=device)
    elif args.fs_method == "output_attribution":
        imp_in = output_attribution(args, model, mask, data_input, label, device=device, epoch=epoch, itr=itr)
    return imp_in


def QuickSelection(args, model, mask,
            data_input, device='cuda:0'):
    """
    Compute input neuron importance based on neuron strength
    """
    # input features importance
    w1 = model.fc1.weight.to(device)
    strength = torch.sum(torch.abs(w1), dim=0).to(device)
    return strength

def Random(args, model, mask,
            data_input, device='cuda:0'):
    """
    Compute input neuron importance randomly
    """
    # input features importance
    w1 = model.fc1.weight.to(device)
    strength = torch.sum(torch.abs(w1), dim=0).to(device)
    # replace strength with random values
    strength = torch.rand(strength.shape).to(device)
    return strength

def output_attribution(args, model,
        mask, data_input, label, device='cpu', epoch=0, itr = 0):
    """
    Compute input neuron importance using neuron attribution
    """
    batch_size = data_input.shape[0]
    def to_latent(input_):
        out, h2_out, h1_out = model(input_)
        return out, h2_out, h1_out
    
    # compute jacobian and neuron attribution
    jacobian = {}
    data_input = data_input.clone().requires_grad_(True)
    jacobian = vmap(jacrev(to_latent))(data_input)
    
    j_output = jacobian[0].to(device)
    imp_input = torch.mean(torch.abs(j_output), dim=0).sum(dim=0).to(device) 
    return imp_input