import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.utils import initialize_weights
from models.linear_vdo import LinearVDO
import numpy as np


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.3))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        ard = -2.
        self.attention_a = [
            LinearVDO(L, D, ard_init=ard),
            nn.Tanh()]

        self.attention_b = [LinearVDO(L, D, ard_init=ard),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.3))
            self.attention_b.append(nn.Dropout(0.3))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = LinearVDO(D, n_classes, ard_init=ard)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class probabilistic_MIL_Bayes(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(probabilistic_MIL_Bayes, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.3))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-2.)
        self.n_classes = n_classes
        self.print_sample_trigger = False
        self.num_samples = 16
        self.temperature = torch.tensor([1.0])

        initialize_weights(self)
        self.top_k=top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.temperature = self.temperature.to(device)

    def forward(self, h, validation=False):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK        

        A, h = self.attention_net(h)

        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h)
        logits = self.classifiers(M)

        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        # results_dict = {}

        # if return_features:
        #     top_features = torch.index_select(h, dim=0, index=top_instance_idx)
        #     results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, A


class probabilistic_MIL_Bayes_fc(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2, top_k = 1):
        super(probabilistic_MIL_Bayes_fc, self).__init__()
        assert n_classes == 2
        self.size_dict = {"small": [1024, 512]}
        size = self.size_dict[size_arg]
        fc = [LinearVDO(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.3))

        fc.append(LinearVDO(size[1], n_classes, ard_init=-3.))
        self.classifier = nn.Sequential(*fc)
        initialize_weights(self)
        self.top_k = top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

    def forward(self, h, return_features=False):
        if return_features:
            h = self.classifier.module[:3](h)
            logits = self.classifier.module[3](h)
        else:
            logits  = self.classifier(h) # K x 1

        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict

class probabilistic_MIL_Bayes_vis(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(probabilistic_MIL_Bayes_vis, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.3))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-4.)
        self.n_classes = n_classes
        self.print_sample_trigger = False
        self.num_samples = 16
        self.temperature = torch.tensor([1.0])

        initialize_weights(self)
        self.top_k=top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.temperature = self.temperature.to(device)

    def forward(self, h, validation=False):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK        

        A, h = self.attention_net(h)

        A = torch.transpose(A, 1, 0)  # KxN

        # A = F.softmax(A, dim=1)  # softmax over N

        A = F.sigmoid(A)

        M = torch.mm(A, h)
        logits = self.classifiers(M)

        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        # results_dict = {}

        # if return_features:
        #     top_features = torch.index_select(h, dim=0, index=top_instance_idx)
        #     results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, A

def get_ard_reg_vdo(module, reg=0):
    """
    :param module: model to evaluate ard regularization for
    :param reg: auxilary cumulative variable for recursion
    :return: total regularization for module
    """
    if isinstance(module, LinearVDO): return reg + module.get_reg()
    if hasattr(module, 'children'): return reg + sum([get_ard_reg_vdo(submodule) for submodule in module.children()])
    return reg

bMIL_model_dict = {
                    'A': probabilistic_MIL_Bayes,
                    'F': probabilistic_MIL_Bayes_fc,
                    'vis': probabilistic_MIL_Bayes_vis,
}


