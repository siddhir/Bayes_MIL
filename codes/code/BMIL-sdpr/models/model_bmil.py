import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.utils import initialize_weights
from models.linear_vdo import LinearVDO, Conv2dVDO
import numpy as np
from torch.distributions import kl

EPS_1 = 1e-16
# EPS_2 = 1e-28

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
            self.module.append(nn.Dropout(0.25))

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
        ard_init = -4.
        self.attention_a = [
            LinearVDO(L, D, ard_init=ard_init),
            nn.Tanh()]

        self.attention_b = [LinearVDO(L, D, ard_init=ard_init),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = LinearVDO(D, n_classes, ard_init=ard_init)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class DAttn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(DAttn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        # print(x.shape)
        return A, x

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

class probabilistic_MIL_Bayes(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(probabilistic_MIL_Bayes, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-3.)
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
            fc.append(nn.Dropout(0.25))

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
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 2)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-3.)
        self.n_classes = n_classes
        self.print_sample_trigger = False
        self.num_samples = 16
        self.temperature = torch.tensor([1.0])
        self.fixed_b = torch.tensor([5.], requires_grad=False)

        initialize_weights(self)
        self.top_k=top_k

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.temperature = self.temperature.to(device)

    def forward(self, h, validation=False):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK        

        A, h = self.attention_net(h)

        # # [1] JUST Sigmoid attn_net-n_classes = 1
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.sigmoid(A)
        # # JUST Sigmoid

        # [2] USING BETA attn_net-n_classes = 2
        # softplus and written form
        # A = F.softplus(A, threshold=8.)
        # A = torch.log(torch.exp(A) + 1 + EPS_1)
        # A = F.relu(A) + EPS_1
        # # print('***********************************')
        # # print(A)
        # # print('*max: {}, min: {}'.format(torch.max(A), torch.min(A)))

        # if torch.isnan(A).sum() > 0:
        #     print(A)
        #     for k, v in self.attention_net.state_dict().items():
        #         print(k, v)
        # postr_sp = torch.distributions.beta.Beta(A[:,0], A[:,1])
        # # A = postr_sp.rsample().unsqueeze(0).clamp(min=1e-20)

        # A = postr_sp.rsample().unsqueeze(0)

        # # print(A.shape)
        # # print(torch.max(A, 1))
        # # print(A[0][torch.max(A, 1)[1]])
        # # A[0][torch.max(A, 1)[1]] += 1e-20
        # A_clone = A.clone()
        # A_clone[0][torch.max(A, 1)[1]] = A_clone[0][torch.max(A, 1)[1]].clamp(min=1e-8)
        # A = A_clone

        # print(torch.max(A), torch.min(A))
        # print(A)
        # print('*max: {}, min: {}'.format(torch.max(A), torch.min(A)))

        # [3] USING BETA, pred-conc parameterization attn_net-n_classes = 2
        # A = F.softplus(A, threshold=8.)
        # a = F.sigmoid(A[:, 0])
        # softplus and writen form
        # b = F.softplus(A[:, 1], threshold=3.)
        # b = torch.log(torch.exp(A[:, 1]) + 1 + EPS_1)

        # # alpha = a * self.fixed_b
        # # beta  = self.fixed_b - a * self.fixed_b
        # alpha = (a * b)
        # beta  = (b - a * b)

        # postr_sp = torch.distributions.beta.Beta(alpha, beta)
        # A = postr_sp.rsample().unsqueeze(0)

        # print('sample max: {0:.4f}, sample min: {1:.4f}.'.format(torch.max(A), torch.min(A)))
        # print('a      max: {0:.4f}, a      min: {1:.4f}.'.format(torch.max(a), torch.min(a)))
        # print('b      max: {0:.4f}, b      min: {1:.4f}.'.format(torch.max(b), torch.min(b)))
        # print('alpha  max: {0:.4f}, alpha  min: {1:.4f}.'.format(torch.max(alpha), torch.min(alpha)))
        # print('beta   max: {0:.4f}, beta   min: {1:.4f}.'.format(torch.max(beta), torch.min(beta)))

        # A = F.relu(A) + EPS_1
        # # print('***********************************')
        # # print(A)
        # # print('*max: {}, min: {}'.format(torch.max(A), torch.min(A)))

        # if torch.isnan(A).sum() > 0:
        #     print(A)
        #     for k, v in self.attention_net.state_dict().items():
        #         print(k, v)
        # postr_sp = torch.distributions.beta.Beta(A[:,0], A[:,1])
        # # A = postr_sp.rsample().unsqueeze(0).clamp(min=1e-20)

        # A = postr_sp.rsample().unsqueeze(0)

        # # print(A.shape)
        # # print(torch.max(A, 1))
        # # print(A[0][torch.max(A, 1)[1]])
        # # A[0][torch.max(A, 1)[1]] += 1e-20
        # A_clone = A.clone()
        # A_clone[0][torch.max(A, 1)[1]] = A_clone[0][torch.max(A, 1)[1]].clamp(min=1e-8)
        # A = A_clone

        # print(torch.max(A), torch.min(A))
        # print(A)
        # print('*max: {}, min: {}'.format(torch.max(A), torch.min(A)))

        # [4] USING DIRICHLET -> BETA attn_net-n_classes = 1

        ### If we add the marginalization, will it still work?

        # A = F.softplus(A, threshold=8.).squeeze(1)
        # A = (F.relu(A) + EPS_1).squeeze(1)
        # # print('***********************************')
        # # print(A)
        # # print('*max: {}, min: {}'.format(torch.max(A), torch.min(A)))
        # postr_sp = torch.distributions.beta.Beta(A, A.sum() - A)
        # A = postr_sp.rsample().unsqueeze(0)
        # print(A.shape)
        # # print('*max: {}, min: {}'.format(torch.max(A), torch.min(A)))

        # [5] USING logistic normal
        # we stick to this one 
        mu = A[:, 0]
        logvar = A[:, 1]
        gaus_samples = self.reparameterize(mu, logvar)
        beta_samples = F.sigmoid(gaus_samples)
        A = beta_samples.unsqueeze(0)
        # print('gaus   max: {0:.4f}, gaus   min: {1:.4f}.'.format(torch.max(gaus_samples), torch.min(gaus_samples)))
        # print('sample max: {0:.4f}, sample min: {1:.4f}.'.format(torch.max(A), torch.min(A)))

        M = torch.mm(A, h) / A.sum()
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


class probabilistic_MIL_Bayes_enc(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(probabilistic_MIL_Bayes_enc, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        first_transform = nn.Linear(size[0], size[1])
        fc1 = [first_transform, nn.ReLU()]
        # fc2 = [first_transform, nn.ReLU()]

        if dropout:
            fc1.append(nn.Dropout(0.25))
            # fc2.append(nn.Dropout(0.25))

        if gate:
            # attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            # postr_net = DAttn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            # prior_net = DAttn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            postr_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 2)
            # prior_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        else:
            # attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            postr_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            prior_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        fc1.append(postr_net)
        # fc2.append(prior_net)

        self.postr_net = nn.Sequential(*fc1)
        # self.prior_net = nn.Sequential(*fc2)
        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-3.)

        # self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes
        self.print_sample_trigger = False
        self.num_samples = 16
        self.temperature = torch.tensor([1.0])
        # self.sf_pos = torch.tensor([2e4], requires_grad=False)
        # self.sf_neg = torch.tensor([2e4], requires_grad=False)
        # self.sf_pos = torch.tensor([1.], requires_grad=False)
        self.prior_mu = torch.tensor([-5., 0.])
        self.prior_logvar = torch.tensor([-1., 3.])

        # self.sf_neg = torch.tensor([1.], requires_grad=False)
        initialize_weights(self)
        self.top_k = top_k

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.attention_net = self.attention_net.to(device)
        self.postr_net = self.postr_net.to(device)
        # self.prior_net = self.prior_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.temperature = self.temperature.to(device)
        # self.sf_pos = self.sf_pos.to(device)
        self.prior_mu = self.prior_mu.to(device)
        self.prior_logvar = self.prior_logvar.to(device)

        # self.sf_neg = self.sf_neg.to(device)

    def kl_logistic_normal(self, mu_pr, mu_pos, logvar_pr, logvar_pos):
        return (logvar_pr - logvar_pos) / 2. + (logvar_pos ** 2 + (mu_pr - mu_pos) ** 2) / (2. * logvar_pr ** 2) -0.5

    def forward(self, h, return_features=False, slide_label=None, validation=False):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK 

        param, h = self.postr_net(h)
        # prior_alpha, _ = self.prior_net(h)

        # if slide_label == 0:
        #     mu_pr = prior_mu[0].expand(h.shape[0])
        #     logvar_pr = self.prior_logvar[0]
        # else:
        #     mu_pr = torch.tensor([0.] * h.shape[0]).cuda()
        #     logvar_pr = self.prior_logvar[1]

        mu = param[:, 0]
        logvar = param[:, 1]
        gaus_samples = self.reparameterize(mu, logvar)
        beta_samples = F.sigmoid(gaus_samples)
        A = beta_samples.unsqueeze(0)

        if not validation:
            mu_pr = self.prior_mu[slide_label.item()].expand(h.shape[0])
            logvar_pr = self.prior_logvar[slide_label.item()]
            kl_div = self.kl_logistic_normal(mu_pr, mu, logvar_pr, logvar)
        else:
            kl_div = None

        # if negative, all patches should be checked with equal probabilities.
        # postr_alpha *= torch.exp(slide_label * torch.tensor([conc_expo]))

        # postr_alpha = F.softplus(torch.transpose(postr_alpha, 1, 0))  # KxN
        # prior_alpha = torch.exp(torch.transpose(prior_alpha, 1, 0))  # KxN
        # postr_alpha = (F.relu(postr_alpha) + EPS).squeeze(1)

        # print('***************************')
        # print('before: ', postr_alpha)
        # print('component 1: ', (self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1)))
        # # print('component 1 clamp: ', (self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1)).clamp(min=1.0))
        # print('component 1 max: {}, min: {}: '.format(torch.max((self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1))),
        #     torch.min((self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1)))))
        # print('component 2: ', (self.sf_neg * torch.softmax(postr_alpha / 5., dim=1)))
        # # print('component 2 clamp: ', (self.sf_neg * torch.softmax(postr_alpha / 5., dim=1)).clamp(max=0.95))
        # print('component 2 max: {}, min: {}: '.format(torch.max((self.sf_neg * torch.softmax(postr_alpha / 5., dim=1))),
        #     torch.min((self.sf_pos * torch.softmax(postr_alpha / 5., dim=1)))))

        # postr_alpha = slide_label.detach() * (self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1)).clamp(min=1.0) \
        # + (1. - slide_label).detach() * (self.sf_neg * torch.softmax(postr_alpha / 5., dim=1)).clamp(max=0.95)

        # if slide_label == 1:
        #     postr_alpha = (self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1)).clamp(min=1.)
        # else:
        #     # postr_alpha = (self.sf_neg * torch.softmax(postr_alpha / 5., dim=1))
        #     postr_alpha = (self.sf_neg * torch.softmax(postr_alpha / 10., dim=1)).clamp(max=0.9)

        # if slide_label == 1:
        #     prior_alpha = torch.ones(h_.shape[0]).cuda()
        # else:
        #     # postr_alpha = (self.sf_neg * torch.softmax(postr_alpha / 5., dim=1))
        #     prior_alpha = torch.tensor([1. / h_.shape[0]]*h_.shape[0]).cuda()

            # postr_alpha = (self.sf_neg * torch.softmax(postr_alpha / 10., dim=1)).clamp(max=0.9)

        # postr_alpha = torch.exp(postr_alpha)

        # print('slide label: ', slide_label)
        # print('after: ', postr_alpha)
        # print('prior_alpha: ', prior_alpha)

        # postr_kl = torch.distributions.dirichlet.Dirichlet(postr_alpha)
        # postr_sp = torch.distributions.beta.Beta(postr_alpha, postr_alpha.sum() - postr_alpha)
        # prior_kl = torch.distributions.dirichlet.Dirichlet(prior_alpha)
        # prior_sp = torch.distributions.beta.Beta(prior_alpha, prior_alpha.sum() - prior_alpha)
        # prior_kl = torch.distributions.dirichlet.Dirichlet(prior_alpha)

        # if self.training:
        #     kl_div = kl.kl_divergence(postr_kl, prior_kl)
        #     # kl_div = kl.kl_divergence(prior_kl, postr_kl)
        #     A = postr_sp.rsample()
        #     # print('postr samples: ', A)
        # else:
        #     prior_sp = torch.distributions.beta.Beta(prior_alpha, prior_alpha.sum() - prior_alpha)
        #     A = postr_sp.sample()
        #     # print('prior samples: ', A)

        # kl_div = kl.kl_divergence(postr_kl, prior_kl)
        # kl_div = kl.kl_divergence(prior_kl, postr_kl)
        # A = 0
        # for i in range(self.num_samples):
        #     A += postr_sp.rsample()
        # A /= self.num_samples
        # A = postr_sp.rsample()
        # print('postr samples: ', A)

        # print('max sample', torch.max(A))
        # print('min sample', torch.min(A))

        # A = prior_sp.rsample()

        # print('samples: ', A)
        # print('max sample', torch.max(A))
        # print('min sample', torch.min(A))

        # if positive
        # A, h = self.attention_net(h)

        # A = torch.transpose(A, 1, 0)  # KxN 

        # A = F.softmax(A, dim=1)  # softmax over N

        # M = torch.mm(A, h_)
        M = torch.mm(A, h) / A.sum()

        logits = self.classifiers(M)

        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1)
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        if not validation:
            return top_instance, Y_prob, Y_hat, kl_div, y_probs, A
        else:
            return top_instance, Y_prob, Y_hat, y_probs, A


class probabilistic_MIL_Bayes_spvis(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(probabilistic_MIL_Bayes_spvis, self).__init__()

        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        ### for the convolution operation ####
        self.conv1 = nn.Conv2d(size[0], size[1],  1, padding=0)
        self.conv2a = Conv2dVDO(size[1], size[2],  1, padding=0, ard_init=-1.)
        self.conv2b = Conv2dVDO(size[1], size[2],  1, padding=0, ard_init=-1.)

        self.conv3 = Conv2dVDO(size[2], 2,  1, padding=0, ard_init=-1.)
        self.gaus_smoothing = GaussianSmoothing(1, 7, 1)
        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-3.)

        # #### use MLP instead ####
        # self.conv1 = nn.Linear(size[0], size[1])
        # self.conv2a = LinearVDO(size[1], size[2], ard_init=-1.)
        # self.conv2b = LinearVDO(size[1], size[2], ard_init=-1.)

        # self.conv3 = LinearVDO(size[2], 2, ard_init=-1.)
        # self.gaus_smoothing = GaussianSmoothing(1, 7, 1)
        # self.classifiers = LinearVDO(size[1], n_classes, ard_init=-3.)

        self.dp_0 = nn.Dropout(0.25)
        self.dp_a = nn.Dropout(0.25)
        self.dp_b = nn.Dropout(0.25)

        initialize_weights(self)
        self.top_k = top_k

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = self.conv1.to(device)
        self.conv2a = self.conv2a.to(device)
        self.conv2b = self.conv2b.to(device)
        self.conv3 = self.conv3.to(device)
        # self.conv3b = self.conv3b.to(device)
        self.dp_0 = self.dp_0.to(device)
        self.dp_a = self.dp_a.to(device)
        self.dp_b = self.dp_b.to(device)
        self.gaus_smoothing = self.gaus_smoothing.to(device)

        self.classifiers = self.classifiers.to(device)
        # self.temperature = self.temperature.to(device)

    def forward(self, h, validation=False):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK      
        h = h.float().unsqueeze(0)

        # comment this if use MLP
        h = h.permute(0, 3, 1, 2)

        h = F.relu(self.dp_0(self.conv1(h)))

        feat_a = self.dp_a(torch.sigmoid(self.conv2a(h)))

        feat_b = self.dp_b(torch.tanh(self.conv2b(h)))

        feat = feat_a.mul(feat_b)
        params = self.conv3(feat)

        mu = params[:, :1, :, :]
        logvar = params[:, 1:, :, :]

        # #### use MLP instead ####
        # mu = params[:, :, :, :1]
        # logvar = params[:, :, :, 1:]

        # mu = F.pad(mu, (3, 3, 3, 3), mode='constant', value=0)
        # mu = self.gaus_smoothing(mu)

        gaus_samples = self.reparameterize(mu, logvar)
        A = F.sigmoid(gaus_samples)
        M = A.mul(h).sum(dim=(2, 3)) / A.sum()

        # #### use MLP instead ####
        # M = A.mul(h).sum(dim=(1, 2)) / A.sum()

        # print(M.sum())
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

        return top_instance, Y_prob, Y_hat, y_probs, A.view((1,-1))

class probabilistic_MIL_Bayes_convis(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(probabilistic_MIL_Bayes_convis, self).__init__()

        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        # fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        # if dropout:
        #     fc.append(nn.Dropout(0.25))
        # if gate:
        #     attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        # else:
        #     attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        # fc.append(attention_net)
        # self.attention_net = nn.Sequential(*fc)

        # self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        # size = self.size_dict[size_arg]

        self.conv11 = Conv2dVDO(size[0], size[1],  3, padding=1, ard_init=-1.)
        # self.conv12 = Conv2dVDO(size[0], size[1],  7, padding=3, ard_init=-3.)
        self.conv13 = Conv2dVDO(size[0], size[1], 11, padding=5, ard_init=-1.)

        self.conv2a1 = Conv2dVDO(size[1], size[2],  3, padding=1, ard_init=-1.)
        # self.conv2a2 = Conv2dVDO(size[1], size[2],  7, padding=3, ard_init=-3.)
        self.conv2a3 = Conv2dVDO(size[1], size[2], 11, padding=5, ard_init=-1.)

        self.conv2b1 = Conv2dVDO(size[1], size[2],  3, padding=1, ard_init=-1.)
        # self.conv2b2 = Conv2dVDO(size[1], size[2],  7, padding=3, ard_init=-3.)
        self.conv2b3 = Conv2dVDO(size[1], size[2], 11, padding=5, ard_init=-1.)

        self.conv3a1 = Conv2dVDO(size[2], 1,  3, padding=1, ard_init=-1.)
        # self.conv3a2 = Conv2dVDO(size[2], 1,  7, padding=3, ard_init=-3.)
        self.conv3a3 = Conv2dVDO(size[2], 1, 11, padding=5, ard_init=-1.)

        self.conv3b1 = Conv2dVDO(size[2], 1,  3, padding=1, ard_init=-1.)
        # self.conv3b2 = Conv2dVDO(size[2], 1,  7, padding=3, ard_init=-3.)
        self.conv3b3 = Conv2dVDO(size[2], 1, 11, padding=5, ard_init=-1.)

        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-3.)
        self.n_classes = n_classes
        self.print_sample_trigger = False
        self.num_samples = 16
        self.temperature = torch.tensor([1.0])
        self.fixed_b = torch.tensor([5.], requires_grad=False)

        initialize_weights(self)
        self.top_k=top_k

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv11 = self.conv11.to(device)
        # self.conv12 = self.conv12.to(device)
        self.conv13 = self.conv13.to(device)

        self.conv2a1 = self.conv2a1.to(device)
        # self.conv2a2 = self.conv2a2.to(device)
        self.conv2a3 = self.conv2a3.to(device)

        self.conv2b1 = self.conv2b1.to(device)
        # self.conv2b2 = self.conv2b2.to(device)
        self.conv2b3 = self.conv2b3.to(device)

        self.conv3a1 = self.conv3a1.to(device)
        # self.conv3a2 = self.conv3a2.to(device)
        self.conv3a3 = self.conv3a3.to(device)

        self.conv3b1 = self.conv3b1.to(device)
        # self.conv3b2 = self.conv3b2.to(device)
        self.conv3b3 = self.conv3b3.to(device)

        self.classifiers = self.classifiers.to(device)
        self.temperature = self.temperature.to(device)

    def forward(self, h, validation=False):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK      
        h = h.float().unsqueeze(0)
        h = h.permute(0, 3, 1, 2)

        # h = F.relu(torch.nn.functional.dropout(self.conv11(h), p=0.25) + 
        #     torch.nn.functional.dropout(self.conv12(h),p=0.25) + 
        #     torch.nn.functional.dropout(self.conv13(h),p=0.25))

        # feat_a = F.sigmoid(self.conv2a1(h) + self.conv2a2(h) + self.conv2a3(h))

        # feat_b = F.tanh(self.conv2b1(h) + self.conv2b2(h) + self.conv2b3(h))

        # feat = feat_a.mul(feat_b)
        # mu = self.conv3a1(feat) + self.conv3a2(feat) + self.conv3a3(feat)
        # logvar = self.conv3b1(feat) + self.conv3b2(feat) + self.conv3b3(feat)


        h = F.relu(torch.nn.functional.dropout(self.conv11(h), p=0.25) + torch.nn.functional.dropout(self.conv13(h),p=0.25))

        feat_a = F.sigmoid(self.conv2a1(h) + self.conv2a3(h))

        feat_b = F.tanh(self.conv2b1(h) + self.conv2b3(h))

        feat = feat_a.mul(feat_b)
        mu = self.conv3a1(feat) + self.conv3a3(feat)
        logvar = self.conv3b1(feat) + self.conv3b3(feat)

        # A, h = self.attention_net(h)

        # mu = A[:, 0]
        # logvar = A[:, 1]
        gaus_samples = self.reparameterize(mu, logvar)
        A = F.sigmoid(gaus_samples)

        M = A.mul(h).sum(dim=(2,3)) / A.sum()

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
    if isinstance(module, LinearVDO) or isinstance(module, Conv2dVDO): return reg + module.get_reg()
    if hasattr(module, 'children'): return reg + sum([get_ard_reg_vdo(submodule) for submodule in module.children()])
    return reg

bMIL_model_dict = {
                    'A': probabilistic_MIL_Bayes,
                    'F': probabilistic_MIL_Bayes_fc,
                    'vis': probabilistic_MIL_Bayes_vis,
                    'enc': probabilistic_MIL_Bayes_enc,
                    'spvis': probabilistic_MIL_Bayes_spvis,
                    'convis': probabilistic_MIL_Bayes_convis,
}


