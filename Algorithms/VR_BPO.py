import collections
import copy

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage.misc import tensor_utils
# from garage.np.algos import BatchPolopt
from .VPG import VPG
from garage.sampler import OnPolicyVectorizedSampler
from garage.torch import (compute_advantages, filter_valids, pad_to_last)
from garage.torch.optimizers import OptimizerWrapper

from garage import log_performance, TrajectoryBatch
from ._utils import CosLR

from .special import *
#from garage.torch.algos._utils import (_Default, compute_advantages, filter_valids,
 #                               make_optimizer, pad_to_last)
from ._utils import (compute_advantages, filter_valids, make_optimizer, pad_to_last, flatten_batch)
from tensorboardX import SummaryWriter
#from garage.torch.utils import flatten_batch
from garage.np.optimizers import BatchDataset
import time
import math

def normalize_gradient(grad):
    n_grad = grad/grad.norm(p=2)
    #print(n_grad.norm(p=2))
    return n_grad
timestamp = time.time()
timestruct = time.localtime(timestamp)
#exp_time = time.strftime('%Y-%m-%d %H:%M:%S', timestruct)
def compute_weights(o_lh, c_lh, th=1.99):
    #mini_bs = o_lh.size()
    lth = th - 1

    o_lh_sum = o_lh
    c_lh_sum = c_lh

    weight = (o_lh_sum - c_lh_sum).exp()
    # weight = torch.exp(weight)
    weight[weight>=th] = th
    weight[weight<lth] = lth
    # print(weight.max())
    # return weigt.unsqueeze(1).expand_as(c_lh)
    return weight

def kl_projection(x_t, t, grad, p=None):
    #convert to simplex
    sign_x = torch.sign(x_t)
    abs_x = torch.abs(x_t)
    simplex_x = abs_x/abs_x.sum()

    #convert to simplex grad
    simplex_grad = sign_x*grad
    # /abs_x.sum()

    x_t1 = simplex_x*torch.exp(-t*simplex_grad)
    x_t1 = x_t1/x_t1.sum()
    #convert back
    x_t1 = x_t1*sign_x*abs_x.sum()
    return x_t1

def pn_link(x, p=2):
    sign_x = torch.sign(x)
    out = sign_x*x.abs().pow(p-1)
    out = out/x.norm(p).pow(p-2)

    return out

def lp_projection(x_t, t, grad, p=2):
    q = p/(p-1)
    p_x = pn_link(x_t,p)
    out = pn_link(p_x-t*grad,q)
    return out

def diag_projection(x_t, t, grad, F=None):
    out = x_t-t*grad/(F.sqrt()+1e-8)
    return out

def diag_projection_id(x_t, t, grad, F=None):
    out = x_t-t*grad/(F+1e-8)
    return out

class VR_BGPO(VPG):
    def __init__(
            self,
            env_spec,
            policy,
            value_function,
            vf_optimizer=None,
            dist_type='KL',
            dist_pow = None,
            policy_lr=1e-2,
            vf_lr = 2.5e-4,
            w = 10,
            c = 100,
            minibatch_size=64,
            vf_minibatch_size=160,
            max_optimization_epochs =10,

            #hyperparameters for HAP
            lam = 0.1,
            grad_factor = 10,
            th = 1.2,

            loss_clip=False,

            max_path_length=500,
            num_train_per_epoch=1,
            discount=0.99,
            gae_lambda=0.97,
            center_adv=True,
            positive_adv=False,
            policy_ent_coeff=0.0,
            g_max = 0.05,

            use_softplus_entropy=False,
            stop_entropy_gradient=False,

            # star_version=True,
            entropy_method='regularized',
            sch = None,
            log_dir='./log',

    ):

        if vf_optimizer is None:
            vf_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=vf_lr)),
                value_function,
                max_optimization_epochs=10,
                minibatch_size=vf_minibatch_size)

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         vf_optimizer=vf_optimizer,
                         max_path_length=max_path_length,
                         num_train_per_epoch=num_train_per_epoch,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         log_dir=log_dir)

        self._policy_optimizer = None
        self._minibatch_size = minibatch_size
        self._max_optimization_epochs = max_optimization_epochs

        self._eps = 1e-8
        self.g_max = g_max

        self.first_flag = True

        # self.sv = star_version

        self.lr = policy_lr

        self._lr_clip_range = 0.01

        self.w = w
        self.c = c

        self.dict = {}
        self.grad_factor = grad_factor

        self.loss_clip=loss_clip

        self.sampler_cls = OnPolicyVectorizedSampler

        self.lam = lam
        self.steps = 0

        self.th = th



        if sch is not None:
            self.sch = sch
            self.sch.reset()
        else:
            self.sch = None


        if dist_type == 'KL':
            self.projection = kl_projection
        elif dist_type == 'Lp':
            self.projection = lp_projection
            self.pow = dist_pow
        elif dist_type == 'Diag':
            self.projection = diag_projection
            self.beta = 0.999
        self.dist_type = dist_type
        print(self.dist_type)

    def _compute_objective(self, advantages, obs, actions, rewards, policy):
        del rewards
        # log_likelihoods = self.policy(obs)[0].log_prob(actions)
        log_likelihoods = policy(obs)[0].log_prob(actions)
        if self.loss_clip:
            # adv_norm = advantages.detach().norm(2)
            advantages = torch.clamp(advantages,min=0)
            # aft_norm = advantages.detach().norm(2)
            # advantages = advantages*(adv_norm/(aft_norm+1e-8))
        return log_likelihoods * advantages, log_likelihoods

    def _compute_loss_with_adv(self, obs, actions, rewards, advantages, policy=None, current_lh=None, display=True):

        if policy is None:
            policy = self.policy

        objectives, log_likelihoods = self._compute_objective(advantages, obs, actions, rewards, policy)

        if self._entropy_regularzied:
            policy_entropies = self._compute_policy_entropy(obs, policy)
            objectives += self._policy_ent_coeff * policy_entropies

        if display:
            return objectives.mean()

        if current_lh is None:

            # if self.loss_clip and (objectives.mean().detach()<=-3.0):
            #     objectives = torch.clamp(objectives, max=0)
            return objectives.mean(), log_likelihoods
        else:
            old_lh = log_likelihoods.detach()
            correct_w = compute_weights(old_lh, current_lh, th=self.th)
            objectives = correct_w*objectives

            # if self.loss_clip and (objectives.mean().detach()<=-3.0):
            #     objectives = torch.clamp(objectives, max=0)

            return objectives.mean()

    def _train_policy(self, obs, actions, rewards, advantages):

        loss, current_likelihoods = self._compute_loss_with_adv(obs, actions, rewards, advantages, self.policy,display=False)



        loss.backward()

        grad = self.policy.get_grads()



        old_loss = self._compute_loss_with_adv(obs, actions, rewards, advantages, self._old_policy, current_likelihoods.detach(),display=False)

        # if self.loss_clip:
        #     old_loss = loss = torch.clamp(old_loss, max=0)

        old_loss.backward()
        old_grad = self._old_policy.get_grads()

        # self._old_policy.set_param_values(self.policy.get_param_values())

        self.vr_bpo_step(grad, old_grad, self.lr)

        return loss

    def vr_bpo_step(self, grads, old_grads, lr):
        G_p = grads.norm(p=2)
        eta_t = lr / ((self.w + self.grad_factor * self.steps) ** (1 / 3))

        if G_p < self.g_max:
            g_max = G_p.item()
        else:
            g_max = self.g_max

        if 'u_buffer' not in self.dict:
            u_buffer = self.dict['u_buffer'] = torch.zeros_like(grads)
        else:
            u_buffer = self.dict['u_buffer']

        if self.dist_type == 'Diag':
            if 'grad_rmean' not in self.dict:
                grad_rmean = self.dict['grad_rmean'] = torch.zeros_like(grads)
            else:
                grad_rmean = self.dict['grad_rmean']

        if self.first_flag:
            u = grads
            if 'grad_rmean' in self.dict:
                grad_rmean = u.pow(2)
        else:
            u = -self.a.item()*grads + (1 - self.a.item()) * (u_buffer - grads + old_grads)
            if 'grad_rmean' in self.dict:
                grad_rmean = self.beta*grad_rmean + (1-self.beta)*u.pow(2)


        self.dict['u_buffer'].copy_(u)
        self.steps += 1

        u = torch.clamp(u, -g_max, g_max)

        params = self.policy.get_param_values()

        if self.dist_type == 'Diag':
            self.dict['grad_rmean'].copy_(grad_rmean)

            hat_grad_rmean = grad_rmean/(1-math.pow(self.beta, self.steps))
            params_hat = self.projection(params, self.lam, u, hat_grad_rmean)
        else:
            params_hat = self.projection(params, self.lam, u, self.pow)

        # params_hat = self.projection(params, self.lam, u, self.pow)
        params = params + eta_t * (params_hat - params)
        # params = params - 1e-4 * params

        self.policy.set_param_values(params)


        self.a = torch.min(torch.Tensor([1.0]), torch.Tensor([self.c * (eta_t ** 2)]))
        self.a = torch.max(self.a, torch.Tensor([0.3]))

        self.eta_t = eta_t
        self.first_flag = False

    def _train(self, obs, actions, rewards, returns, advs):

        for dataset in self.get_minibatch(
                obs, actions, rewards, advs):
            self._train_policy(*dataset)
        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._train_value_function(*dataset)

        if self.sch is not None:
            self.lam = self.sch.step()

        print(self.eta_t)
        print(self.a)
        print(self.c*(self.eta_t**2))
        print(self.lam)
    def get_minibatch(self, *inputs):
        batch_dataset = BatchDataset(inputs, self._minibatch_size)

        for _ in range(self._max_optimization_epochs):
            for dataset in batch_dataset.iterate():
                yield dataset

    def _compute_policy_entropy(self, obs, policy=None):

        if policy is None:
            policy = self.policy
        else:
            policy = policy

        if self._stop_entropy_gradient:
            with torch.no_grad():
                policy_entropy = policy(obs)[0].entropy()
        else:
            policy_entropy = policy(obs)[0].entropy()

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy
