""" PyTorch Lamb optimizer w/ behaviour similar to NVIDIA FusedLamb

This optimizer code was adapted from the following (starting with latest)
* https://github.com/HabanaAI/Model-References/blob/2b435114fe8e31f159b1d3063b8280ae37af7423/PyTorch/nlp/bert/pretraining/lamb.py
* https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py
* https://github.com/cybertronai/pytorch-lamb

Use FusedLamb if you can (GPU). The reason for including this variant of Lamb is to have a version that is
similar in behaviour to APEX FusedLamb if you aren't using NVIDIA GPUs or cannot install/use APEX.

In addition to some cleanup, this Lamb impl has been modified to support PyTorch XLA and has been tested on TPU.

Original copyrights for above sources are below.

Modifications Copyright 2021 Ross Wightman
"""
# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) 2019 cybertronai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math

import torch
from torch.optim import Optimizer
from timm.utils.castTF32 import cast_fp32_tf32, cast_fp32_tf32_float,cast_fp32_tf32_gpu, cast_fp32_tf32_inplaceV2


class Lamb(Optimizer):
    """Implements a pure pytorch variant of FuseLAMB (NvLamb variant) optimizer from apex.optimizers.FusedLAMB
    reference: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py

    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm (default: 1.0)
        trust_clip (bool): enable LAMBC trust ratio clipping (default: False)
        always_adapt (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)

    .. _Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
            self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-6,
            weight_decay=0.01, grad_averaging=True, max_grad_norm=1.0, trust_clip=False, always_adapt=False):
        # init-weights cast
        # lr = cast_fp32_tf32_float(lr) #AttributeError: 'float' object has no attribute 'view'  0.008 -> 0.008
        # eps = cast_fp32_tf32_float(eps) #1e-06 -> 9.9931e-07
        # betas = (cast_fp32_tf32_float(betas[0]), cast_fp32_tf32_float(betas[1])) #(0.9, 0.999) -> (0.8999, 0.9985)
        # weight_decay = cast_fp32_tf32_float(weight_decay)
        # max_grad_norm = cast_fp32_tf32_float(max_grad_norm)

        defaults = dict(
            lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay,
            grad_averaging=grad_averaging, max_grad_norm=max_grad_norm,
            trust_clip=trust_clip, always_adapt=always_adapt)
        # print(defaults) 
        # # {'lr': 0.008, 'bias_correction': True, 'betas': (0.9, 0.999), 'eps': 1e-06, 'weight_decay': 0.0, 'grad_averaging': True, 'max_grad_norm': 1.0, 'trust_clip': False, 'always_adapt': False}
        # # {'lr': 0.00800323486328125, 'bias_correction': True, 'betas': (0.89990234375, 0.9990234375), 'eps': 1.000240445137024e-06, 'weight_decay': 0.0, 'grad_averaging': True, 'max_grad_norm': 1.0, 'trust_clip': False, 'always_adapt': False}
        # input('---stop here---')    
        ''' len(params)=2 /timm/optim/optim_factory.py add_weight_decay() line38
        [{'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]'''

        super().__init__(params, defaults)  

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        one_tensor = torch.tensor(1.0, device=device)  # because torch.where doesn't handle scalars correctly
        global_grad_norm = torch.zeros(1, device=device)
        count=0
        # f=open('./logs/optimizer_grad.log','w+')
        for i,group in enumerate(self.param_groups): #len(self.param_groups)=2
            # f=open('./logs/Optimizer_param_groups.log','w+')
            # f.write(str(group))
            # f.close()
            # input('stop here')
            for p in group['params']: #p already cast in backward
                if p.grad is None:
                    count +=1
                    continue
                grad = p.grad
                # print("-----Optimizer Grad-----",grad.shape,grad)
                # grad_str= str(grad.shape)+str(grad)+'\n'
                # f.write(grad_str)
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')
                global_grad_norm.add_(grad.pow(2).sum())
            # print("OPTIMIZER Params:",i,len(group['params']), count)
        #     f.write("\n---NEXT---\n")
        # f.close()

        global_grad_norm = torch.sqrt(global_grad_norm)
        # add for cast
        # global_grad_norm = cast_fp32_tf32(global_grad_norm) 

        # FIXME it'd be nice to remove explicit tensor conversion of scalars when torch.where promotes
        # scalar types properly https://github.com/pytorch/pytorch/issues/9190
        max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device) #1.0
        clip_global_grad_norm = torch.where(
            global_grad_norm > max_grad_norm,
            global_grad_norm / max_grad_norm,
            one_tensor) 
        # add for cast
        # clip_global_grad_norm = cast_fp32_tf32(clip_global_grad_norm) 

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            if bias_correction:
                bias_correction1 = 1 - beta1 ** group['step']
                bias_correction2 = 1 - beta2 ** group['step']
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0
            # add for cast
            # bias_correction1 = cast_fp32_tf32_float(bias_correction1) 
            # bias_correction2 = cast_fp32_tf32_float(bias_correction2)

            # f_opt=open('./logs/opt/opt_verify.log','w+')
            for index,p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.div_(clip_global_grad_norm)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=beta3)  # m_t    calculate inplace
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t
                # add for cast
                # exp_avg = cast_fp32_tf32(exp_avg) 
                # exp_avg_sq = cast_fp32_tf32(exp_avg_sq) 

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                # add for cast
                # denom = cast_fp32_tf32(denom) 

                update = (exp_avg / bias_correction1).div_(denom) #rt
                # add for cast
                # update = cast_fp32_tf32(update) 

                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay) #rt + \lambda \theta_{t-1}
                # add for cast
                # update = cast_fp32_tf32(update) 

                if weight_decay != 0 or group['always_adapt']: #group['always_adapt']=False
                    # Layer-wise LR adaptation. By default, skip adaptation on parameters that are
                    # excluded from weight decay, unless always_adapt == True, then always enabled.
                    w_norm = p.norm(2.0) #2-norm  
                    g_norm = update.norm(2.0) 
                    # FIXME nested where required since logical and/or not working in PT XLA
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, w_norm / g_norm, one_tensor),
                        one_tensor,
                    )
                    if group['trust_clip']: #group['trust_clip']=False
                        # LAMBC trust clipping, upper bound fixed at one
                        trust_ratio = torch.minimum(trust_ratio, one_tensor)
                    update.mul_(trust_ratio)
                    # add for cast
                    # update = cast_fp32_tf32(update) 

                p.add_(update, alpha=-group['lr'])
                # add for cast
                # print(p.device) # cuda:0
                # if(index==0):
                #     opt_str = 'in lamb before cast:' + str(p) +'\n'
                #     f_opt.write(opt_str)
                cast_fp32_tf32_inplaceV2(p) ###wrong
                # if(index==0):
                #     opt_str = 'in lamb before cast:' + str(p) +'\n'
                #     f_opt.write(opt_str)
                # print(p.device) # cuda:0
                # input('stop here')
            # f_opt.close()

        return loss
