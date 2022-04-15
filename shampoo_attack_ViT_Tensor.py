import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import torch
import torch.nn as nn

from attack import Attack

def _matrix_power(matrix, power):
    # use CPU for svd for speed up
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t())#.cuda()

def apply_reshape_forward(tensor, patch_num_side, patch_sidelen):
    return tensor.view(-1, 3, patch_num_side, patch_sidelen, patch_num_side, patch_sidelen).permute(0, 1, 2, 4, 3, 5)

def apply_reshape_backward(tensor, patch_num_side, patch_sidelen):
    return tensor.permute(0, 1, 2, 4, 3, 5).view(-1, 3, patch_num_side*patch_sidelen, patch_num_side*patch_sidelen)

class ShampooAttackViT_Tensor(Attack):
    '''
    Attach Vision Transformer using Shampoo Optimizer (https://arxiv.org/pdf/1802.09568.pdf)
    We formulate the problem by abstracting the image as a set of parameters
    For each patch in the image, we intialize a separate Precondition Tensor H, analogous to the per-layer H in neural networks
    '''
    def __init__(self, model, image_dim, patch_sidelen, lr=1e-1, momentum=0, weight_decay=0, epsilon=0.007, update_freq=1, steps=40, perturb_bound=0.05):
        assert image_dim % patch_sidelen == 0
        super().__init__("ShampooAttachViT", model)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.update_freq = update_freq

        self._supported_mode = ['default', 'targeted']
        self.image_dim = image_dim
        self.patch_sidelen = patch_sidelen
        self.patch_num_side = int(self.image_dim / self.patch_sidelen)

        self.steps=steps
        self.perturb_bound = perturb_bound
        # in total self.patch_num_side ** 2 patches
    

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)
        
        loss = nn.CrossEntropyLoss()

        advimages = images.clone().detach().to(self.device)
        # attack using Shampoo Optimizer Tensor Case https://arxiv.org/pdf/1802.09568.pdf

        # stacking together the patches
        advimages = apply_reshape_forward(advimages, self.patch_num_side, self.patch_sidelen)
        # print(advimages.shape)

        # the above line is equivalent to the following code chunk
        # new_tensor = torch.zeros_like(advimages)
        # for i in range(images.shape[0]):
        #     for j in range(3):
        #         for k in range(self.patch_num_side):
        #             for x in range(self.patch_num_side):
        #                 new_tensor[i, j, k, x, :, :] = images[i, j, k*self.patch_sidelen:(k+1)*self.patch_sidelen, x*self.patch_sidelen:(x+1)*self.patch_sidelen]

        # # initialize the precond and inv precond matrices
        state = dict()

        for dim_id, dim in enumerate(advimages.size()):
            # precondition matrices
            if dim_id == 0:
                continue # do not need precondition matrix for first dim because it is batch size
            state['precond_{}'.format(dim_id)] = self.epsilon * torch.eye(dim, out=advimages.new(dim, dim))
            state['inv_precond_{dim_id}'.format(dim_id=dim_id)] = advimages.new(dim, dim).zero_()
        
        advimages = apply_reshape_backward(advimages, self.patch_num_side, self.patch_sidelen)

        for s in range(self.steps):
            advimages.requires_grad = True
            outputs = self.model(advimages)
            #outlabels = torch.argmax(outputs)

            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, advimages,
                    retain_graph=False, create_graph=False)[0].to(self.device)
            # print(f"grad shape: {grad.shape}")
            
            advimages.requires_grad = False

            grad = apply_reshape_forward(grad, self.patch_num_side, self.patch_sidelen)
            original_size = grad.size()
            advimages = apply_reshape_forward(advimages, self.patch_num_side, self.patch_sidelen)
            for dim_id, dim in enumerate(original_size):
                if dim_id == 0:
                    continue
                precond = state['precond_{}'.format(dim_id)]
                inv_precond = state['inv_precond_{}'.format(dim_id)]

                # mat_{dim_id}(grad)
                grad = grad.transpose_(0, dim_id).contiguous()
                transposed_size = grad.size()
                grad = grad.view(dim, -1)

                grad_t = grad.t()
                precond.add_(grad @ grad_t)
                if s % self.update_freq == 0:
                    inv_precond.copy_(_matrix_power(precond, -1 / len(grad.size())))

                if dim_id == len(original_size) - 1:
                    # finally
                    grad = grad_t @ inv_precond
                    # grad: (-1, last_dim)
                    grad = grad.view(original_size)
                else:
                    # if not final
                    grad = inv_precond @ grad
                    # grad (dim, -1)
                    grad = grad.view(transposed_size)

            advimages = advimages - self.lr*grad
            # print(f"adv shape: {advimages.shape}")
            # print(f"grad shape: {grad.shape}")
            advimages = apply_reshape_backward(advimages, self.patch_num_side, self.patch_sidelen)
            delta = torch.clamp(advimages-images, min=-self.perturb_bound, max=self.perturb_bound)
            advimages = images + delta
        # # print(advimages-images)
        # # print(torch.max(advimages-images))
        # # assert torch.max(advimages - images) <= self.perturb_bound
        print(advimages-images)
        print(torch.max(advimages-images))
        return advimages, images