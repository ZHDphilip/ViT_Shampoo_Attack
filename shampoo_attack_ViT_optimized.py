import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import torch
import torch.nn as nn

import time

from attack import Attack

def _matrix_power(matrix, power):
    # use CPU for svd for speed up
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t())#.cuda()

def _batch_matrix_power(matrix, power):
    print(matrix.shape)
    ret = torch.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                ret[i,j,k,:,:] = _matrix_power(matrix[i,j,k,:,:], power)
    return ret

def apply_reshape_forward(tensor, patch_num_side, patch_sidelen):
    return tensor.view(-1, 3, patch_num_side, patch_sidelen, patch_num_side, patch_sidelen).permute(0, 1, 2, 4, 3, 5)

def apply_reshape_backward(tensor, patch_num_side, patch_sidelen):
    return tensor.permute(0, 1, 2, 4, 3, 5).view(-1, 3, patch_num_side*patch_sidelen, patch_num_side*patch_sidelen)

class ShampooAttackViT_opt(Attack):
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
        # print(advimages.shape)

        # split images into patches
            # for each image, 
            #   we need n = self.patch_num_w * self.patch_num_h patches, 
            #   and thus n sets of L and R preconditioners
        # attack using Shampoo Optimizer Matrix Case https://arxiv.org/pdf/1802.09568.pdf

        # initializing precondition matrices
        preconds_L = torch.zeros(advimages.shape[0], self.patch_num_side, self.patch_num_side, 3, self.patch_sidelen, self.patch_sidelen).to(self.device)
        preconds_R = torch.zeros(advimages.shape[0], self.patch_num_side, self.patch_num_side, 3, self.patch_sidelen, self.patch_sidelen).to(self.device)

        preconds_L[:,:,:,:] = self.epsilon * torch.eye(self.patch_sidelen)
        preconds_R[:,:,:,:] = self.epsilon * torch.eye(self.patch_sidelen)

        for _ in range(self.steps):
            advimages.requires_grad = True
            outputs = self.model(advimages)
            # for cifar10 model
            # outputs = outputs[0]
            #outlabels = torch.argmax(outputs)
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, advimages,
                    retain_graph=False, create_graph=False)[0].to(self.device)
            # print(f"grad shape: {grad.shape}")
            
            advimages.requires_grad = False

            grad_new = apply_reshape_forward(grad, self.patch_num_side, self.patch_sidelen)
            # print(grad_new.shape)
            original_size = grad.size()
            advimages = apply_reshape_forward(advimages, self.patch_num_side, self.patch_sidelen)

            # update L/R (RGB)
            for i in range(3):
                # tmp = grad_new[:,i,:,:]
                # print(tmp.shape)
                # tmp2 = torch.transpose(grad_new[:,i,:,:], dim0=3, dim1=4)
                # print(tmp2.shape)
                # tmp3 = tmp @ tmp2
                # print(tmp3.shape)
                # tmp4 = tmp[0,0,0,:,:] @ tmp2[0,0,0,:,:]
                preconds_L[:,:,:,i] = preconds_L[:,:,:,i] + grad_new[:,i,:,:] @ torch.transpose(grad_new[:,i,:,:], dim0=3, dim1=4)
                preconds_R[:,:,:,i] = preconds_R[:,:,:,i] + grad_new[:,i,:,:] @ torch.transpose(grad_new[:,i,:,:], dim0=3, dim1=4)
                advimages[:,i,:,:] = advimages[:,i,:,:] + \
                    self.lr * torch.sign(_batch_matrix_power(preconds_L[:,:,:,i], -1/4) @ grad_new[:,i,:,:] @ _batch_matrix_power(preconds_R[:,:,:,i], -1/4))

            advimages = apply_reshape_backward(advimages, self.patch_num_side, self.patch_sidelen)
            # for i in range(advimages.shape[0]):
            #     for j in range(self.patch_num_side):
            #         for k in range(self.patch_num_side):
            #             gradient_block = grad[i, :, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen]
            # #             # print(f"Update Iteration: {_+1}, image NO.{i+1}, patch NO.{j+1}-{k+1} -- {self.patch_num_side}-{self.patch_num_side}")
            # #             preconds_L[i][j][k][0] = preconds_L[i][j][k][0] + gradient_block[0, :, :] @ gradient_block[0, :, :].t()   # update L Red
            # #             preconds_L[i][j][k][1] = preconds_L[i][j][k][1] + gradient_block[1, :, :] @ gradient_block[1, :, :].t()   # update L Green
            # #             preconds_L[i][j][k][2] = preconds_L[i][j][k][2] + gradient_block[2, :, :] @ gradient_block[2, :, :].t()   # update L Blue
            # #             preconds_R[i][j][k][0] = preconds_R[i][j][k][0] + gradient_block[0, :, :].t() @ gradient_block[0, :, :]     # update R Red
            # #             preconds_R[i][j][k][1] = preconds_R[i][j][k][1] + gradient_block[1, :, :].t() @ gradient_block[1, :, :]     # update R Green
            # #             preconds_R[i][j][k][2] = preconds_R[i][j][k][2] + gradient_block[2, :, :].t() @ gradient_block[2, :, :]     # update R Blue
            # #             # print(advimages[i, 0, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen].shape)
            # #             # print((self.epsilon * _matrix_power(preconds[i][j][k][0], -1/4) @ gradient_block[:, :, 0] @ _matrix_power(preconds[i][j][k][3], -1/4)).shape)
            #             advimages[i, 0, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] = \
            #                 advimages[i, 0, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] \
            #                 + self.lr * _matrix_power(preconds_L[i][j][k][0], -1/4) @ gradient_block[0, :, :] @ _matrix_power(preconds_R[i][j][k][0], -1/4)     # update Red Channel
            #             advimages[i, 1, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] = \
            #                 advimages[i, 1, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] \
            #                 + self.lr * _matrix_power(preconds_L[i][j][k][1], -1/4) @ gradient_block[1, :, :] @ _matrix_power(preconds_R[i][j][k][1], -1/4)     # update Green Channel
            #             advimages[i, 2, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] = \
            #                 advimages[i, 2, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] \
            #                 + self.lr * _matrix_power(preconds_L[i][j][k][2], -1/4) @ gradient_block[2, :, :] @ _matrix_power(preconds_R[i][j][k][2], -1/4)     # update Blue Channel
            
            delta = torch.clamp(advimages-images, min=-self.perturb_bound, max=self.perturb_bound)
            advimages = images + delta
        # print(advimages-images)
        # print(torch.max(advimages-images))
        # assert torch.max(advimages - images) <= self.perturb_bound
        return advimages, images