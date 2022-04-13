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

class ShampooAttackViT(Attack):
    '''
    Attach Vision Transformer using Shampoo Optimizer (https://arxiv.org/pdf/1802.09568.pdf)
    We formulate the problem by abstracting the image as a set of parameters
    For each patch in the image, we intialize a separate Precondition Tensor H, analogous to the per-layer H in neural networks
    '''
    def __init__(self, model, image_dim, patch_sidelen, lr=1e-1, momentum=0, weight_decay=0, epsilon=1e-4, update_freq=1):
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
        # in total self.patch_num_side ** 2 patches
    

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)
        
        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)
        #outlabels = torch.argmax(outputs)

        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images,
                retain_graph=False, create_graph=False)[0]

        # split images into patches
        # for each image, 
        #   we need n = self.patch_num_w * self.patch_num_h patches, 
        #   and thus n sets of L and R preconditioners

        # initializing precondition matrices
        preconds = torch.zeros(images.shape[0], self.patch_num_side, self.patch_num_side, 6, self.patch_sidelen, self.patch_sidelen)
        for i in range(images.shape[0]):
            for j in range(self.patch_num_side):
                for k in range(self.patch_num_side):
                    preconds[i][j][k][0] = self.epsilon * torch.eye(self.patch_sidelen)    # initialize L Red
                    preconds[i][j][k][1] = self.epsilon * torch.eye(self.patch_sidelen)    # initialize L Green
                    preconds[i][j][k][2] = self.epsilon * torch.eye(self.patch_sidelen)    # initialize L Blue
                    preconds[i][j][k][3] = self.epsilon * torch.eye(self.patch_sidelen)    # initialize R Red
                    preconds[i][j][k][4] = self.epsilon * torch.eye(self.patch_sidelen)    # initialize R Green
                    preconds[i][j][k][5] = self.epsilon * torch.eye(self.patch_sidelen)    # initialize R Blue
        
        advimages = images.clone().detach().to(self.device)
        print(advimages.shape)

        print(f"grad shape:{grad.shape}")
        grad = grad.permute(0, 2, 3, 1)
        print(f"grad shape permuted:{grad.shape}")

        for i in range(images.shape[0]):
            for j in range(self.patch_num_side):
                for k in range(self.patch_num_side):
                    gradient_block = grad[i, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen, :]
                    preconds[i][j][k][0] = preconds[i][j][k][0] + gradient_block[:, :, 0] @ gradient_block[:, :, 0].t()   # update L Red
                    preconds[i][j][k][1] = preconds[i][j][k][1] + gradient_block[:, :, 1] @ gradient_block[:, :, 1].t()   # update L Green
                    preconds[i][j][k][2] = preconds[i][j][k][2] + gradient_block[:, :, 2] @ gradient_block[:, :, 2].t()   # update L Blue
                    preconds[i][j][k][3] = preconds[i][j][k][3] + gradient_block[:, :, 0].t() @ gradient_block[:, :, 0]     # update R Red
                    preconds[i][j][k][4] = preconds[i][j][k][4] + gradient_block[:, :, 1].t() @ gradient_block[:, :, 1]     # update R Green
                    preconds[i][j][k][5] = preconds[i][j][k][5] + gradient_block[:, :, 2].t() @ gradient_block[:, :, 2]     # update R Blue
                    # print(advimages[i, 0, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen].shape)
                    # print((self.epsilon * _matrix_power(preconds[i][j][k][0], -1/4) @ gradient_block[:, :, 0] @ _matrix_power(preconds[i][j][k][3], -1/4)).shape)
                    advimages[i, 0, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] = \
                        advimages[i, 0, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] \
                        + self.epsilon * _matrix_power(preconds[i][j][k][0], -1/4) @ gradient_block[:, :, 0] @ _matrix_power(preconds[i][j][k][3], -1/4)     # update Red Channel
                    print(advimages[i, 0, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen])
                    print(_matrix_power(preconds[i][j][k][0], -1/4) @ gradient_block[:, :, 0] @ _matrix_power(preconds[i][j][k][3], -1/4))
                    advimages[i, 1, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] = \
                        advimages[i, 1, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] \
                        + self.epsilon * _matrix_power(preconds[i][j][k][1], -1/4) @ gradient_block[:, :, 1] @ _matrix_power(preconds[i][j][k][4], -1/4)     # update Green Channel
                    advimages[i, 2, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] = \
                        advimages[i, 2, j*self.patch_sidelen:(j+1)*self.patch_sidelen, k*self.patch_sidelen:(k+1)*self.patch_sidelen] \
                        + self.epsilon * _matrix_power(preconds[i][j][k][2], -1/4) @ gradient_block[:, :, 2] @ _matrix_power(preconds[i][j][k][5], -1/4)     # update Blue Channel
        
        return advimages