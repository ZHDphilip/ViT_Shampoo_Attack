from shampoo_attack_ViT import ShampooAttackViT
from shampoo_attack_ViT_Tensor import ShampooAttackViT_Tensor

import torch
import torch.nn as nn

from tqdm import tqdm

from PIL import Image
import timm
import torchvision
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import matplotlib.pyplot as plt

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def main():
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # batch_size = 4

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                         shuffle=True, num_workers=2)

    # # initialize Shampoo Attacker
    # atk = ShampooAttackViT(model, 224, 16, steps=20, perturb_bound=0.01)

    # success = 0
    # num_examples = 0

    # for data, label in tqdm(trainloader):
    #     advinputs, original_input = atk(data, label)
    #     adv_output = model(advinputs)
    #     adv_labels = torch.argmax(adv_output, dim=1)
    #     num_examples += len(label)
    #     success += torch.sum(adv_labels!=label).item()
    # print(f"Metrics: \n Attack Success Rate = {success / num_examples}")


    img = Image.open("../test.jpeg")
    inputs = transform(img)[None, ]
    stackedinputs = torch.stack([inputs, inputs]).squeeze(dim=1)
    out = model(stackedinputs)
    clsidx = torch.argmax(out, dim=1)
    # print(f"original prediction: {clsidx}")
    atk = ShampooAttackViT_Tensor(model, 224, 16, steps=20, perturb_bound=0.05)
    advinputs, original_input = atk(stackedinputs, clsidx)
    # print(torch.eq(inputs, original_input))
    out2 = model(advinputs)
    clsidx2 = torch.argmax(out2, dim=1)
    print(f"original prediction: {clsidx}, new prediction: {clsidx2}")

    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(inputs[0].permute(1, 2, 0))
    # axarr[1].imshow(advinputs[0].permute(1, 2, 0))
    # plt.show()


if __name__ == "__main__":
    main()