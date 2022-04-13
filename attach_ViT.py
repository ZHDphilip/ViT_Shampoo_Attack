from shampoo_attack_ViT import ShampooAttackViT

import torch
import torch.nn as nn

from PIL import Image
import timm
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def main():
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    img = Image.open("../test.jpeg")
    inputs = transform(img)[None, ]
    stackedinputs = torch.stack([inputs, inputs]).squeeze(dim=1)
    out = model(stackedinputs)
    clsidx = torch.argmax(out, dim=1)
    print(f"original prediction: {clsidx}")
    atk = ShampooAttackViT(model, 224, 16, epsilon=1)
    advinputs = atk(stackedinputs, clsidx)
    out2 = model(advinputs)
    clsidx2 = torch.argmax(out2, dim=1)
    print(f"new prediction: {clsidx2}")


if __name__ == "__main__":
    main()