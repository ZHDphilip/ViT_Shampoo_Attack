import time

import argparse

import ml_collections

from shampoo_attack_ViT import ShampooAttackViT
from shampoo_attack_ViT_optimized import ShampooAttackViT_opt

from models.modeling import VisionTransformer

from torchvision.utils import save_image
import torchattacks
import os

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from PIL import Image
# import timm
import torchvision
import torchvision.transforms as transforms
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import matplotlib.pyplot as plt

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def run(benchmark, eps, max_img):
    # for ImageNet pretrained Deit
    # model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()
    # model.eval()

    # for CIFAR10 Pretrained ViT
    configs = get_b16_config()
    model = VisionTransformer(configs, 224, zero_head=True, num_classes=10)
    model.load_state_dict(torch.load("../cifar10-100_500_checkpoint.pt"))
    model.to("cuda:0")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # imagenet
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # cifar10
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4
    
    # load CIFAR10
    trainset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    print(os.listdir("../original_img"))
    print([name for name in os.listdir("../original_img") if "png" in name])
    namecounter = len([name for name in os.listdir("../original_img") if "png" in name])

    # initialize Shampoo Attacker
    for e in eps:
      # for s in [1/255, 2/255, 4/255]:
      s = e / 4
      print(f"experiment with epsilon = {e}, step size = {s}")
      if benchmark:
          atk = torchattacks.PGD(model, eps=e, alpha=s, steps=10)
      else:
          atk = ShampooAttackViT_opt(model, 224, 16, steps=10, perturb_bound=e, lr=s)

      success = 0
      num_examples = 0

      mean = []
      std = []

      for data, label in tqdm(trainloader):
          data = data.to("cuda:0")
          advinputs = atk(data, label)
          # print(data)
          # print(advinputs)
          # exit()
          mean.append(torch.mean(advinputs-data).cpu())
          std.append(torch.std(advinputs-data).cpu())
          Maxdiff = torch.max(advinputs-data)

          adv_output = model(advinputs)
          adv_labels = torch.argmax(adv_output, dim=1)

          num_examples += len(label)
          success += torch.sum(adv_labels.cuda()!=label.cuda()).item()
          print(f"Generated {num_examples} adversarial images, success in {success}, diff mean {mean[-1]}, diff std {std[-1]}, max {Maxdiff}\n")
          
          for i in range(len(label)):
            if label[i] == adv_labels[i]:
              print(f"failed on an image of {classes[label[i]]}")
              save_image(data[i].cpu(), "../original_img/"+str(namecounter)+classes[label[i]]+".png")
              print(f"saving img NO. {namecounter}")
              save_image(advinputs[i].cpu(), "../adv_img/"+str(namecounter)+classes[adv_labels[i]]+".png")
              namecounter += 1
          
          if num_examples >= max_img:
            break
      print(f"Metrics: \n Attack Success Rate for epsilon {e}, step size {s} = {success / num_examples}, diff mean: {np.mean(mean)}")


    # img = Image.open("../test.jpeg")
    # inputs = transform(img)[None, ]
    # stackedinputs = torch.stack([inputs, inputs, inputs, inputs, inputs, inputs, inputs, inputs]).squeeze(dim=1).cuda()
    # out = model(stackedinputs)
    # clsidx = torch.argmax(out, dim=1)
    # # print(f"original prediction: {clsidx}")
    # atk = ShampooAttackViT_opt(model, 224, 16, steps=20, perturb_bound=2/255)
    # atk_vanilla = ShampooAttackViT(model, 224, 16, steps=20, perturb_bound=2/255)
    # s_atk = time.time()
    # advinputs, original_input = atk(stackedinputs, clsidx)
    # e_atk = time.time()
    # advinputs_vanilla, original_input_vanilla = atk_vanilla(stackedinputs, clsidx)
    # print(f"vanilla attack for 4 img costs {time.time()-e_atk}")
    # print(f"attack for 4 img costs {e_atk-s_atk}")
    # # print(torch.eq(inputs, original_input))
    # out2 = model(advinputs)
    # clsidx2 = torch.argmax(out2, dim=1)
    # out2_vanilla = model(advinputs_vanilla)
    # clsidx2_vanilla = torch.argmax(out2_vanilla, dim=1)
    # print(f"opt: original prediction: {clsidx}, new prediction: {clsidx2}")
    # print(f"vanilla: original prediction: {clsidx}, new prediction: {clsidx2_vanilla}")
    # print(out)
    # print(out2)
    # print(out2_vanilla)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--PGD", type=int, default=0, help="to use PGD bench mark or not")
    parser.add_argument("--eps_low", type=float, default=2, help="test eps starts from arg/255")
    parser.add_argument("--nstep", type=int, default=3, \
      help="number of steps to increase eps for testing, eg eps_low=2/255 nstep=3 will test on eps=[2/255,4/255,6/255]")
    parser.add_argument("--max_img", type=int, default=10000, help="number of images to attack for each set of parameter")
    parser.add_argument("--iter", type=int, default=10, help="number of iterations in PGD or Shampoo attack")
    args = parser.parse_args()
    benchmark = False
    if args.PGD == 1:
      benchmark = True
    eps = [(args.eps_low/255)*i for i in range(1,args.nstep+1)]
    run(benchmark, eps, args.max_img)