import time 

import ml_collections

import argparse

from shampoo_attack_ViT import ShampooAttackViT
from shampoo_attack_ViT_Tensor import ShampooAttackViT_Tensor
from shampoo_attack_ViT_optimized import ShampooAttackViT_opt

from models.modeling import VisionTransformer

import torch
import torch.nn as nn

from tqdm import tqdm

from PIL import Image
import timm
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import matplotlib.pyplot as plt
import numpy as np

import os, os.path

import json

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

# ImageNet class dict
# jsfile = open("../imagenet_class_index.json")
# classes = json.load(jsfile)
# jsfile.close()

# CIFAR10 class dict
classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def run():
    # ImageNet Pretrained Deit
    # model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    # model.eval()

    # CIFAR10 Pretrained ViT
    configs = get_b16_config()
    model = VisionTransformer(configs, 224, zero_head=True, num_classes=10)
    model.load_state_dict(torch.load("../cifar10-100_500_checkpoint.pt", map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # imagenet
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # cifar10
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4
    
    # load Tiny ImageNet
    # trainset = torchvision.datasets.ImageFolder(root='../Tiny_ImageNet/val', transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            # shuffle=True, num_workers=2)

    # load CIFAR10
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


    # f = open("../attack_soft_output.txt", 'a')

    print(os.listdir("../original_img"))
    print([name for name in os.listdir("../original_img") if "png" in name])
    namecounter = len([name for name in os.listdir("../original_img") if "png" in name])
    # original_count = namecounter
    print(namecounter)

    #  testing version #
    for e in [2/255, 4/255, 8/255]:
      # for s in [1/255, 2/255, 4/255]:
      s = e / 4
      print(f"experiment with epsilon = {e}, step size = {s}")
      atk = ShampooAttackViT_opt(model, 224, 16, steps=10, perturb_bound=e, lr=s)

      success = 0
      num_examples = 0

      mean = []
      std = []

      for data, label in tqdm(trainloader):

          advinputs, original_input = atk(data, label)

          mean.append(torch.mean(advinputs-original_input).cpu())
          std.append(torch.std(advinputs-original_input).cpu())
          Maxdiff = torch.max(advinputs-original_input)

          adv_output = model(advinputs)
          adv_labels = torch.argmax(adv_output, dim=1)

          num_examples += len(label)
          success += torch.sum(adv_labels.cuda()!=label.cuda()).item()
          print(f"Generated {num_examples} adversarial images, success in {success}, diff mean {mean[-1]}, diff std {std[-1]}, max {Maxdiff}\n")
          
          for i in range(len(label)):
            if label[i] == adv_labels[i]:
              print(f"failed on an image of {classes[label[i]]}")
              save_image(original_input[i].cpu(), "../original_img/"+str(namecounter)+classes[label[i]]+".png")
              print(f"saving img NO. {namecounter}")
              save_image(advinputs[i].cpu(), "../adv_img/"+str(namecounter)+classes[adv_labels[i]]+".png")
              namecounter += 1
          
          # if num_examples >= 20:
          #   break
      print(f"Metrics: \n Attack Success Rate for epsilon {e}, step size {s} = {success / num_examples}, diff mean: {np.mean(mean)}")

    # # for matrix case
    # # ======================== #
    # for data, label in tqdm(trainloader):
    #     original_output = model(data)
    #     # for cifar10 ViT
    #     # original_output = original_output[0]
    #     # print(original_output)
    #     original_label = torch.argmax(original_output, dim=1)
    #     adv_inputs, original_input = atk(data, original_label)
    #     adv_output = model(adv_inputs)
    #     print(original_output)
    #     print(adv_output)
    #     # for cifar10 Vit
    #     # adv_output = adv_output[0]
    #     adv_labels = torch.argmax(adv_output, dim=1)
    #     logline = "results for image NO." + str(max(namecounter, original_count)) + " to NO." +str(max(namecounter, original_count)+batch_size)
    #     num_examples += len(label)
    #     success += torch.sum(adv_labels!=original_label).item()
    #     print(f"Generated {num_examples} adversarial images, success in {success}")

    #     gt_label_for_output = [classes[str(i.item())] for i in label]
    #     gt_label_log = "original labels: "
    #     for item in gt_label_for_output:
    #         gt_label_log = gt_label_log + " " + str(item) 

    #     ori_label_for_output = [classes[str(i.item())] for i in original_label]
    #     ori_label_log = "original labels: "
    #     for item in ori_label_for_output:
    #         ori_label_log = ori_label_log + " " + str(item) 
    #     adv_label_for_output = [classes[str(i.item())] for i in adv_labels]
    #     adv_label_log = "adversarial labels: "
    #     for item in adv_label_for_output:
    #         adv_label_log = adv_label_log + " " + str(item) 

    #     # print(gt_label_log)
    #     print(ori_label_log)
    #     print(adv_label_log)
    #     # save soft label output for debug purpose
    #     f.write(logline+'\n')
    #     # np.savetxt(f, original_output.detach().numpy())
    #     # np.savetxt(f, adv_output.detach().cpu().numpy())
    #     # f.write(gt_label_log+'\n')
    #     f.write(ori_label_log+'\n')
    #     f.write(adv_label_log+'\n')
    #     f.write("================================\n")

    #     # _, axarr = plt.subplots(2, batch_size)
    #     for i in range(batch_size):
    #         # save images
    #         save_image(original_input[i].cpu(), "../original_img/"+str(namecounter)+".png")
    #         print(f"saving img NO. {namecounter}")
    #         save_image(adv_inputs[i].cpu(), "../adv_img/"+str(namecounter)+".png")
    #         # visualize images
    #         # axarr[0, i].imshow(original_input[i].cpu().permute(1, 2, 0))
    #         # axarr[1, i].imshow(adv_inputs[i].cpu().permute(1, 2, 0))
    #         namecounter += 1
    #     # plt.show()

    #     if num_examples >= 10*batch_size:
    #         break

    # print(f"Metrics: \n Attack Success Rate = {success / num_examples}")

    # f.close()
    # # ======================== #

    # for tensor case
    # ======================= #
    # img = Image.open("../test.jpeg")
    # inputs = transform(img)[None, ]
    # stackedinputs = torch.stack([inputs, inputs]).squeeze(dim=1)
    # out = model(stackedinputs)
    # clsidx = torch.argmax(out, dim=1)
    # # print(f"original prediction: {clsidx}")
    # atk = ShampooAttackViT_tensor(model, 224, 16, steps=20, perturb_bound=0.05, lr=10)
    # advinputs, original_input = atk(stackedinputs, clsidx)
    # # print(torch.eq(inputs, original_input))
    # out2 = model(advinputs)
    # clsidx2 = torch.argmax(out2, dim=1)
    # print(f"original prediction: {clsidx}, new prediction: {clsidx2}")
    # print(out)
    # print(out2)
    # ====================== #

    # for opt case
    # ======================= #
    # img = Image.open("../test.jpeg")
    # inputs = transform(img)[None, ]
    # stackedinputs = torch.stack([inputs, inputs, inputs, inputs, inputs, inputs, inputs, inputs]).squeeze(dim=1)
    # out = model(stackedinputs)
    # clsidx = torch.argmax(out, dim=1)
    # # print(f"original prediction: {clsidx}")
    # atk = ShampooAttackViT_opt(model, 224, 16, steps=20, perturb_bound=0.05)
    # atk_vanilla = ShampooAttackViT(model, 224, 16, steps=20, perturb_bound=0.05)
    # s_atk = time.time()
    # advinputs, original_input = atk(stackedinputs, clsidx)
    # e_atk = time.time()
    # advinputs_vanilla, _ = atk_vanilla(stackedinputs, clsidx)
    # print(f"vanilla attack for 4 img costs {time.time()-e_atk}")
    # print(f"attack for 4 img costs {e_atk-s_atk}")
    # # print(torch.eq(inputs, original_input))
    # out2 = model(advinputs)
    # clsidx2 = torch.argmax(out2, dim=1)
    # print(f"original prediction: {clsidx}, new prediction: {clsidx2}")
    # print(out)
    # print(out2)
    # # ====================== #

    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(inputs[0].permute(1, 2, 0))
    # axarr[1].imshow(advinputs[0].permute(1, 2, 0))
    # plt.show()

    # f.close()


if __name__ == "__main__":
    run()