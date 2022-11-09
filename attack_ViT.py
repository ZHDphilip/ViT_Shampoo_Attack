import time

import argparse

import ml_collections

from shampoo_attack_ViT import ShampooAttackViT
from shampoo_attack_ViT_optimized import ShampooAttackViT_opt
from shampoo_attack_ViT_Tensor import ShampooAttackViT_Tensor
from PGD_attack import PGD

from models.modeling import VisionTransformer
from models.Vit import vit_base_patch16_224_in21k, vit_base_patch16_224

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

def run(method, eps, max_img, iteration, lr, stats, model_type, scale, args=None):
    # for ImageNet pretrained Deit
    # model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()
    # model.eval()
    
    # for CIFAR10 Pretrained ViT
    configs = get_b16_config()
    model = VisionTransformer(configs, 224, zero_head=True, num_classes=10)
    model.load_state_dict(torch.load("./cifar10-100_500_checkpoint.pt"))
    model.to("cuda:0")
    model.eval()

    transform = transforms.Compose([
          transforms.Resize(args.resize),
          # transforms.CenterCrop(args.crop),
          transforms.ToTensor(),
          # imagenet
          # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          # cifar10
          # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
      ])

    if model_type == 'IMAGENET':
      model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()
      model.eval()
    elif model_type == 'Adv_CIFAR10':
      model = eval('vit_base_patch16_224')(
        pretrained=False, 
        # patch_size=args.patch, 
        img_size=args.crop, num_classes=10, args=args
      ).cuda()
      if args.patch == 4:
        checkpoint = torch.load("./checkpoint_20_p4_05trans")
      else:
        checkpoint = torch.load("./checkpoint_20_p16")
      model.load_state_dict(checkpoint['state_dict'])
      model.eval()

      cifar10_mean = (0.4914, 0.4822, 0.4465)
      cifar10_std = (0.2471, 0.2435, 0.2616)
      transform = transforms.Compose([
          transforms.Resize(args.resize),
          # transforms.CenterCrop(args.crop),
          transforms.ToTensor(),
          # imagenet
          # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          # cifar10
          transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
      ])

    batch_size = 1
    
    # load CIFAR10
    trainset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    print(os.listdir("../original_img"))
    print([name for name in os.listdir("../original_img") if "png" in name])
    namecounter = len([name for name in os.listdir("../original_img") if "png" in name])

    # initialize Shampoo Attacker
    # for e in eps:
    #   # for scale in lr_scale:
    #     s = e / 4
    print(f"experiment with epsilon = {eps}, step size = {lr}")
    if method == 0:
        atk = ShampooAttackViT_opt(model, args.crop, args.patch, steps=iteration, perturb_bound=eps, lr=lr, scale=scale)
    elif method == 1:
        atk = PGD(model, eps=eps, alpha=lr, steps=iteration, scale=scale)
    elif method == 2:
        atk = ShampooAttackViT_Tensor(model, args.crop, args.patch, steps=iteration, perturb_bound=eps, lr=lr)

    success = 0
    num_examples = 0
    no_need = 0

    mean = []
    std = []


    for data, label in tqdm(trainloader):
        data = data.to("cuda:0")
        print(torch.max(data), torch.min(data))
        ori_pred = torch.argmax(model(data),dim=1)
        if ori_pred[0].cpu() != label[0]:
          no_need += 1
          num_examples += 1
          continue
        advinputs = atk(data, label)
        # print(data)
        # print(advinputs)
        # exit()
        mean.append(torch.mean(advinputs-data).cpu())
        std.append(torch.std(advinputs-data).cpu())
        Maxdiff = torch.max(advinputs-data)
        Mindiff = torch.min(advinputs-data)
        stats = torch.unique((advinputs-data).flatten(), sorted=True, return_inverse=False, return_counts=True, dim=-1) 

        adv_output = model(advinputs)
        adv_labels = torch.argmax(adv_output, dim=1)

        num_examples += len(label)
        success += torch.sum(adv_labels.cuda()!=label.cuda()).item()
        print(f"Generated {num_examples} adversarial images, success in {success}, with {no_need} instances original pred wrong; diff mean {mean[-1]}, diff std {std[-1]}, max {Maxdiff}, min {Mindiff}")
        if num_examples % 100 == batch_size and stats:
          # print(f"stats: {stats[0]}")
          stat = dict(zip(stats[0].cpu(), stats[1].cpu()))
          print(f"stats: {sorted(stat.items(), key=lambda x:x[1] ,reverse=True)}, {torch.sum(stats[1])} items in total")
        
        for i in range(len(label)):
          if label[i] == adv_labels[i] and num_examples % 100 == 0:
            print(f"failed on an image of {classes[label[i]]}")
            save_image(data[i].cpu(), "../original_img/"+str(namecounter)+classes[label[i]]+".png")
            print(f"saving img NO. {namecounter}")
            save_image(advinputs[i].cpu(), "../adv_img/"+str(namecounter)+classes[adv_labels[i]]+".png")
            namecounter += 1
        
        if num_examples >= max_img:
          break
    print(f"Metrics: \n Attack Success Rate for epsilon {eps}, step size {lr} = {success / (num_examples-no_need)}, with {no_need} instances original pred wrong; diff mean: {np.mean(mean)}")


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
    parser.add_argument("--model", type=str, default='CIFAR10', help="which model to attack, [CIFAR10, IMAGENET, Adv_CIFAR10")
    parser.add_argument("--method", type=int, default=0, help="which attack algorithm to use, 0 shampoo matrix, 1 PGD, 2 shampoo tensor")
    parser.add_argument("--eps", type=float, default=2, help="test eps starts from arg/255")
    # parser.add_argument("--nstep_eps", type=int, default=3, \
    #   help="number of times to multiply eps by 2 for testing, eg eps_low=2/255 nstep=3 will test on eps=[2/255,4/255,6/255]")
    parser.add_argument("--lr", type=float, default=0.5, help="test learing starts from arg/255")
    parser.add_argument("--print_stats", action='store_true')
    parser.add_argument("--max_img", type=int, default=10000, help="number of images to attack for each set of parameter")
    parser.add_argument("--iter", type=int, default=10, help="number of iterations in PGD or Shampoo attack")
    parser.add_argument("--scale", type=int, default=1, help="scaling factor of image pixel compare to 0-1")

    # from robust train vit
    parser.add_argument('--run-dummy', action='store_true')
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--prefetch', type=int, default=1)#2)
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--tfds-dir', type=str, default='~/dataset/tar')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval-steps', type=int, default=1000)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-eval', default=512, type=int)
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--no-timm', action='store_true')
    parser.add_argument('--crop', type=int, default=32)
    parser.add_argument('--resize', type=int, default=32)
    parser.add_argument('--load', type=str)
    parser.add_argument('--data-loader', type=str, default='torch', choices=['torch'])
    parser.add_argument('--no-inception-crop', action='store_true')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--custom-vit', action='store_true')
    parser.add_argument('--base-lr', type=float, default=0.03)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--attack-iters', type=int, default=7, help='for pgd training')
    parser.add_argument('--patch', type=int, default=4)
    parser.add_argument('--load-state-dict-only', action='store_true')
    parser.add_argument('--patch-embed-scratch', action='store_true')
    parser.add_argument('--num-layers', type=int)

    parser.add_argument('--eval-restarts', type=int, default=1)
    parser.add_argument('--eval-iters', type=int, default=10)
    parser.add_argument('--downsample-factor', action='store_true')
    parser.add_argument('--eval-all', action='store_true')
    parser.add_argument('--eval-aa', action='store_true')
    parser.add_argument('--num-classes', type=int, default=10)

    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-decay-milestones', type=int, nargs='+', default=[15,18])
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-natural', default=5e-4, type=float)
    parser.add_argument('--weight-decay', default=2e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', '--dir', default='output_dir', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    # parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
    #     help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    # parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
    #     help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    # parser.add_argument('--master-weights', action='store_true',
    #     help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('--pretrain-pos-only', action='store_true')

    args = parser.parse_args()
    eps = args.eps / 255
    lr = args.lr / 255
    run(args.method, eps, args.max_img, args.iter, lr, args.print_stats, args.model, int(args.scale), args)