import argparse
import collections
import cProfile
import os
import pdb
import random
import shutil
import sys
import time
import warnings
from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import SAsimulate3
import weight_mapping as wmp
from binary_converter import bit2float, float2bit
from models import *
from pyinstrument import Profiler
from pytorch_memlab import MemReporter, profile_every
from utils import progress_bar

# Specify gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Tensorboard run id
now = datetime.now().date()
ran = random.randint(1, 231242)

def test(net, testloader, device, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
    # Save best accuracy.
    acc = 100.0 * correct / total
    return acc

def main():

    print("Loading mapped float weights:")
    mapped_float = load_mapped_weights()
    # mapped_binary = load_mapped_binary()
    print("Weights loaded")

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    args = parser.parse_args()

    print("==> Preparing data..")

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    device = torch.device("cuda")

    writer = SummaryWriter("runs/{}-{}".format(now, "cifar10-method1 (100 points) 1e-10 --> 1e-05 (10) -- code fix"))

    # Create model
    model = ResNet18().to(device)
    # Load model state dict
    state_dict = torch.load("./checkpoint/resnet.pt")["net"]
    simulate = SAsimulate(
        test_loader,
        model,
        state_dict,
        method="method1",
        mapped_float=mapped_float,
        writer=writer,
    )
    error_range = np.linspace(1e-10, 1e-05, 10)
    simulate.run(error_range)

    # count = 0
    # avr_error = 0.
    # for error_total in error_range:
    #     running_error = []
    #     count += 1
    #     print("Error rate: ", error_total)
    #     for i in range(10):
    #         model = method1(model, mapped_gen, error_total)
    #         acc1 = test(print_freq=10, model=model, device=device, test_loader=test_loader)

    #         running_error.append(100. - acc1.item())
    #     avr_error = sum(running_error)/len(running_error)
    #     print("Avarage classification Error: ", avr_error)
    #     writer.add_scalar("Average Error", avr_error, count)
    #     writer.close()

    # for error_total in np.linspace(1e-10, 1e-05, 100):
    #     print("Error rate: ", error_total)
    #     weight_generator = weight_gen(mapped_gen)
    #     model2 = method1(model.to(device), weight_generator, error_total=error_total)
    #     acc1 = test(print_freq=10, model=model2, device=device, test_loader=test_loader)
    #     print(acc1)


class SAsimulate:
    def __init__(self, test_loader, model, state_dict, method, mapped_float, writer):
        self.test_loader = test_loader
        self.model = model
        self.state_dict = state_dict
        self.method = method
        self.mapped_float = mapped_float
        self.device = torch.device("cuda")
        self.writer = writer
        model.load_state_dict(state_dict)
        model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def run(self, error_range):
        count = 0
        avr_error = 0.0
        des = np.arange(7.071e-05, 0.0001, 9.999e-07)
        print_freq = 10
        orig_model = self.model

        if self.method == "method0":
            for error_total in error_range:
                running_error = []
                count += 1
                print("Error rate: ", error_total)
                for i in range(50):
                    model = method0(self.model, error_total)
                    acc1 = test(print_freq, model, self.device, self.test_loader)
                    running_error.append(100.0 - acc1.item())
                avr_error = sum(running_error) / len(running_error)
                print("Avarage classification Error: ", avr_error)
                self.writer.add_scalar("Average Error", avr_error, count)
                self.writer.close()

        if self.method == "method1":
            for error_total in error_range:
                running_error = []
                count += 1
                print("Error rate: ", error_total)
                for i in range(100):
                    model = method1(orig_model, self.mapped_float, error_total)
                    acc1 = test(model, self.test_loader, self.device, self.criterion)
                    running_error.append(100.0 - acc1)
                    orig_model.load_state_dict(self.state_dict)

                avr_error = sum(running_error) / len(running_error)
                print("Avarage classification Error: ", avr_error)
                self.writer.add_scalar("Average Error", avr_error, count)
                self.writer.close()

# Inject error without doing anything to the weight
def method0(model, error_total):
    total_param = 0
    with torch.no_grad():
        for param in model.parameters():
            total_param += param.numel()
        for param in model.parameters():
            shape = param.data.shape
            error_layer = (param.numel() / total_param) * error_total
            param_binary = float2bit(
                param.data, num_e_bits=8, num_m_bits=23, bias=127.0
            )
            mask, mask1 = SAsimulate2.create_mask(param_binary, error_layer)
            mask, mask1 = mask.to("cuda"), mask1.to("cuda")
            output = SAsimulate2.make_SA2(param.data.view(-1), mask, mask1)
            param.data = output.view(shape)
    return model


# XOR address mapping weight to reduce stuck at fault bits
def method1(model, mapped_float, error_total):
    total_param = 1173962
    with torch.no_grad():
        for name, param in model.named_parameters() :
            # TODO: Skip running mean and var and num_batches_tracked layer
            error_layer = (param.numel() / total_param) * error_total
            # print("Loading: " + str(name))
            # if name == "layer4.0.conv1.weight":
            mapped_binary_dict = torch.load(
                "./save_weights/"+ str(name) + "_binary.pt",
                map_location=torch.device("cuda"),
            )
            mapped_binary_val = mapped_binary_dict[name]
            output = weight_map2(
                param.data, mapped_float[name], mapped_binary_val, error_layer
            )
            param.data = output
    return model


## Create mask --> Find minimum mapping --> Inject error --> Remap
## Input: weights in 1 layer, error rate of layer
## Output: new weight with less error

# @profile_every(1)

def weight_map2(weights, mapped_float, mapped_binary, error_rate):
    shape = weights.shape
    weights_flat = weights.view(-1)
    device = torch.device("cuda")
    if weights_flat.numel() > 16:
        weight_binary = mapped_binary
    else:
        return weights
    # Creating masks for all weights in one layer
    # conv_binary = float2bit(weights, num_e_bits=8, num_m_bits=23, bias=127.)
    mask0_binary, mask1_binary = SAsimulate3.create_mask(shape, error_rate=error_rate)
    # mask0_binary, mask1_binary = mask0_binary.to(torch.device("cuda")), mask1_binary.to(torch.device("cuda"))
    # mask0_binary, mask1_binary = mask0_binary.repeat(16, 1).view(16, int(weights_flat.shape[0]/16), 16, 32),\
    #         mask1_binary.repeat(16, 1).view(16, int(weights_flat.shape[0]/16), 16, 32)
    # mask0_binary, mask1_binary = mask0_binary.transpose_(1, 0), mask1_binary.transpose_(1, 0)
    # reporter = MemReporter()
    # reporter.report()
    mask0_binary, mask1_binary = (
        mask0_binary.view(int(mask0_binary.numel() / 32 / 16), 16, 32),
        mask1_binary.view(int(mask1_binary.numel() / 32 / 16), 16, 32),
    )
    new_weight_binary = torch.empty([*mapped_binary.shape], device = device)
    for i in range(16):
        new_weight_binary[:, :, i, :] = SAsimulate3.make_SA(
            mapped_binary[:, :, i, :], mask0_binary, mask1_binary
        )
    del mask0_binary
    del mask1_binary
    del mapped_binary
    torch.cuda.empty_cache()

    # new_weight = bit2float(
    #     new_weight_binary, num_e_bits=8, num_m_bits=23, bias=127.0
    # )
    new_weight = new_map_gen(new_weight_binary)

    # mapped_binary = mapped_binary.to(device)
    binary_index = 0
    weight_index = 0

    index = torch.arange(16).to("cuda")
    index_map = wmp.mapallweights(index)
    for weight_cases, new_weight_16 in zip(mapped_float[:, ...], new_weight):
        if weight_cases.numel() < 16:
            break
        else:
            origin_weight = weights_flat[weight_index : weight_index + 16]
            # dev = torch.empty(16, 16)
            dev = abs(weight_cases - new_weight_16)
            # for i in range(16):
            #     # weight remap
            #     weight_remap = torch.index_select(
            #         new_weight_16[i, ...], 0, index_map[0]["w" + str(i)]
            #     )
                # dev[i, ...] = abs(origin_weight - weight_remap)
            dev_sum = torch.sum(dev, dim=0)
            min_dev, best_map = torch.min(dev_sum, dim=0)
            _, indicies = torch.sort(index_map[0]["w" + str(best_map.item())])
            weight_remap2 = torch.index_select(
                new_weight_16[best_map.item(), ...],
                0,
                indicies,
            )
            weights_flat[weight_index : weight_index + 16] = weight_remap2
            binary_index = binary_index + 512
            weight_index = weight_index + 16
    new_weights = weights_flat.view(shape)
    del new_weight
    del new_weight_binary
    del weights
    torch.cuda.empty_cache()
    return new_weights

def new_map_gen(binary_map):
    for i in range(binary_map.shape[0]):
        yield bit2float(binary_map[i, ...], 8, 23, 127.0)

######## Load mapped weights from file and return a generator of weights for each layer


def load_mapped_weights(path="saved_weights.pt", device=torch.device("cuda")):
    print("Loading float weights ...")
    mapped_weights = torch.load(path, map_location=device)
    return mapped_weights


def load_mapped_binary(path="saved_binarymap.pt", device=torch.device("cuda")):
    print("Loading binary weights ...")
    mapped_weights = torch.load(path, map_location=device)
    return mapped_weights


def weight_gen(mapped_weights):
    for layer in mapped_weights.items():
        yield layer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# import cProfile
# device = torch.device("cuda")
# state_dict = torch.load("mnist_cnn.pt")
# weights = state_dict["fc1.weight"]
# mapped_float = load_mapped_weights()
# mapped_binary = load_mapped_binary()
# weight_float = mapped_float["fc1.weight"]
# weight_binary = mapped_binary["fc1.weight"]
# weight_map2(weights, weight_float, weight_binary, 0.001)

# cProfile.run("weight_map2(weights, weight_float, weight_binary, 0.01)")

if __name__ == "__main__":
    main()
