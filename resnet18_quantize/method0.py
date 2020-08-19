import argparse
import collections
import cProfile
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pdb
import random
import shutil
import time
import warnings
from datetime import datetime
from pprint import pprint
from pyinstrument import Profiler
import logging
from collector import collector_context
import distiller

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import SAsimulate2
import weight_mapping as wmp
from binary_converter import bit2float, float2bit, integer2bit
from models import *
from utils import progress_bar

msglogger = logging.getLogger()


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

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    distiller.quantization.add_post_train_quant_args(parser)
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

    # Initialize Tensorboard

    writer = SummaryWriter(
        "runs/{}-{}".format(now, "method0-8bitquantized (30 points) 1e-10 - 1e-01, 100")
    )
    model = ResNet18().to(device)
    state_dict = torch.load("./checkpoint/resnet.pt")["net"]
    model.load_state_dict(state_dict)

    # Create model
    if args.quantize_eval:
        quantizer = distiller.quantization.PostTrainLinearQuantizer.from_args(
            model, args
        )
        quantizer.prepare_model(torch.randn(100, 3, 32, 32))
    # Load model state dict
    quantized_state_dict = model.state_dict()
    simulate = SAsimulate(
        test_loader,
        model,
        quantized_state_dict,
        method="method0",
        mapped_gen=[],
        writer=writer,
    )
    # error_range = np.linspace(1e-10, 1., 1000)
    error_range = np.logspace(-10, -1, 100)
    simulate.run(error_range)

class SAsimulate:
    def __init__(self, test_loader, model, state_dict, method, mapped_gen, writer):
        self.test_loader = test_loader
        self.model = model
        self.state_dict = state_dict
        self.method = method
        self.mapped_gen = mapped_gen
        self.device = torch.device("cuda")
        self.writer = writer
        # model.load_state_dict(state_dict)
        self.criterion = nn.CrossEntropyLoss()
        # Calculate total_param
        self.total_param = 0
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            else:
                self.total_param += param.numel()
        print("Total param: {}".format(self.total_param))

    def run(self, error_range):
        count = 0
        avr_error = 0.0

        orig_model = self.model

        if self.method == "method0":
            for error_total in error_range:
                running_error = []
                running_deviation = []
                count += 1
                print("Error rate: ", error_total)
                for i in range(30):
                    pdb.set_trace()
                    model = method0(orig_model, self.total_param, error_total)
                    acc1 = test(model, self.test_loader, self.device, self.criterion)
                    running_error.append(100.0 - acc1)
                    orig_model.load_state_dict(self.state_dict)

                avr_error = sum(running_error) / len(running_error)
                print("Avarage classification Error: ", avr_error)
                self.writer.add_scalar("Average Error", avr_error, count)
                self.writer.close()


# Inject error without doing anything to the weight
def method0(model, total_param, error_total):
    device = torch.device("cuda")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            else:
                shape = param.data.shape
                error_layer = (param.numel() / total_param) * error_total
                print(error_layer)
                param_binary = integer2bit(param.data)
                mask, mask1 = SAsimulate2.create_mask(param_binary, error_layer)
                # mask, mask1 = mask.to(device), mask1.to(device)
                output = SAsimulate2.make_SA2(param.data.view(-1), mask, mask1)
                param.data = output.view(shape)
    return model

if __name__ == "__main__":
    main()
