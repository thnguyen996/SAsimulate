import torch
import numpy as np
import pdb
from torch.utils.tensorboard import SummaryWriter
import math

np.set_printoptions(threshold=np.inf)

def create_mask(weight_max, weight_min, total_weight, bit_error_rate):
    weight_max, weight_min = (
        weight_max.cpu().numpy(),
        weight_min.cpu().numpy(),
    )
    num_error_weights = int(bit_error_rate * total_weight)
    mask = np.random.uniform(
        low=weight_min, high=weight_max, size=num_error_weights 
    ).astype(np.float32)
    mask = mask.view(np.uint32)

    zeros_shape = total_weight - num_error_weights
    zeros = np.zeros(zeros_shape, dtype=np.uint32)
    mask = np.concatenate((mask, zeros))
    np.random.shuffle(mask)
    return mask0, mask1

def make_SA(weights, mask0, mask1):
    weights = weights.numpy().view(np.uint32)
    stuck_at_1 = np.bitwise_or(weights, mask1)
    pdb.set_trace()
    return weights

def calculate_weight_range(state_dict):
    weights = state_dict["conv1.weight"].view(-1)
    total = weights.numel()
    for layer, value in state_dict.items():
        flatten = value.view(-1)
        if layer == "conv1.weight":
            continue
        elif "weight" in layer:
            weights = torch.cat((weights, flatten), dim=0)
            total += flatten.numel()
        # weights = torch.cat((weights, flatten))
    torch.save(
        weights.cpu(), "whole_weights_cpu.pt",
    )
    return torch.max(weights), torch.min(weights), total

if __name__ == "__main__":
    # Comment out these 2 line to calculate weight_max, weight_min and totol number of weight
    # state_dict = torch.load("./checkpoint/resnet.pt")['net']
    # weight_max, weight_min, total = calculate_weight_range(state_dict)
    weights = torch.load("./whole_weights_cpu.pt")
    print(weights.device)
    weight_max = torch.tensor([0.7436], device="cuda:0")
    weight_min = torch.tensor([-0.6743], device="cuda:0")
    total_weight = 11169152 

    mask0, mask1 = create_mask(weight_max, weight_min, total_weight, bit_error_rate=0.001)
    make_SA(weights, mask0, mask1)

    # print(weight_max, weight_min, total)
    print(mask0.shape, mask1.shape)
