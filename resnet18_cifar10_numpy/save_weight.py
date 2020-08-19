import argparse
import collections
import json
import os
import pdb

import numpy as np
import torch
from tqdm import tqdm

import weight_mapping as wmp
from binary_converter import bit2float, float2bit

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="Map weights and save to file")
parser.add_argument(
    "-w", "--weights", default="./checkpoint/resnet.pt", help="path to weight dict"
)
args = parser.parse_args()

state_dict = torch.load(args.weights)["net"]
save_weights = collections.OrderedDict({})
save_binary = collections.OrderedDict({})

# Save mapped float weights

# for layer in tqdm(state_dict):
#     weights = state_dict[layer].view(-1)
#     map_cases = wmp.mapallweights2(weights)
#     save_weights.update({layer: map_cases})
# torch.save(save_weights, "saved_weights.pt")

# for i, layer in enumerate(state_dict):
#     weights = state_dict[layer].view(-1)
#     map_cases = wmp.mapallweights2(weights)
#     torch.save(map_cases,"layer" + str(i) + ".pt")
#     print("Layer", i, "is saved")

# ################# Save mapped binary weights:
for layer in state_dict:
    save_binary = collections.OrderedDict({})
    weights = state_dict[layer].view(-1)
    map_cases = wmp.mapallweights2(weights)
    if (weights.numel() >= 16):
        map_binary = float2bit(map_cases, num_e_bits=8, num_m_bits=23, bias=127.0)
        save_binary.update({layer: map_binary})
        torch.save(save_binary, "./save_weights/"+ str(layer) + "_binary.pt")
    else:
        map_binary = weights
        save_binary.update({layer: map_binary})
        torch.save(save_binary, "./save_weights/"+ str(layer) + "_binary.pt")
        print("Numbers of weights smaller than 16")

# def weight_gen(mapped_weights):
#     for layer in mapped_weights:
#         yield mapped_weights[layer]

# # Load mapped binary weights
# print("Loading binary weights ...")
# mapped_binary = torch.load("saved_binarymap.pt", map_location=device)
# print("Binary Weights loaded")

# weight_gen = weight_gen(mapped_binary)
# for key in weight_gen:
#     print(key.shape)
