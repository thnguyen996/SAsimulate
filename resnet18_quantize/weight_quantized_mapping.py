import numpy as np
from pprint import pprint
import torch
import pdb
import collections


def switch1(w, num_weights):
    weight = w.view(int(w.numel()/num_weights), num_weights).clone()
    temp = weight.clone()
    # pdb.set_trace()
    weight[0::2, :] = temp[1::2, :]
    weight[1::2, :] = temp[0::2, :]
    return weight.view(-1)


def switch2(w, num_weights):
    weight = w.view(int(w.numel()/num_weights), num_weights).clone()
    weight = weight.view(int(weight.shape[0] / 2), 2, num_weights)
    temp = weight.clone()
    weight[0::2, :, :] = temp[1::2, :, :]
    weight[1::2, :, :] = temp[0::2, :, :]
    return weight.view(-1)


def switch4(w, num_weights):
    weight = w.view(int(w.numel()/num_weights), num_weights).clone()
    weight = weight.view(int(weight.shape[0] / 4), 4, num_weights)
    temp = weight.clone()
    weight[0::2, :, :] = temp[1::2, :, :]
    weight[1::2, :, :] = temp[0::2, :, :]
    return weight.view(-1)


# def remap(weight_tensor, best_key):
#   new_weight = torch.randn(weight_tensor.shape)
#   index = torch.tensor(range(16))
#   mapped_index = map_cases(index)[best_key]
#   for i in range(16):
#       new_weight[i] = weight_tensor[torch.where(mapped_index == i)]
#   return new_weight


def remap(weight_tensor, index):
    stack_index = torch.stack((weight_tensor, index))
    new_weights = stack_index.sort(1).values[0]
    return new_weights


# Given a weight tensor, calculate all the possible mapping case

def map_cases(weight_tensor, num_bits):
    num_weights = int(512./num_bits/16.)
    w = weight_tensor.view(-1)
    w0 = w.clone()
    w1 = switch1(w, num_weights)
    w2 = switch2(w, num_weights)
    w3 = switch1(w2, num_weights)
    w4 = switch4(w, num_weights)
    w5 = switch1(w4, num_weights)
    w6 = switch2(w4, num_weights)
    w7 = switch1(w6, num_weights)
    w8 = torch.roll(w, 8)
    w9 = torch.roll(w1, 8)
    w10 = torch.roll(w2, 8)
    w11 = torch.roll(w3, 8)
    w12 = torch.roll(w4, 8)
    w13 = torch.roll(w5, 8)
    w14 = torch.roll(w6, 8)
    w15 = torch.roll(w7, 8)
    weight_cases = torch.stack(
        (w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15)
    )
    return torch.transpose(weight_cases, 0, 1)


# Return list of mapped weights
def mapallweights(weights, num_bits):
    assert weights.shape == weights.view(-1).shape
    num_weights = int(512/num_bits)
    num_groups = int(weights.numel() / num_weights)
    map_tensor = torch.empty(num_groups, num_weights, 16)
    if (weights.numel() % num_weights) == 0 and weights.numel() >= num_weights:
        for i, j in zip(range(0, weights.numel(), num_weights), range(map_tensor.shape[0])):
            weight16 = weights[i : i + num_weights]
            map_dict = map_cases(weight16, num_bits)
            map_tensor[j, ...] = map_dict
        return torch.transpose(map_tensor, 1, 2)
    else:
        remainder = weights.numel() % num_weights
        print("Weights are not divisible by ", num_weights, ", skipping: ", remainder, "weights")
        if weights.numel() > num_weights:
            weights_map_length = weights.numel() - remainder
            for i, j in zip(
                range(0, weights_map_length, num_weights), range(map_tensor.shape[0])
            ):
                weight16 = weights[i : i + num_weights]
                map_dict = map_cases(weight16, num_bits)
                map_tensor[j, ...] = map_dict
            return torhc.transpose(map_tensor, 1, 2), weights[weights_map_length : weights.numel()]
        else:
            return weights


## For debugging


def main():

    x = torch.arange(64)
    num_bits = 8
    torch.set_printoptions(profile="full")
    pprint(mapallweights(x, num_bits).shape)
    torch.set_printoptions(profile="default")


if __name__ == "__main__":
    main()
