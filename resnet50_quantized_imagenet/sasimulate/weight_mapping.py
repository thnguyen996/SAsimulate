import numpy as np
from pprint import pprint
import torch
import pdb
import collections
def switch1(w):
    weight = w.clone()
    temp = weight.clone()
    weight[0::2] = temp[1::2]
    weight[1::2] = temp[0::2]
    return weight

def switch2(w):
    weight = w.clone()
    weight = weight.view(int(w.numel()/2), 2)
    temp = weight.clone()
    weight[0::2, :] = temp[1::2, :]
    weight[1::2, :] = temp[0::2, :]
    return weight.view(-1)

def switch4(w):
    weight = w.clone()
    weight = weight.view(int(w.numel()/4), 4)
    temp = weight.clone()
    weight[0::2, :] = temp[1::2, :]
    weight[1::2, :] = temp[0::2, :]
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

def map_cases(weight_tensor):
    w = weight_tensor.view(-1)
    weight_cases = collections.OrderedDict({})
    weight_cases.update({"w0" : w})
    weight_cases.update({"w1" : switch1(w)})
    weight_cases.update({"w2" : switch2(w)})
    weight_cases.update({"w3" : switch1(weight_cases["w2"])})
    weight_cases.update({"w4" : switch4(w)})
    weight_cases.update({"w5" : switch1(weight_cases["w4"])})
    weight_cases.update({"w6" : switch2(weight_cases["w4"])})
    weight_cases.update({"w7" : switch1(weight_cases["w6"])})
    weight_cases.update({"w8" : torch.roll(w, 8)})
    weight_cases.update({"w9" : torch.roll(weight_cases["w1"], 8)})
    weight_cases.update({"w10" : torch.roll(weight_cases["w2"], 8)})
    weight_cases.update({"w11" : torch.roll(weight_cases["w3"], 8)})
    weight_cases.update({"w12" : torch.roll(weight_cases["w4"], 8)})
    weight_cases.update({"w13" : torch.roll(weight_cases["w5"], 8)})
    weight_cases.update({"w14" : torch.roll(weight_cases["w6"], 8)})
    weight_cases.update({"w15" : torch.roll(weight_cases["w7"], 8)})
    return weight_cases

def map_cases2(weight_tensor):
    w = weight_tensor.view(-1)
    w0 = w.clone()
    w1 = switch1(w)
    w2 = switch2(w)
    w3 = switch1(w2)
    w4 = switch4(w)
    w5 = switch1(w4)
    w6 = switch2(w4)
    w7 = switch1(w6)
    w8 = torch.roll(w, 8)
    w9 = torch.roll(w1, 8)
    w10 = torch.roll(w2, 8)
    w11 = torch.roll(w3, 8)
    w12 = torch.roll(w4, 8)
    w13 = torch.roll(w5, 8)
    w14 = torch.roll(w6, 8)
    w15 = torch.roll(w7, 8)
    weight_cases = torch.stack((w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15))
    return weight_cases

# Return list of mapped weights 
def mapallweights(weights):
    assert weights.shape == weights.view(-1).shape
    map_save = []
    if (weights.numel() % 16) == 0 and weights.numel() >= 16:
        for i in range(0, weights.numel(), 16): 
            map_dict = collections.OrderedDict({})
            weight16 = weights[i:i+16]
            map_dict = map_cases(weight16)
            map_save.append(map_dict)
        return map_save
    else:
        remainder = weights.numel() % 16
        print("Weights are not divisible by 16, skipping: ", remainder, "weights")
        if weights.numel() > 16:
            weights_map_length = weights.numel() - remainder
            for i in range(0, weights_map_length, 16): 
                map_dict = {}
                weight16 = weights[i:i+16]
                map_dict = map_cases(weight16)
                map_save.append(map_dict)
            map_save.append(weights[weights_map_length:weights.numel()])
            return map_save
        else:
            map_save.append(weights)
            return map_save

def mapallweights2(weights):
    assert weights.shape == weights.view(-1).shape
    num_groups = int(weights.numel()/16)
    map_tensor = torch.empty(num_groups, 16, 16)
    if (weights.numel() % 16) == 0 and weights.numel() >= 16:
        for i, j in zip(range(0, weights.numel(), 16), range(map_tensor.shape[0])): 
            weight16 = weights[i:i+16]
            map_dict = map_cases2(weight16)
            map_tensor[j, ...] = map_dict
        return map_tensor
    else:
        remainder = weights.numel() % 16
        print("Weights are not divisible by 16, skipping: ", remainder, "weights")
        if weights.numel() > 16:
            weights_map_length = weights.numel() - remainder
            for i, j in zip(range(0, weights_map_length, 16), range(map_tensor.shape[0])): 
                weight16 = weights[i:i+16]
                map_dict = map_cases2(weight16)
                map_tensor[j, ...] = map_dict
            return map_tensor, weights[weights_map_length : weights.numel()]
        else:
            return weights
## For debugging

def main():

    x = torch.arange(17)
    x = x.view(-1)
    print(len(mapallweights2(x)))
    
if __name__ == "__main__":
    main()
