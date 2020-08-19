import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import argparse
import os
import random
import shutil
import time
import warnings
import SAsimulate2
import weight_mapping as wmp
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from binary_converter import float2bit, bit2float
# import pandas as pd
from datetime import datetime
import pdb
import os
import collections
from pprint import pprint
import cProfile
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

now = datetime.now().time()
ran = random.randint(1, 231242)

def test(print_freq, model, device, test_loader):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.to(device)
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % print_freq == 0:
                progress.display(batch_idx)
            # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg

def main():
   
    print('Loading mapped weights:')
    mapped_gen = load_mapped_weights()
    print("Weights loaded")

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    args = parser.parse_args()
    kwargs = {'num_workers': 1, 'pin_memory': True}

    # Load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    device = torch.device("cuda")
   
    writer = SummaryWriter('runs/{}-{}'.format(now, "method1"))
    # Create model
    model = SAsimulate2.Net().to(device)
    #Load model state dict
    state_dict = "mnist_cnn.pt"
    model.load_state_dict(torch.load(state_dict))

    simulate = SAsimulate(test_loader, model, state_dict, method="method1", mapped_gen = mapped_gen, writer = writer)  
    error_range = np.linspace(1e-10, 1e-05, 100)
    simulate.run(error_range)
    writer.add_scalar("Clasification error", avr_error, count)
    writer.close()

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
    def __init__(self, test_loader, model, state_dict, method, mapped_gen, writer):
        self.test_loader = test_loader
        self.model = model
        self.state_dict = state_dict
        self.method = method
        self.mapped_gen = mapped_gen
        self.device = torch.device("cuda")
        self.writer = writer
        model.load_state_dict(torch.load(state_dict))
        model.to(self.device)

    def run(self, error_range):
      count = 0
      avr_error = 0.
      des = np.arange(7.071e-05, 0.0001, 9.999e-07)
      print_freq = 10

      if self.method == "method0":
          for error_total in error_range:
              running_error = []
              count += 1
              print("Error rate: ", error_total)
              for i in range(50):
                  model = method0(self.model, error_total)
                  acc1 = test(print_freq, model, self.device, self.test_loader)
                  running_error.append(100. - acc1.item())
              avr_error = sum(running_error)/len(running_error)
              print("Avarage classification Error: ", avr_error)
              self.writer.add_scalar("Average Error", avr_error, count)
              self.writer.close()
  
      if self.method == "method1":
          for error_total in error_range:
              running_error = []
              count += 1
              print("Error rate: ", error_total)
              for i in range(10):
                  weight_generator = weight_gen(self.mapped_gen)
                  # pdb.set_trace()
                  model = method1(self.model, weight_generator, error_total)
                  acc1 = test(print_freq, model, self.device, self.test_loader)
                  running_error.append(100. - acc1.item())
              avr_error = sum(running_error)/len(running_error)
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
        error_layer = (param.numel()/total_param)*error_total
        param_binary = float2bit(param.data, num_e_bits=8, num_m_bits=23, bias=127.)
        mask, mask1 = SAsimulate2.create_mask(param_binary, error_layer)
        mask, mask1 = mask.to("cuda"), mask1.to("cuda")
        # pdb.set_trace()
        output = SAsimulate2.make_SA2(param.data.view(-1), mask, mask1)
        param.data = output.view(shape)
    return model

# XOR address mapping weight to reduce stuck at fault bits
def method1(model, mapped_gen, error_total):
    total_param = 0
    with torch.no_grad():
      for param in model.parameters():
          total_param += param.numel()
      for param, mapped_weight in zip(model.parameters(), mapped_gen):
        error_layer = (param.numel()/total_param)*error_total
        output = weight_map2(param.data, mapped_weight[1], error_layer)
        param.data = output
    return model

## Create mask --> Find minimum mapping --> Inject error --> Remap
## Input: weights in 1 layer, error rate of layer
## Output: new weight with less error

def weight_map(weights, error_rate):
    shape = weights.shape
    weights_flat = weights.view(-1)
    # Creating masks for all weights in one layer
    conv_binary = float2bit(weights, num_e_bits=8, num_m_bits=23, bias=127.)
    mask0_binary, mask1_binary = SAsimulate2.create_mask(conv_binary, error_rate=error_rate)
    binary_index = 0
    for i in range(0, weights_flat.numel(), 16):
        weights16 = weights_flat[i:i+16].clone()
        weight_cases = wmp.map_cases(weights16)       # return list of 16 mapped cases
        least_stuck = 0
        best_key = 0
  
        weight_cases_values = torch.cat([value for key, value in weight_cases.items()])   #concat all cases into a list
  
        weight_16_b = float2bit(weight_cases_values, num_e_bits=8, num_m_bits=23, bias=127.).view(-1)
        weight_16_binary =  weight_16_b.split(512)      # Split binary weight into 16 cases
        # Take portion of mask (16 weights)
        mask0_16_binary = mask0_binary[binary_index:binary_index+512]
        mask1_16_binary = mask1_binary[binary_index:binary_index+512]
        for key, value in enumerate(weight_16_binary):
          # Calculate total number of stuck bits
          stuck_total = SAsimulate2.calculate_stuck(value, mask0_16_binary, mask1_16_binary)
          if least_stuck == 0:
              least_stuck = stuck_total.item()
          elif stuck_total.item() <= least_stuck:
              least_stuck = stuck_total.item()
              best_key = key
  
        min_mapping = weight_cases["w"+str(best_key)]
        print(best_key)
        new_weight_16 = SAsimulate2.make_SA(min_mapping, mask0_16_binary, mask1_16_binary)  # inject error
        weight_remap = wmp.remap(new_weight_16, "w"+str(best_key))        #remap weights
        weights_flat[i:i+16] = weight_remap
        binary_index = binary_index + 512
    new_weights = weights_flat.view(shape)
    return new_weights

def weight_map2(weights, mapped_weights, error_rate):
    shape = weights.shape
    weights_flat = weights.view(-1)
     # Creating masks for all weights in one layer
    conv_binary = float2bit(weights, num_e_bits=8, num_m_bits=23, bias=127.)
    mask0_binary, mask1_binary = SAsimulate2.create_mask(conv_binary, error_rate=error_rate)
    mask0_binary, mask1_binary = mask0_binary.to(torch.device("cuda")), mask1_binary.to(torch.device("cuda"))
    binary_index = 0
    weight_index = 0 

    # return the list of mapped cases in 1 layers
    mapped_weights_list = mapped_weights
    if weights_flat.numel() > 16 :
        weight_binary_iters = binary_weights_mapped(weights_flat, mapped_weights = mapped_weights_list) 
    else:
        return weights
    index = torch.arange(16).to("cuda")
    index_map = wmp.mapallweights(index)

    for weight_cases, weight_16_binary in zip(mapped_weights_list, weight_binary_iters):
        if torch.is_tensor(weight_cases):
            break
        else:
            least_stuck = 0
            best_key = 0

            # Take portion of mask (16 weights)
            mask0_16_binary = mask0_binary[binary_index:binary_index+512].repeat(16, 1)
            mask1_16_binary = mask1_binary[binary_index:binary_index+512].repeat(16, 1)

            # Split binary weight into 16 cases
            weight_16_binary=  weight_16_binary.view(int(weight_16_binary.numel()/512), 512)     
            stuck_total = SAsimulate2.calculate_stuck(weight_16_binary, mask0_16_binary, mask1_16_binary)
            best_key = torch.where(stuck_total == min(stuck_total))[0][0].item()
            min_mapping = weight_cases["w"+str(best_key)]

            # Inject error 
            new_weight_16 = SAsimulate2.make_SA(weight_16_binary[best_key, :], mask0_16_binary[0, :], mask1_16_binary[0, :]) 
            #weight remap
            weight_remap  = torch.index_select(new_weight_16, 0, index_map[0]["w"+str(best_key)])

            weights_flat[weight_index:weight_index+16] = weight_remap
            binary_index = binary_index + 512
            weight_index = weight_index + 16
    new_weights = weights_flat.view(shape)
    return new_weights

# Generate mapped weights into binary form
# Input: Weights in 1 layer (Flatten)
# Output: Generator of tensor contain binary weights groups by 16 weights 

def binary_weights_mapped(weights, mapped_weights=None):
    assert weights.shape == weights.view(-1).shape
    merged = list(mapped_weights[0].values()) 
    for i in range(1, len(mapped_weights)):
        merged += list(mapped_weights[i].values()) 
    weights_tensor = torch.cat(merged)

    # Split weights in chunks
    if weights.numel() > 1000000:
        weights_split = weights_tensor.view(8, int(weights_tensor.numel()/8)) 
    else:
        weights_split = weights_tensor.view(2, int(weights_tensor.numel()/2)) 
    for i in range(weights_split.shape[0]):
        binary = float2bit(weights_split[i, :], num_e_bits=8, num_m_bits=23, bias=127.)
        weights_binary_16 = binary.view(int(binary.shape[0]/256), 256,  32)
        # pdb.set_trace()
        for y in range(weights_binary_16.shape[0]):
            yield torch.flatten(weights_binary_16[y, :, :])

##############################################################################

def binary_iter(binary_maps):
    for i in range(0, len(binary_maps), 16):
        yield binary_maps[i:i+16]

######## Load mapped weights from file and return a generator of weights for each layer

def load_mapped_weights(path="saved_weights.pt", device = torch.device('cuda')):
    print("Loading weights ...")
    mapped_weights = torch.load(path, map_location=device)
    return mapped_weights

def weight_gen(mapped_weights):
    for layer in mapped_weights.items():
        yield layer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
 
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


# torch.manual_seed(2410)
# device = torch.device("cuda")
# weights = torch.randn(10).to(device)
# mapped_weights = wmp.mapallweights(weights.view(-1))
# weight_map2(weights, mapped_weights, 0.01)
# # cProfile.run("weight_map2(weights, mapped_weights, 0.01)")

if __name__ == '__main__':
    main()

