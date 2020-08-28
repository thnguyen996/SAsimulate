import torch
from quantizer import *
import cupy as cp
import pdb
import argparse


parser = argparse.ArgumentParser(description='Quantized pretrained model and save to file')
parser.add_argument('--state_dict', metavar='DIR', default="path_to_state_dict",
                    help='path to state_dict')
parser.add_argument('--num_bits', metavar='DIR', default=8,
                    help='path to state_dict')

def main():
    global args
    args = parser.parse_args()
    state_dict = torch.load(args.state_dict)
    qparams ={}
    qparams["num_bits"] = args.num_bits

    print("###################### Quantizing weight ###################")
    for name, param in state_dict.items():
        if "weight" in name:
            scale, zero_point, quantized_weight = dorefa_quantize_param(param, args.num_bits, False)
            qparams[name + ".scale"] = scale
            qparams[name + ".zero_point"] = zero_point
            param.copy_(quantized_weight)

    # Save quantized parameters
    torch.save(qparams, "./qparams.pt")

    # Convert to cupy and save
    for name, param in state_dict.items():
        if "weight" in name:
            param_cp = torch_to_cp(param)
            cp.save("./save_weights/" + name + '.npy', param_cp)



def torch_to_cp(tensor):
    tensor_np = tensor.cpu().numpy()
    return cp.asarray(tensor_np)

if __name__ == '__main__':
    main()

