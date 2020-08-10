import torch
import pdb
state = torch.load("./save_weights/layer4.0.conv2.weight_binary.pt",
            map_location="cuda")

value = state['layer4.0.conv2.weight']

tensor = value[0, ...]
pdb.set_trace()
torch.save(tensor, "test.pt")

