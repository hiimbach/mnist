import os 
import sys
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch 
from model.model import ConvNet

def trace(weight_path, save_path="weight/scripted.pt"):
    # Load model
    net = ConvNet()
    net.load_state_dict(torch.load(weight_path))
    
    # Trace
    rand_inp = torch.rand(1,1,28,28)
    module = torch.jit.trace(net.forward, rand_inp)
    module.save(save_path)
    
    
if __name__ == "__main__":
    trace("weight/weight.pt")