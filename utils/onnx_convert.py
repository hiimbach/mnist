import os 
import sys
import onnx
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch 
from model.model import ConvNet

def onnx_convert(weight_path, save_path="weight/mnist.onnx"):
    # Load model
    net = ConvNet()
    net.load_state_dict(torch.load(weight_path))
    
    # Set to inference model and convert to onnx
    net.eval()
    
    # Input to the model
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28, requires_grad=True)
    torch_out = net(x)

    # Export the model
    torch.onnx.export(net,                         # model being run
                        x,                         # model input (or a tuple for multiple inputs)
                        save_path,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=10,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})
        
        
    # Check onnx model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"ONNX model is saved in {save_path}")
    
    
if __name__ == "__main__":
    onnx_convert("weight/weight.pt")