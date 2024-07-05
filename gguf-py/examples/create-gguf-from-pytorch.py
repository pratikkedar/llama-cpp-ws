#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

import numpy as np

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFWriter, GGMLQuantizationType

#import quants package
from gguf import quants

class PyTorchModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.state_dict = None
        self.tensor_dict = {}

    def load_model(self):
        """Load the PyTorch model state dictionary from a file."""
        self.state_dict = torch.load(self.model_path)

    def extract_tensors(self):
        """Extract tensors and their names from the state dictionary."""
        for name, tensor in self.state_dict.items():
            self.tensor_dict[name] = tensor

    def get_tensors(self):
        """Return the dictionary containing tensor names and tensor data."""
        return self.tensor_dict

    def print_tensors(self):
        """Print the tensor names and tensor data."""
        for name, tensor in self.tensor_dict.items():
            print(f"Tensor Name: {name}")
            print(tensor.shape)
            print(tensor.dtype)
            print(tensor)
            print()


# Example usage:
def writer_example(tensor_dict) -> None:
    # Example usage with a file
    gguf_writer = GGUFWriter("test-2-float-model.gguf", "llama")

    for name, tensor in tensor_dict.items():
	
		#default block based quantization 
		#quant_tensor = quants.quantize_q8_0(tensor.numpy())
        #gguf_writer.add_tensor(name, tensor.numpy())
		#gguf_writer.add_tensor(name, quantized_tensor, raw_dtype=GGMLQuantizationType.Q8_0)
		
		gguf_writer.add_tensor(name, tensor.numpy())
        
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()


if __name__ == "__main__":
    
    model_path = '/content/test_2_float_model.pt'  # Specify the path to your PyTorch model file
    loader = PyTorchModelLoader(model_path)
    loader.load_model()
    loader.extract_tensors()
    tensors = loader.get_tensors()

    writer_example(tensors)
