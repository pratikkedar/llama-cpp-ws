import torch
import torch.nn as nn
import torch.onnx

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.activation1 = nn.ReLU()
        self.matmul1 = nn.Linear(64, 64, bias=False) 
        self.activation2 = nn.ReLU()
        self.matmul2 = nn.Linear(64, 64, bias=False)  
        self.activation3 = nn.ReLU()

    def forward(self, x):
        x = self.activation1(x)
        x = self.matmul1(x)
        x = self.activation2(x)
        x = self.matmul2(x)
        x = self.activation3(x)
        return x

model = CustomModel()
input_tensor = torch.randn(1, 64) 
output = model(input_tensor)
print(output)

# Define the path where the model will be saved
model_save_path = "/content/test_2_float_model.pt"

# Save the model state dictionary to the specified path
torch.save(model.state_dict(), model_save_path)

# Print a message to indicate that the model has been saved
print("Model saved to:", model_save_path)

# Define the path where the ONNX model will be saved
onnx_model_save_path = "/content/test_2_float_model.onnx"

# Export the model to ONNX format
torch.onnx.export(model, input_tensor, onnx_model_save_path)

# Print a message to indicate that the model has been saved
print("Model exported to ONNX format:", onnx_model_save_path)


