import torch
import onnx
from polkash_translator_main import Seq2seq

# Load your trained model
model = MyModel()
model.load_state_dict(torch.load("model_seq2seq.pt"))
model.eval()

# Define a dummy input (adjust the shape based on your model's input)
dummy_input = torch.randn(1, 3, 224, 224)  # Example for image models

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "model_se2seq_polkash.onnx", opset_version=11)
