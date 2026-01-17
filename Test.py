from transformers import pipeline
import torch

# Set device to MPS
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = pipeline("text-generation", model="microsoft/Phi-3.5-mini-instruct", device=device)
response = model("Xin chào ngày mới")
print(response)
