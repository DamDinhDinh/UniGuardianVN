from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEFAULT_DEVICE = None
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

class ModelLoader:
    def __init__(self, model_name=MODEL_NAME, device = DEFAULT_DEVICE):
        self.model_name = model_name
        self.device = device

    def get_model(self):
        device_map="auto" if self.device == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else None,
            device_map=device_map,
        )
        if device_map is None:
            model.to(self.device)
        model.eval()

        return model

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)
