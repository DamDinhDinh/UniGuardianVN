from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"


class ModelLoader:
    def __init__(self, model_name=MODEL_NAME, device = DEFAULT_DEVICE):
        self.model_name = model_name
        self.device = device

    def get_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        )
        model.to(self.device)
        model.eval()

        return model

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)
