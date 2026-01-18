from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"


class ModelLoader:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name

    def get_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        )
        model.to(device)
        model.eval()

        return model

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)
