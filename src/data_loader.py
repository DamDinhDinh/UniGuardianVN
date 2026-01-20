from datasets import load_dataset
from src.prompt_data import PromptData

DEFAULT_DATA_SET = "deepset/prompt-injections"


class DataLoader:
    def __init__(self, data_set_name=DEFAULT_DATA_SET):
        self.dataset_name = data_set_name
        self.dataset = load_dataset(data_set_name)
        self.train_prompt = self.get_train_prompt()
        self.test_prompt = self.get_test_prompt()

    def get_train_prompt(self):
        train_set = self.dataset["train"]
        print(f"get_train_prompt {train_set.data[0]}")
        prompt_data_list = []
        for data in train_set:
            prompt = PromptData(data["text"], data["label"])
            prompt_data_list.append(prompt)

        return prompt_data_list

    def get_test_prompt(self):
        test_set = self.dataset["test"]
        print(f"get_test_prompt {test_set.data[0]}")
        prompt_data_list = []
        for data in test_set:
            prompt = PromptData(data["text"], data["label"])
            prompt_data_list.append(prompt)

        return prompt_data_list
