from src.prompt_data import PromptData
from src.utils import get_config, get_default_config
import re


class MaskProcessor:
    def __init__(self, mask="<unk>",
                 num_masked_instructions=2,
                 num_masks_per_instruction=0.3,
                 max_masks_per_instruction=8):
        self.mask = mask
        self.num_masked_instructions = num_masked_instructions
        self.num_masks_per_instruction = num_masks_per_instruction
        self.max_masks_per_instruction = max_masks_per_instruction

    def get_masked_prompts(self, tokenizer, prompt):
        instruction = prompt.prompt
        prompt = tokenizer.apply_chat_template(prompt.create_message(), add_generation_prompt=True, return_tensors="pt",
                                               tokenize=False)
        instruction_splited = re.split(rf'(\s|{tokenizer.eos_token[0]}.*?{tokenizer.eos_token[-1]})', instruction)
        prompt_splited = re.split(rf'(\s|{tokenizer.eos_token[0]}.*?{tokenizer.eos_token[-1]})', prompt)

        config = get_config("")

        print(instruction)
        print(instruction_splited)
        print(prompt_splited)

        begin_index = next((i for i in range(len(prompt_splited) - len(instruction_splited) + 1) if
                            prompt_splited[i:i + len(instruction_splited)] == instruction_splited), -1)
        # mask_indices_list = [[i, i + 1] for i in range(begin_index, begin_index + len(instruction_splited))]
        available_index = []
        for i in range(begin_index, begin_index + len(instruction_splited)):
            word = prompt_splited[i]
            if len(word) > 1:
                available_index.append(i)
            elif word.isprintable() and not word.isspace():
                available_index.append(i)

        K = config.detection.num_masks_per_instruction
        if isinstance(K, float):
            K = int(len(available_index) ** K)
        K = min(min(max(1, K), config.detection.max_masks_per_instruction), len(available_index))

        num_masked_instructions = config.detection.num_masked_instructions
        if isinstance(num_masked_instructions, float):
            num_masked_instructions = int(len(available_index) * num_masked_instructions)
        num_masked_instructions = max(1, num_masked_instructions)

        print("len:", len(available_index), "K:", K, "num_masked_instructions:", num_masked_instructions)

        # len_prompt = len(prompt.prompt)
        # num_masked_prompt = self.num_masked_instructions * len_prompt
        # num_masks_per_prompt = min(max(1, len_prompt ** self.num_masks_per_instruction), self.max_masks_per_instruction)
        # print("Length of prompts: ", len_prompt)
        # print("Number of masked prompt: ", num_masked_prompt)
        # print("Number of masked prompt per instruction: ", num_masks_per_prompt)
