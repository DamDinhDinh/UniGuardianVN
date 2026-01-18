import re
import random


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
        instruction_splited = re.split(rf'(\s|{tokenizer.eos_token[0]}.*?{tokenizer.eos_token[-1]})', instruction)
        print(instruction)
        print(instruction_splited)

        available_index = []
        for i in range(0, len(instruction_splited)):
            word = instruction_splited[i]
            if len(word) > 1:
                available_index.append(i)
            elif word.isprintable() and not word.isspace():
                available_index.append(i)

        available_len = len(available_index)
        num_masked_prompt = self.num_masked_instructions * available_len
        num_masks_per_prompt = int(min(max(1, round(available_len ** self.num_masks_per_instruction)),
                                       self.max_masks_per_instruction))

        print("Prompt: ", instruction)
        print("Length of words: ", available_len)
        print("Number of masked prompt: ", num_masked_prompt)
        print("Number of masked prompt per instruction: ", num_masks_per_prompt)

        masked_prompts = []
        # masked_prompts.append(instruction)
        for i in range(num_masked_prompt):
            masked_prompt = self.get_masked_prompt(instruction_splited, available_index, num_masks_per_prompt)
            masked_prompts.append(masked_prompt)

        return masked_prompts

    def get_masked_prompt(self, instruction_splited, available_index, m):
        masked_prompt = instruction_splited.copy()
        to_mask = random.sample(available_index, min(m, len(available_index)))
        for idx in to_mask:
            masked_prompt[idx] = self.mask
        return "".join(masked_prompt)
