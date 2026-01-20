from src.mask_processor import MaskProcessor
from src.model_loader import ModelLoader
from src.prompt_attack_detector import PromptAttackDetector
from src.output_processor import *
from src.data_loader import DataLoader
from src.utils import *
import torch

MASK_BATCH_SIZE = 8
MAX_NEW_TOKENS = 64

args = parse_args()

if args.max_batch_size is not None:
    MASK_BATCH_SIZE = args.max_batch_size

if args.max_new_tokens is not None:
    MAX_NEW_TOKENS = args.max_new_tokens

output_processor = OutputProcessor()

data_loader = DataLoader()

prompt_list = data_loader.get_train_prompt()
if args.max_prompts is not None:
    prompt_list = prompt_list[:args.max_prompts]

model_loader = ModelLoader() if args.model is None \
    else ModelLoader(args.model)

tokenizer = model_loader.get_tokenizer()
model = model_loader.get_model()
masked_processor = MaskProcessor()

print("prompt list length", len(prompt_list))
for iterator, prompt_data in enumerate(prompt_list):
    print(f"Input {iterator}: label", prompt_data.label)

    base_inputs = tokenizer(prompt_data.prompt, return_tensors="pt").to(model_loader.device)
    base_input_ids = base_inputs["input_ids"][0]
    with torch.no_grad():
        print("Base prompt: ", prompt_data.prompt)
        masked_data = masked_processor.get_masked_prompts(
            tokenizer,
            prompt_data
        ) if args.num_masks is None \
            else masked_processor.get_masked_prompts(
            tokenizer,
            prompt_data
        )[:args.num_masks]

        all_prompts = [prompt_data.prompt] + [m.prompt for m in masked_data]

        inputs = tokenizer(
            all_prompts,
            return_tensors="pt",
            padding=True
        ).to(model_loader.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        past_key_values = None
        S = torch.zeros(len(masked_data), device=model_loader.device)

        for step in range(MAX_NEW_TOKENS):
            outputs = model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )

            logits = outputs.logits[:, -1, :]  # (1 + N, vocab)
            past_key_values = outputs.past_key_values

            base_logits = logits[0]
            base_token = torch.argmax(base_logits)

            for i in range(len(masked_data)):
                Li = logits[i + 1, base_token]
                Lb = base_logits[base_token]
                S[i] += (torch.sigmoid(Li) - torch.sigmoid(Lb)) ** 2

            # force SAME token for everyone
            next_tokens = base_token.unsqueeze(0).repeat(input_ids.size(0), 1)
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_tokens)], dim=1
            )

        S_list = (S / MAX_NEW_TOKENS).tolist()

        prompt_attack_detector = PromptAttackDetector()
        z_scores, mean_S, std_S = prompt_attack_detector.process_data(S_list)
        print("mean_S", mean_S)
        print("std_S", std_S)
        print(f"z_scores:", z_scores)

        z_max, threshold, is_poisoned = prompt_attack_detector.detect_by_max_z(z_scores)
        print("z_max", z_max)
        print("threshold", threshold)
        print("is_poisoned", is_poisoned)

        is_detected = prompt_data.label == 1 and is_poisoned
        is_false_alarm = prompt_data.label == 0 and is_poisoned
        is_missed = prompt_data.label == 1 and not is_poisoned

        print("is_detected", is_detected)
        print("is_false_alarm", is_false_alarm)
        print("is_missed", is_missed)

        for i, masked in enumerate(masked_data):
            output_processor.add_data(
                {
                    # Prompt-level
                    "prompt_id": iterator,
                    "label": prompt_data.label,
                    "base_prompt": prompt_data.prompt,

                    # Mask-level
                    "mask_id": i,
                    "masked_prompt": masked.prompt,
                    "masked_words": masked.words,
                    "masked_indexes": masked.indexes,

                    # UniGuardian math
                    "S_i": S_list[i],
                    "z_i": z_scores[i],
                    "z_max": z_max,
                    "mean_S": mean_S,
                    "std_S": std_S,
                    "threshold": threshold,
                    "is_poisoned": is_poisoned,

                    # Metadata
                    "timestamp": global_timestamp,
                }
            )

print("Writing output...")
output_processor.write_output()
print("Done!")
