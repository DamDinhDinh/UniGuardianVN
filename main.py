from src.mask_processor import MaskProcessor
from src.model_loader import ModelLoader
from src.prompt_attack_detector import PromptAttackDetector
from src.output_processor import *
from src.data_loader import DataLoader
from src.utils import *
import torch

args = parse_args()

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
        base_out = model.generate(
            **base_inputs,
            do_sample=True,
            temperature=0.1,
            top_p=1.0,
            max_new_tokens=64,
            min_new_tokens=8,
            return_dict_in_generate=True,
            output_logits=True,
        )

        base_logits_seq = base_out.logits
        base_logits_len = len(base_logits_seq)

        base_prompt_length = base_inputs.input_ids.shape[1]
        full_generated_ids = base_out.sequences[0]
        base_generated_ids = full_generated_ids[base_prompt_length:]
        base_generated_text = tokenizer.decode(
            base_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        print("base_logits_len", base_logits_len)
        print("Base response: \n", base_generated_text)
        print("\n")

        S_list = []
        masked_responses = []
        masked_data = masked_processor.get_masked_prompts(
            tokenizer,
            prompt_data
        ) if args.num_masks is None \
            else masked_processor.get_masked_prompts(
            tokenizer,
            prompt_data
        )[:args.num_masks]
        print("masked data length", len(masked_data))

        for i, masked in enumerate(masked_data):
            masked_inputs = tokenizer(masked.prompt, return_tensors="pt").to(model_loader.device)
            masked_input_ids = masked_inputs["input_ids"][0]
            print(f"Masked prompt {i}: \n", masked.prompt)
            print("\n")

            masked_prompt_len = masked_inputs.input_ids.shape[1]
            forced_ids = torch.cat(
                [
                    masked_inputs.input_ids,
                    base_generated_ids.unsqueeze(0)
                ],
                dim=1
            )
            attention_mask = torch.ones_like(forced_ids)
            masked_out = model(
                input_ids=forced_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            masked_logits_seq = masked_out.logits[
                :, masked_prompt_len - 1: masked_prompt_len - 1 + base_logits_len, :
            ]

            masked_logits_len = len(masked_logits_seq)

            masked_prompt_length = masked_inputs.input_ids.shape[1]
            full_masked_generated_ids = masked_out.sequences[0]
            masked_generated_ids = full_masked_generated_ids[masked_prompt_length:]
            masked_generated_text = tokenizer.decode(
                masked_generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()
            masked_responses.append(masked_generated_text)

            print(f"masked_{i}_logits_len", masked_logits_len)
            print(f"Masked response {i}: \n", masked_generated_text)
            print("\n")

            k = min(base_logits_len, masked_logits_len)
            scores = []

            base_gen_ids = base_out.sequences[0][-base_logits_len:]
            for j in range(k):
                token_id = base_generated_ids[j].item()

                Lb_j = base_logits_seq[j][0, token_id]
                Li_j = masked_logits_seq[0, j, token_id]

                diff = torch.sigmoid(Li_j) - torch.sigmoid(Lb_j)
                scores.append(diff ** 2)

            S_i = torch.stack(scores).mean().item()
            S_list.append(S_i)
            print(f"Uncertainty score {i}:", S_i)

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
                    "base_response": base_generated_text,

                    # Mask-level
                    "mask_id": i,
                    "masked_prompt": masked.prompt,
                    "masked_words": masked.words,
                    "masked_indexes": masked.indexes,
                    "masked_response": masked_responses[i],

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
