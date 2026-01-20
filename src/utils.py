import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model name (optional)"
    )

    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Limit number of prompts (optional)"
    )

    parser.add_argument(
        "--num_masks",
        type=int,
        default=None,
        help="Limit number of masked prompts per prompt (optional)"
    )

    return parser.parse_args()
