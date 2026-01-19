import numpy as np


class PromptDetector:
    def __init__(self, threshold=3.0):
        self.threshold = threshold

    def get_z_scores(self, S_list):
        S = np.array(S_list, dtype=np.float32)
        mean_S = S.mean()
        std_S = S.std()
        eps = 1e-8

        z_scores = (S - mean_S) / (std_S + eps)
        is_invalid_std_S = std_S < 1e-6
        print(f"std_S:", std_S, "is valid", not is_invalid_std_S)
        if is_invalid_std_S:
            z_scores = np.zeros_like(S)

        return z_scores
