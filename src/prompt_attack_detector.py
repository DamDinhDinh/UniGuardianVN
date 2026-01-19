import numpy as np


class PromptAttackDetector:
    def __init__(self, threshold=3.0):
        self.threshold = threshold

    def process_data(self, S_list):
        S = np.array(S_list, dtype=np.float32)
        mean_S = S.mean()
        std_S = S.std()
        eps = 1e-8

        z_scores = (S - mean_S) / (std_S + eps)
        is_invalid_std_S = std_S < 1e-6

        if is_invalid_std_S:
            z_scores = np.zeros_like(S)

        return z_scores, mean_S, std_S

    def detect_by_max_z(self, z_scores):
        z_max = float(z_scores.max())
        is_poisoned = z_max > self.threshold
        return z_max, self.threshold, is_poisoned
