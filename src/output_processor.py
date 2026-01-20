from datetime import datetime
from pathlib import Path
import os
import pandas as pd

global_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
global_script_dir = Path(__file__).parent.resolve()
global_root_folder = os.path.abspath(os.path.join(global_script_dir, os.pardir))
global_output_folder = os.path.join(global_root_folder, "output")
global_session_folder = os.path.join(global_output_folder, global_timestamp)

class OutputProcessor:
    def __init__(self):
        self.data = []
        self.excel_path = os.path.join(global_session_folder, f"uniguardian_{global_timestamp}.xlsx")

    def add_data(self, data):
        self.data.append(data)

    def write_output(self):
        if not os.path.exists(global_session_folder):
            os.makedirs(global_session_folder)
        for row in self.data:
            self.write_to_excel(row)

    def write_to_excel(self, data):
        new_row = pd.DataFrame([data])

        # If file does not exist → create it with mask_level sheet
        if not os.path.exists(self.excel_path):
            with pd.ExcelWriter(
                    self.excel_path,
                    engine="openpyxl",
                    mode="w"
            ) as writer:
                new_row.to_excel(writer, sheet_name="mask_level", index=False)

            print("output_file", self.excel_path)
            return

        # File exists → append to mask_level
        with pd.ExcelWriter(
                self.excel_path,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="overlay"
        ) as writer:
            try:
                df = pd.read_excel(self.excel_path, sheet_name="mask_level")
                df = pd.concat([df, new_row], ignore_index=True)
            except ValueError:
                # Sheet does not exist yet
                df = new_row

            df.to_excel(writer, sheet_name="mask_level", index=False)

        print("output_file", self.excel_path)

    def write_prompt_summary(self):
        df = pd.read_excel(self.excel_path, sheet_name="mask_level")

        grouped = df.groupby("prompt_id")

        rows = []
        for pid, g in grouped:
            idx_max = g["z_i"].idxmax()
            top_row = g.loc[idx_max]

            row = {
                "prompt_id": pid,
                "label": int(top_row["label"]),
                "num_masks": len(g),
                "z_max": float(top_row["z_max"]),
                "mean_S": float(top_row["mean_S"]),
                "std_S": float(top_row["std_S"]),
                "is_poisoned": bool(top_row["is_poisoned"]),
                "is_correct": bool(
                    (top_row["label"] == 1 and top_row["is_poisoned"]) or
                    (top_row["label"] == 0 and not top_row["is_poisoned"])
                ),
                "top_mask_id": int(top_row["mask_id"]),
                "top_masked_words": top_row["masked_words"],
            }
            rows.append(row)

        summary_df = pd.DataFrame(rows)

        with pd.ExcelWriter(
                self.excel_path,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="replace",
        ) as writer:
            summary_df.to_excel(writer, sheet_name="prompt_summary", index=False)


    def write_overall_summary(self, excel_path):
        df = pd.read_excel(excel_path, sheet_name="prompt_summary")

        TP = ((df["label"] == 1) & (df["is_poisoned"] == True)).sum()
        FP = ((df["label"] == 0) & (df["is_poisoned"] == True)).sum()
        FN = ((df["label"] == 1) & (df["is_poisoned"] == False)).sum()
        TN = ((df["label"] == 0) & (df["is_poisoned"] == False)).sum()

        num_poisoned = (df["label"] == 1).sum()
        num_clean = (df["label"] == 0).sum()

        summary = {
            "num_prompts": len(df),
            "num_poisoned": num_poisoned,
            "num_clean": num_clean,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "TPR (Recall)": TP / max(num_poisoned, 1),
            "FPR": FP / max(num_clean, 1),
            "Accuracy": (TP + TN) / max(len(df), 1),
        }

        summary_df = pd.DataFrame([summary])

        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            summary_df.to_excel(writer, sheet_name="overall_summary", index=False)
