from datetime import datetime
from pathlib import Path
import os
import json
import pandas as pd

global_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
global_script_dir = Path(__file__).parent.resolve()
global_root_folder = os.path.abspath(os.path.join(global_script_dir, os.pardir))
global_output_folder = os.path.join(global_root_folder, "output")
global_session_folder = os.path.join(global_output_folder, global_timestamp)


class OutputProcessor:
    def process_output(self):
        pass

    def write_output(self, data):
        if not os.path.exists(global_session_folder):
            os.makedirs(global_session_folder)
        self.write_to_excel(data)
        self.write_to_jsonl(data)

    def write_to_excel(self, data):
        excel_path = os.path.join(global_session_folder, f"uniguardian_{global_timestamp}.xlsx")

        new_row = pd.DataFrame([data])

        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
        print("output_file", excel_path)
        df.to_excel(excel_path, index=False, engine='openpyxl')

    def write_to_jsonl(self, data):
        output_file = os.path.join(global_session_folder, f"uniguardian_results_{global_timestamp}.jsonl")
        print("output_file", output_file)

        with open(output_file, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
