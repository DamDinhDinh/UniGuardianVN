class PromptData:
    def __init__(self, prompt, label):
        self.prompt = prompt
        self.label = label

    def create_message(self, system_prompt = ""):
        messages = [
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"{self.prompt}"},
        ]
        return messages
