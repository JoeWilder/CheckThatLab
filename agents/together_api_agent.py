from together import Together

class TogetherAgent():
    def __init__(self, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        self.client = Together()
        self.model = model

    def ask(self,
            prompt=[{"role": "user", "content": "How are you?"}]):
        """Ask the given model the given prompt"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt
        )
        return response.choices[0].message.content