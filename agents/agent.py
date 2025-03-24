class Agent():
    def ask(self, prompt=[{"role": "user", "content": "How are you?"}]):
        raise NotImplementedError("ask method must be implemented in subclass")