from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from datasets import Dataset

import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")


import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth import is_bfloat16_supported

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

import sys
import os


class LlamaAgent:
    def __init__(self, model_id="meta-llama/Llama-3.2-1B-Instruct", device="cuda", temperature: float = 0.1, top_p: float = 0.8):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)

        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_pipeline = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, temperature=temperature, top_p=top_p, return_full_text=False)

    def ask(self, prompt=[{"role": "user", "content": "How are you?"}]):
        formatted_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)

        output = self.model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def format_for_finetuning(self, tokenizer, custom_dataset):
        formatted_prompts = []
        output_texts = []

        system_prompt = {
            "role": "system",
            "content": "You are an AI assistant designed to extract claims from a given passage of text. Keep it short and return the claim in the text. Only return the big idea and exclude unneeded details.",
        }

        for item in custom_dataset:
            text = item["text"]
            claim = item["claim"]

            prompt = [system_prompt, {"role": "user", "content": text}, {"role": "assistant", "content": claim}]

            print(prompt)

            formatted_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

            formatted_prompts.append(formatted_prompt)
            output_texts.append(claim)

        return {"input_text": formatted_prompts, "output_text": output_texts}

    def set_model(self, model_path: str):
        """
        Load a fine-tuned model from the specified directory.
        """
        print(f"Loading fine-tuned model from {model_path}...")

        original_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

        model, tokenizer = FastLanguageModel.from_pretrained(model_path, load_in_4bit=True, device_map="cuda")

        tokenizer.chat_template = original_tokenizer.chat_template

        # Apply LoRA adapters
        # model = FastLanguageModel.get_peft_model(model)

        self.model = model
        self.tokenizer = tokenizer

        self.generation_pipeline = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, temperature=self.temperature, top_p=self.top_p, return_full_text=False)

        print("Fine-tuned model loaded successfully!")

    @staticmethod
    def finetune(model_id, dataset_csv: str, output_dir="./output", num_train_epochs=3, batch_size=1, learning_rate=2e-5):
        """
        Fine-tune the Llama model using the ClaimVerificationDataset with unsloth.
        """

        max_seq_length = 1024
        lora_rank = 16
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_id, max_seq_length=max_seq_length, load_in_4bit=True, dtype=None)
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
            use_rslora=True,
            use_gradient_checkpointing="unsloth",
            random_state=32,
            loftq_config=None,
        )

        data = pd.read_csv("data/reduced-train-eng.csv")
        data["Context_length"] = data["text"].apply(len)
        filtered_data = data[data["Context_length"] <= 1500]

        data_prompt = """Analyze the provided text from a fact verification perspecitve. Extract the main claim from the text in a short response.

        ### Input:
        {}

        ### Response:
        {}"""

        EOS_TOKEN = tokenizer.eos_token

        def formatting_prompt(examples):
            inputs = examples["text"]
            outputs = examples["claim"]
            texts = []
            for input_, output in zip(inputs, outputs):
                text = data_prompt.format(input_, output) + EOS_TOKEN
                texts.append(text)
            return {
                "text": texts,
            }

        training_data = Dataset.from_pandas(filtered_data)
        training_data = training_data.map(formatting_prompt, batched=True)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=training_data,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=1,
            packing=True,
            args=TrainingArguments(
                learning_rate=3e-4,
                lr_scheduler_type="linear",
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                num_train_epochs=1,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                warmup_steps=10,
                output_dir="output",
                seed=0,
            ),
        )

        trainer.train()

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        return model, tokenizer


# if __name__ == "__main__":

#    train_dataset = ClaimVerificationDataset(f"data/train-eng.csv")
#    test_dataset = ClaimVerificationDataset(f"data/dev-eng.csv")

#    print(f"Train dataset length: {len(train_dataset)}")
#    print(f"Test dataset length: {len(test_dataset)}")
