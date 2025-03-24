import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import csv
from torch.utils.data import Dataset as TorchDataset
from typing import List, Tuple
from ftfy import fix_text
import random
import copy

class ClaimVerificationDataset(TorchDataset):
    def __init__(self, csv_path: str, fix_text_data: bool = True):
        self.csv_path = csv_path
        self.data = self.parse_csv(self.csv_path)
        self.fix_text_data = fix_text_data

    def parse_csv(self, csv_path: str) -> List[dict]:
        csv_data = []
        try:
            with open(csv_path, "r", encoding="utf8") as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # skip header

                for row in csv_reader:
                    csv_data.append({"text": row[0], "claim": row[1]})
            return csv_data

        except FileNotFoundError:
            print(f"Error: File not found at '{csv_path}'")
        except Exception as e:
            print(f"An error occurred: {e}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        if self.fix_text_data:
            return {key: fix_text(value) for key, value in self.data[index].items()}
        else:
            return self.data[index]
        
    def generate_subset(self, percentage: float = 0.05) -> 'ClaimVerificationDataset':
        subset_size = max(1, int(len(self) * percentage))
        diverse_subset = random.sample(self.data, subset_size)
        subdataset = copy.deepcopy(self)
        subdataset.data = diverse_subset
        return subdataset
    
    def export_subset_to_csv(self, output_csv_path: str, percentage: float = 0.05):
        subset = self.generate_subset(percentage)
        try:
            with open(output_csv_path, mode='w', newline='', encoding='utf8') as file:
                writer = csv.writer(file)
                writer.writerow(["text", "claim"])
                
                for sample in subset.data:
                    writer.writerow([sample["text"], sample["claim"]])
            
            print(f"Subset successfully exported to {output_csv_path}")
        except Exception as e:
            print(f"An error occurred while writing to CSV: {e}")


class LLMFineTuner:
    def __init__(
        self,
        model_name="meta-llama/Llama-2-7b-hf",
        output_dir="./finetuned_model",
        max_length=512,
        use_peft=True
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.use_peft = use_peft
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def prepare_dataset_from_claimverification(self, claim_dataset):
        """
        Converts ClaimVerificationDataset to a formatted dataset for instruction fine-tuning.
        """
        # Convert to instruction format
        formatted_data = []
        
        for i in tqdm(range(len(claim_dataset)), desc="Formatting dataset"):
            sample = claim_dataset[i]
            
            instruction = f"Normalize the following text into a clear claim:\n{sample['text']}"
            response = sample['claim']
            
            if "llama" in self.model_name.lower():
                prompt = f"<s>[INST] {instruction} [/INST] {response} </s>"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
                
            formatted_data.append({"formatted_text": prompt})
        
        hf_dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
        
        return hf_dataset
    
    def tokenize_dataset(self, dataset):
        """Tokenize the dataset for training"""
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["formatted_text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["formatted_text"]
        )
        
        return tokenized_dataset
    
    def load_model(self):
        """Load the base model with appropriate configuration"""
        print(f"Loading model: {self.model_name}")
        
        if self.device == "cuda":
            if not self.use_peft:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    load_in_4bit=True,
                    device_map="auto"
                )
                
                model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map={"": self.device}
            )
        
        if self.use_peft:
            print("Applying LoRA adapter for parameter-efficient fine-tuning")
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            model = get_peft_model(model, peft_config)
            
        model.config.use_cache = False  # Disable KV cache for training
        return model
    
    def train(self, train_dataset, val_dataset=None, epochs=3, batch_size=4, gradient_accumulation_steps=8):

        
        print("Preparing datasets...")
        train_data = self.prepare_dataset_from_claimverification(train_dataset)
        train_data = self.tokenize_dataset(train_data)
        
        if val_dataset:
            val_data = self.prepare_dataset_from_claimverification(val_dataset)
            val_data = self.tokenize_dataset(val_data)
        else:
            split_data = train_data.train_test_split(test_size=0.1)
            train_data = split_data["train"]
            val_data = split_data["test"]
        
        model = self.load_model()
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            learning_rate=2e-5,
            weight_decay=0.01,
            fp16=self.device == "cuda",
            load_best_model_at_end=True,
            report_to="none"  # Disable wandb or other reporting
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator
        )
        
        print("Starting training...")
        trainer.train()
        
        print(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return model
    
    def generate_claim(self, text, max_new_tokens=100, temperature=0.7):
        """Generate a normalized claim from input text using the fine-tuned model"""
        
        if "llama" in self.model_name.lower():
            prompt = f"<s>[INST] Normalize the following text into a clear claim:\n{text} [/INST]"
        else:
            prompt = f"### Instruction:\nNormalize the following text into a clear claim:\n{text}\n\n### Response:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        if not hasattr(self, 'model'):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.output_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map={"": self.device}
                )
            except:
                print("Fine-tuned model not found. Loading base model.")
                self.model = self.load_model()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "llama" in self.model_name.lower():
            response = generated_text.split("[/INST]")[-1].strip()
        else:
            response = generated_text.split("### Response:")[-1].strip()
            
        return response


from huggingface_hub import login

def main():

    login()

    csv_path = "data/train-eng.csv"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir = "./claim_normalizer_model"
    
    print("Loading dataset...")
    full_dataset = ClaimVerificationDataset(csv_path)
    
    train_dataset = full_dataset.generate_subset(percentage=0.2)  # 20% of data
    
    val_dataset = train_dataset.generate_subset(percentage=0.1)
    
    print("Initializing fine-tuner...")
    finetuner = LLMFineTuner(
        model_name=model_name,
        output_dir=output_dir,
        use_peft=True
    )
    
    print("Starting training...")
    finetuner.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=3,
        batch_size=4,
        gradient_accumulation_steps=8
    )
    
    test_text = "Lieutenant Retired General Asif Mumtaz appointed as Chairman Pakistan Medical Commission PMC"
    normalized_claim = finetuner.generate_claim(test_text)
    print(f"Original text: {test_text}")
    print(f"Generated claim: {normalized_claim}")


if __name__ == "__main__":
    main()