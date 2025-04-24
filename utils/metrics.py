import nltk

nltk.download("wordnet")
from nltk.translate import meteor
from nltk import word_tokenize
from tqdm import tqdm
from typing import List, Dict

from utils.dataset import ClaimVerificationDataset
from agents.agent import Agent


def evaluate_claim_extraction(text_passage: str, claim: str, precision: int = 4):
    return round(meteor([word_tokenize(text_passage)], word_tokenize(claim)), precision)


def evaluate_on_dataset(test_dataset: ClaimVerificationDataset, agent: Agent, prompt: List[Dict], limit=None):
    data = []
    scores = []

    counter = 0
    for entry in tqdm(test_dataset, desc="Evaluating LLM claim extraction"):
        user_prompt = entry["text"]
        prompt.append({"role": "user", "content": user_prompt})
        output = agent.ask(prompt)
        meteor_score = evaluate_claim_extraction(entry["claim"], output)

        data.append({"ground_truth_claim": entry["claim"], "generated_claim": output, "meteor_score": meteor_score})

        scores.append(meteor_score)
        prompt.pop()

        counter += 1
        if limit and counter >= limit:
            break

    avg_score = sum(scores) / len(scores) if scores else 0

    return data, avg_score
