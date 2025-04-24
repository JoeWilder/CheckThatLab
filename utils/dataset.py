import csv
from torch.utils.data import Dataset
from typing import List, Tuple
from ftfy import fix_text
import random
import copy


class ClaimVerificationDataset(Dataset):
    def __init__(self, csv_path: str, fix_text_data: bool = True):
        self.csv_path = csv_path
        self.data = self.parse_csv(self.csv_path)
        self.fix_text_data = fix_text_data

    def parse_csv(self, csv_path: str) -> List[Tuple[str, str]]:
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

    def __getitem__(self, index: int) -> Tuple[str, str]:
        if self.fix_text_data:
            return {key: fix_text(value) for key, value in self.data[index].items()}
        else:
            return self.data[index]

    def generate_subset(self, percentage: float = 0.05) -> "ClaimVerificationDataset":
        subset_size = max(1, int(len(self) * percentage))
        diverse_subset = random.sample(self.data, subset_size)
        subdataset = copy.deepcopy(self)
        subdataset.data = diverse_subset
        return subdataset

    def export_subset_to_csv(self, output_csv_path: str, percentage: float = 0.05):
        subset = self.generate_subset(percentage)
        try:
            with open(output_csv_path, mode="w", newline="", encoding="utf8") as file:
                writer = csv.writer(file)
                writer.writerow(["text", "claim"])

                for sample in subset.data:
                    writer.writerow([sample["text"], sample["claim"]])

            print(f"Subset successfully exported to {output_csv_path}")
        except Exception as e:
            print(f"An error occurred while writing to CSV: {e}")
