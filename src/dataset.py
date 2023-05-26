import torch
from torch.utils.data import Dataset

class HateXPlainDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer) -> None:
        super().__init__()
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.sentences[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        label = torch.tensor(self.labels[idx])

        return {
            "input_ids": input_ids,
            "sentences": self.sentences[idx],
            "attention_mask": attention_mask,
            "label": label
        }