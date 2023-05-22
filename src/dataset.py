from nltk.corpus import stopwords
import torch
from torch.utils.data import Dataset

STOP_WORDS = stopwords.words('english')

class HateXPlainDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer) -> None:
        super().__init__()

        if not len(sentences) == len(labels):
            raise Exception("Unequal amount of labels and shapes")

        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "sentence": sentence,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label)
        }