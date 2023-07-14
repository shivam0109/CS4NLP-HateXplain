from datasets import load_dataset, concatenate_datasets
import torch
from transformers import T5Tokenizer

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class HateXplainDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.dataset[index]["input_ids"])
        attention_mask = torch.tensor(self.dataset[index]["attention_mask"])
        decoder_input_ids = torch.tensor(self.dataset[index]["decoder_input_ids"])
        decoder_attention_mask = torch.tensor(self.dataset[index]["decoder_attention_mask"])
        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask

class HateXplainDataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 32):
        super().__init__()

        self.batch_size = batch_size
        
        self.prepare_data_per_node = False

        self.max_sentence = 572
        self.max_rationale = 940 # Hardcoded as they are the same either way

    def preprocess_sample(self, sample: dict):
        # Sentences are currently saved as a list of words so we preprocess it into a sentence and remove stop words and add a prefix
        sentence = " ".join([word for word in sample['post_tokens']])
        sentence = "Classify as Hate Speech / Offensive / Normal: " + sentence

        # Classify each sample based on the opinion of the annotators. We take the mean of opinions and classify it then.
        SCORE = {0: 1, 1: 0, 2: 0.5} # 0: Hate Speech, 1: Normal, 2: Offensive. We therefore have to rearange it.
        avg_score = sum(SCORE[score] for score in sample['annotators']['label']) / len(sample['annotators']['label'])
        # classification = 2 if avg_score >= 0.666 else 1 if avg_score >= 0.333 else 0
        classification = "Hate Speech" if avg_score >= 0.666 else "Offensive" if avg_score >= 0.333 else "Normal"

        # Take the words that were highlighted by the annotators and try to create a rationale out of it
        rationale = [sum(elements) for elements in zip(*sample['rationales'])]
        words = [sample['post_tokens'][i] for i, rationale in enumerate(rationale) if rationale > 0]
        if len(words) > 0 and classification != "Normal":
            explanation = f"This sentence was classified as {classification} because it contains harmful words such as: "
            rationale = "Rationale: " + explanation + ", ".join([f"'{word}'" for word in words])
        elif classification == "Hate Speech" or classification == "Offensive":
            rationale =  "Rationale: " + f"This sentence has a harmful tone and was thus classified as {classification}"
        else:
            rationale = "Rationale: " + "This sentence was classified as normal because it contains no harmful words."

        sentence  = self.tokenizer(sentence, max_length=self.max_sentence, padding="max_length", truncation=True, return_tensors='pt')
        # sample["labels"] = torch.tensor([classification])

        rationale = self.tokenizer(rationale, max_length=self.max_rationale, padding="max_length", truncation=True, return_tensors='pt')
        sample["input_ids"] = sentence["input_ids"].squeeze()
        sample["attention_mask"] = sentence["attention_mask"].squeeze()
        sample["decoder_input_ids"] = rationale["input_ids"].squeeze()
        sample["decoder_attention_mask"] = rationale["attention_mask"].squeeze()

        return sample

    def prepare_data(self) -> None:
        self.dataset = load_dataset('hatexplain')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def setup(self, stage: str) -> None:
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.preprocess_sample,
                remove_columns=["id", "annotators", "rationales", "post_tokens"]
            )
            # self.dataset[split] = {entry for entry in self.dataset[split] if type(entry["input_ids"]) == torch.Tensor}

    def train_dataloader(self):
        return DataLoader(HateXplainDataset(self.dataset["train"]), shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(HateXplainDataset(self.dataset["test"]))
    
    def val_dataloader(self):
        return DataLoader(HateXplainDataset(self.dataset["validation"]))