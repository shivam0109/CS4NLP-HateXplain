import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from transformers import T5Tokenizer, T5Model
from datasets import load_dataset, concatenate_datasets

SCORE = {0: 1, 1: 0, 2: 0.5} # 0: Hate Speech, 1: Normal, 2: Offensive
STOP_WORDS = stopwords.words('english')
LEMMATIZER = WordNetLemmatizer()

class HateXplainDataset(Dataset):
    def __init__(self, sentences, classifications, tokenizer):
        self.inputs = tokenizer.batch_encode_plus(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        self.classifications = classifications

    def __len__(self):
        return len(self.classifications)
    
    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx]
        attention_mask = self.inputs["attention_mask"][idx]
        classification = self.classifications[idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "classifications": classification
        }

class HateXplainDataModule(pl.LightningDataModule):
    def __init__(self, sentences, classifications, tokenizer, batch_size=8):
        super().__init__()
        self.sentences = sentences
        self.classifications = classifications
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        X_train, X_test, y_train, y_test = train_test_split(self.sentences, self.classifications, test_size=0.2, random_state=42)
        self.train_dataset = HateXplainDataset(X_train, y_train, self.tokenizer)
        self.val_dataset = HateXplainDataset(X_test, y_test, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
class T5Finetuner(pl.LightningModule):
    def __init__(self, model_name, num_classes=2, learning_rate=1e-4):
        super().__init__()
        self.model = T5Model.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, num_classes)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.linear(outputs.last_hidden_state[:, 0])
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["classifications"]

        logits = self(input_ids, attention_mask)
        print(logits)

        loss = self.loss(logits, labels.float())

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["classifications"]

        logits = self(input_ids, attention_mask)
        loss = self.loss(logits, labels.float())

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def preprocess_sample(sample: dict) -> dict:
    words = [LEMMATIZER.lemmatize(word) for word in sample['post_tokens'] if word not in STOP_WORDS]
    sample["sentence"] = " ".join([word for word in words])
    avg_score = sum(SCORE[score] for score in sample['annotators']['label']) / len(sample['annotators']['label'])
    sample["classification"] = 1 if avg_score >= 0.5 else 0
    return sample

def get_data():
    dataset = load_dataset('hatexplain')
    concatenated_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    concatenated_dataset = concatenated_dataset.map(preprocess_sample, remove_columns=["id", "annotators", "rationales", "post_tokens"])

    sentences = list(concatenated_dataset["sentence"])
    classifications = np.array(concatenated_dataset["classification"])

    return sentences, classifications    

def main():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5Finetuner(model_name)

    sentences, classifications = get_data()
    datamodule = HateXplainDataModule(sentences[0:100], classifications[0:100], tokenizer)
    datamodule.setup()

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()