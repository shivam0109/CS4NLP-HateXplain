from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.corpus import stopwords
from datasets import load_dataset

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from dataset import HateXPlainDataset


torch.manual_seed(42)

RESULTS = Path().absolute().parent.joinpath("results")

STOP_WORDS = stopwords.words('english')
CLASSIFICATION = {0: "NORMAL", 1: "OFFENSIVE", 2: "HATE SPEECH"}


class BERTclassifier(torch.nn.Module):
    def __init__(self):
        super(BERTclassifier, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.l2 = torch.nn.Linear(768, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 3)
        self.criterion = torch.nn.CrossEntropyLoss()

        # self.l1.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        l1_out = self.l1(input_ids, attention_mask=attention_mask)
        l2_out = self.l2(self.dropout(l1_out["pooler_output"]))
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)
        return l4_out
    
    def loss_fn(self, output, label):
        softmax = F.softmax(output, dim=1)
        loss = self.criterion(softmax, label)
        return loss

def preprocess_post_tokens(text):
    sentences = [word for word in text if word not in STOP_WORDS]
    # return sentences
    return " ".join(sentences)

def preprocess_annotators(text):
    SCORE = {0: 1, 1: 0, 2:.5}

    scores = [SCORE[score] for score in text['label']]
    avg_score = sum(scores) / len(scores)
    if avg_score >= 0.666: # Hate
        return 2
    elif avg_score >= 0.333: # Offensive
        return 1
    else:
        return 0

def validation(model, validation_dataloader, device):
    f1_metric = MulticlassF1Score(num_classes=3).to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=3).to(device)
    model.eval()
    with torch.no_grad():
        for batch in validation_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, max_indices = torch.max(outputs, dim=1)
            f1_metric.update(max_indices, labels)
            accuracy_metric.update(max_indices, labels)

    f1 = f1_metric.compute()
    accuracy = accuracy_metric.compute()
    print(f"F1: {f1}, Accuracy: {accuracy}")

def train(model, train_dataloader, optimizer, device):
    model.train()
    for _, batch in enumerate(train_dataloader, 0):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask = attention_mask)
        loss = model.loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, test_dataloader, device):
    f1_metric = MulticlassF1Score(num_classes=3).to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=3).to(device)

    model.eval()
    results = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            sentences = batch["sentences"]
            outputs = model(input_ids, attention_mask=attention_mask)
            _, max_indices = torch.max(outputs, dim=1)
            f1_metric.update(max_indices, labels)
            accuracy_metric.update(max_indices, labels)

            out_labels = [CLASSIFICATION[entry] for entry in max_indices.tolist()]
            out_truth = [CLASSIFICATION[entry] for entry in labels.tolist()]

            results += list(zip(sentences, out_labels, out_truth))

    results += [("Result", f"F1: {f1_metric.compute()}", f"Accuracy: {accuracy_metric.compute()}")]
    
    df = pd.DataFrame(results, columns=["Sentence", "Label", "Classification"])
    df.to_csv(RESULTS.joinpath("bert.csv"), index=False)

def main():
    hatexplain = load_dataset('hatexplain')
    hatexplain = hatexplain.map(lambda x: {
        'post_tokens': preprocess_post_tokens(x['post_tokens']),
        'annotators': preprocess_annotators(x['annotators'])
        })
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = HateXPlainDataset(
        sentences = hatexplain['train']['post_tokens'],
        labels = hatexplain['train']['annotators'],
        tokenizer = tokenizer
    )
    test_dataset = HateXPlainDataset(
        sentences = hatexplain['test']['post_tokens'],
        labels = hatexplain['test']['annotators'],
        tokenizer = tokenizer
    )
    validation_dataset = HateXPlainDataset(
        sentences = hatexplain['validation']['post_tokens'],
        labels = hatexplain['validation']['annotators'],
        tokenizer = tokenizer
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTclassifier()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        print(f"Epoch {epoch}:")
        train(model, train_dataloader, optimizer, device)
        validation(model, validation_dataloader, device)

    test(model, test_dataloader, device)

if __name__ == "__main__":
    main()