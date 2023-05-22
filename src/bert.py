from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from nltk.corpus import stopwords

from dataset import HateXPlainDataset

RESULTS = Path().absolute().parent.joinpath("results")

STOP_WORDS = stopwords.words('english')
CLASSIFICATION = {0: "NORMAL", 1: "OFFENSIVE", 2: "HATE SPEECH"}

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

    train_dataloader = DataLoader(train_dataset)
    test_dataloader = DataLoader(test_dataset)
    validation_dataloader = DataLoader(validation_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        print(epoch)
        model.train()
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0

            for batch in validation_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted_labels = torch.max(logits, dim=1)

                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)

            accuracy = total_correct / total_samples
            print(f"Epoch {epoch + 1}: Accuracy = {accuracy}")

    # Test
    total_correct = 0
    total_samples = 0

    results = []
    for batch in test_dataloader:
        model.eval()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)

        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)

        results.append(
            [batch["sentence"][0], CLASSIFICATION[int(labels.item())], CLASSIFICATION[int(predicted_labels.item())]]
        )
        
    df = pd.DataFrame(results, columns=["Sentence", "Label", "Prediction"])

    accuracy = total_correct / total_samples
    print(f"Validation Accuracy = {accuracy}")
    df.to_csv(RESULTS.joinpath("bert_results.csv"), index=False)

if __name__ == "__main__":
    main()