import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration

from torch.utils.data import Dataset, DataLoader

def preprocess_sample(sample: dict):
    # Sentences are currently saved as a list of words so we preprocess it into a sentence and remove stop words and add a prefix
    sentence = " ".join([word for word in sample['post_tokens']])
    sample['sentence'] = "Classify as Hate Speech / Offensive / Normal: " + sentence

    # Classify each sample based on the opinion of the annotators. We take the mean of opinions and classify it then.
    SCORE = {0: 1, 1: 0, 2: 0.5} # 0: Hate Speech, 1: Normal, 2: Offensive. We therefore have to rearange it.
    avg_score = sum(SCORE[score] for score in sample['annotators']['label']) / len(sample['annotators']['label'])
    classification = "Hate Speech" if avg_score >= 0.666 else "Offensive" if avg_score >= 0.333 else "Normal"

    # Take the words that were highlighted by the annotators and try to create a rationale out of it
    rationale = [sum(elements) for elements in zip(*sample['rationales'])]
    words = [sample['post_tokens'][i] for i, rationale in enumerate(rationale) if rationale > 0]
    if len(words) > 0 and classification != "Normal":
        explanation = f"This sentence was classified as {classification} because it contains harmful words such as: "
        sample['rationale'] = "Rationale: " + explanation + ", ".join([f"'{word}'" for word in words])
    elif classification == "Hate Speech" or classification == "Offensive":
        sample['rationale'] =  "Rationale: " + f"This sentence has a harmful tone and was thus classified as {classification}"
    else:
        sample['rationale'] = "Rationale: " + "This sentence was classified as normal because it contains no harmful words."
    return sample

def load_datasets():
    # Loads the datasets from huggingface and preprocesses them
    dataset = load_dataset('hatexplain')
    train_dataset = dataset["train"].map(preprocess_sample, remove_columns=["id", "annotators", "rationales", "post_tokens"])
    test_dataset = dataset["test"].map(preprocess_sample, remove_columns=["id", "annotators", "rationales", "post_tokens"])
    val_dataset = dataset["validation"].map(preprocess_sample, remove_columns=["id", "annotators", "rationales", "post_tokens"])
    return train_dataset, test_dataset, val_dataset

def get_max_sizes(train_dataset, test_dataset, val_dataset):
    # Returns the size of the longest Sentence/Classification/Rationale
    tokenized_inputs = concatenate_datasets([train_dataset, test_dataset, val_dataset])

    max_s = max([len(sample) for sample in tokenized_inputs["sentence"]])
    max_r = max([len(sample) for sample in tokenized_inputs["rationale"]])

    return max_s, max_r

def tokenize_sample(tokenizer, sample, max_sentence, max_rationale):
    sample['sentence']  = tokenizer(sample['sentence'], max_length=max_sentence, padding="max_length", truncation=True, return_tensors='pt')
    sample['rationale'] = tokenizer(sample['rationale'], max_length=max_rationale, padding="max_length", truncation=True, return_tensors='pt')
    return sample

class HateXplainDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = len(dataset)

    def __len__(self):
        return self.len
     
    def __getitem__(self, index):
        return self.dataset[index]

if __name__ == "__main__":
    train_dataset, test_dataset, val_dataset = load_datasets()
    max_sentence, max_rationale = get_max_sizes(train_dataset, test_dataset, val_dataset)

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    train_dataset = train_dataset.map(lambda sample: tokenize_sample(tokenizer, sample, max_sentence, max_rationale))
    test_dataset = test_dataset.map(lambda sample: tokenize_sample(tokenizer, sample, max_sentence, max_rationale))
    val_dataset = val_dataset.map(lambda sample: tokenize_sample(tokenizer, sample, max_sentence, max_rationale))
    
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    outputs = model.generate(
        input_ids = torch.tensor(train_dataset[1]['sentence']['input_ids']),
        attention_mask = torch.tensor(train_dataset[1]['sentence']['attention_mask'])
    )
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
