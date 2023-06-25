from datasets import load_dataset, concatenate_datasets
from transformers import T5Tokenizer

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

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
    sample['tokenized_sentence']  = tokenizer(sample['sentence'], max_length=max_sentence, padding="max_length", truncation=True, return_tensors='pt')
    sample['tokenized_rationale'] = tokenizer(sample['rationale'], max_length=max_rationale, padding="max_length", truncation=True, return_tensors='pt')
    return sample

def get_datasets(tokenizer):
    # Used for testing. Can be safely removed later
    train_dataset, test_dataset, val_dataset = load_datasets()
    max_sentence, max_rationale = get_max_sizes(train_dataset, test_dataset, val_dataset)

    train_dataset = HateXplainDataset(train_dataset.map(lambda sample: tokenize_sample(tokenizer, sample, max_sentence, max_rationale)))
    test_dataset = HateXplainDataset(test_dataset.map(lambda sample: tokenize_sample(tokenizer, sample, max_sentence, max_rationale)))
    val_dataset = HateXplainDataset(val_dataset.map(lambda sample: tokenize_sample(tokenizer, sample, max_sentence, max_rationale)))

    return train_dataset, test_dataset, val_dataset

class HateXplainDataset(Dataset):
    def __init__(self, dataset):
        self.sentences = dataset["sentence"]
        self.rationale = dataset["rationale"]
        self.tokenized_sentence = dataset["tokenized_sentence"]
        self.tokenized_rationale = dataset["tokenized_rationale"]

    def __len__(self):
        return len(self.sentences)
     
    def __getitem__(self, index):
        return self.tokenized_sentence[index], self.tokenized_rationale[index]
    
class HateXplainDataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 32):
        self.batch_size = batch_size
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

        train_dataset, test_dataset, val_dataset = load_datasets()
        max_sentence, max_rationale = get_max_sizes(train_dataset, test_dataset, val_dataset)

        # Each entry consists of "sentence", "rationale", "tokenized_sentence", "tokenized_rationale"
        self.train_dataset = HateXplainDataset(train_dataset.map(lambda sample: tokenize_sample(self.tokenizer, sample, max_sentence, max_rationale)))
        self.test_dataset = HateXplainDataset(test_dataset.map(lambda sample: tokenize_sample(self.tokenizer, sample, max_sentence, max_rationale)))
        self.val_dataset = HateXplainDataset(val_dataset.map(lambda sample: tokenize_sample(self.tokenizer, sample, max_sentence, max_rationale)))

        print(len(self.train_dataset), len(self.test_dataset), len(self.val_dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)