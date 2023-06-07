from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pytorch_lightning as pl

class HateXPlainDataset(Dataset):
    def __init__(self, dataset, tokenizer):

        self.dataset = dataset.map(lambda x: {
        'post_tokens': self.preprocess_post_tokens(x['post_tokens']),
        'annotators': self.preprocess_annotators(x['annotators'])
        })
        data = [(entry['post_tokens'], entry['annotators']) for entry in self.dataset]

        inputs = []
        outputs = []
        for sentence, classification in data:
            input_string = f"How would you classify this sentence? '{sentence}'. Answer: '{classification}'"

            input_tokens = tokenizer.encode_plus(input_string, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            output_tokens = tokenizer.encode_plus(classification, padding="max_length", truncation=True, max_length=32, return_tensors="pt")

            inputs.append(input_tokens["input_ids"])
            outputs.append(output_tokens["input_ids"])

        self.input_ids = torch.cat(inputs, dim=0)
        self.output_ids = torch.cat(outputs, dim=0)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int):
        sentence = self.sentences[index]
        classification = self.classifications[index]
        return sentence, classification    

    def preprocess_post_tokens(self, sentence: list[str]):
        return " ".join(sentence)

    def preprocess_annotators(self, text: dict):
        SCORE = {0: 1, 1: 0, 2: 0.5}
        avg_score = sum(SCORE[score] for score in text['label']) / len(text['label'])
        return "Hate Speech" if avg_score > 0.666 else "Offensive" if avg_score > 0.333 else "Normal"


tokenizer = T5Tokenizer.from_pretrained("t5-small")
dataset = load_dataset('hatexplain')
HateXPlainDataset(dataset['train'], tokenizer)