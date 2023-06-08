from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pytorch_lightning as pl

STOP_WORDS = stopwords.words('english')
LEMMATIZER = WordNetLemmatizer()

class HateXPlainDataset(Dataset):
    def __init__(self, dataset):

        self.dataset = dataset.map(lambda x: {
        'post_tokens': self.preprocess_post_tokens(x['post_tokens']),
        'annotators': self.preprocess_annotators(x['annotators'])
        })

        self.sentences = [sentence['post_tokens'] for sentence in self.dataset]
        self.classifications = [label['annotators'] for label in self.dataset]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        classification = self.classifications[index]
        return sentence, classification    

    def preprocess_post_tokens(self, text):
        sentences = [LEMMATIZER.lemmatize(word.lower()) for word in text if word not in STOP_WORDS]
        return " ".join(sentences)

    def preprocess_annotators(self, text):
        SCORE = {0: 1, 1: 0, 2:.5}

        scores = [SCORE[score] for score in text['label']]
        avg_score = sum(scores) / len(scores)
        if avg_score > 0.666:
            return [0, 0, 1] # Hate Speech
        elif avg_score > 0.333:
            return [0, 1, 0] # Offensive
        else:
            return [1, 0, 0] # Normal
        
    def extract_sensitive_words(self, text, rationales)
        
class HateXplainDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def setup(self, stage=None):
        dataset = load_dataset('hatexplain')
        self.train_dataset = HateXPlainDataset(dataset['train'])
        self.test_dataset = HateXPlainDataset(dataset['test'])
        self.val_dataset = HateXPlainDataset(dataset['validation'])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        sentences, classifications = zip(*batch)

        encoded_inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        classifications = torch.tensor(classifications).float()

        return input_ids, attention_mask, classifications, sentences