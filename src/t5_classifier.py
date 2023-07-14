import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
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
    main()import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, concatenate_datasets
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer

SCORE = {0: 1, 1: 0, 2: 0.5} # 0: Hate Speech, 1: Normal, 2: Offensive
STOP_WORDS = stopwords.words('english')
LEMMATIZER = WordNetLemmatizer()


def preprocess_sample(sample: dict) -> dict:
    words = [LEMMATIZER.lemmatize(word) for word in sample['post_tokens'] if word not in STOP_WORDS]
    sample["sentence"] = " ".join([word for word in words])
    avg_score = sum(SCORE[score] for score in sample['annotators']['label']) / len(sample['annotators']['label'])
    sample["classification"] = 1 if avg_score >= 0.5 else 0
    return sample

def get_data(vectorize: bool = True):
    dataset = load_dataset('hatexplain')
    concatenated_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    concatenated_dataset = concatenated_dataset.map(preprocess_sample, remove_columns=["id", "annotators", "rationales", "post_tokens"])

    sentences = list(concatenated_dataset["sentence"])
    classifications = np.array(concatenated_dataset["classification"])

    return sentences, classifications

def imdb_reviews(
    dataset,
    prefix='sentiment',
    output_classes=('negative', 'positive')
    ):
  """Preprocessor to handle imdb movie reviews.

  Preprocessor converts an example from the IMDB movie review dataset to the
  text to text format. The following is an example from IMDB dataset.
  {
    'text': 'This is a bad movie. Skip it.'
    'label': 0,
  }

  The example will be transformed to the following format by the preprocessor:
  {
    'inputs': 'sentiment review: This is a bad movie. Skip it.'
    'targets': 'negative'
  }

  Examples with no label (-1) will have '<unk>' as their target.

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    output_classes: list of output classes in the input dataset. Defaults to
      ['negative', 'positive'] for the movie reviews dataset.

  Returns:
    a tf.data.Dataset

  """
  def my_fn(x):
    """Helper function to transform a rationale dataset to inputs/targets."""
    inputs = tf.strings.join([prefix + ':', x['text']], separator=' ')

    class_label = tf.cond(
        x['label'] > -1,
        lambda: tf.gather(output_classes, x['label']),
        lambda: '<unk>')

    return {'inputs': inputs, 'targets': class_label}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def map_sst2(str):
   return "sst2 sentence: " + str

def map_label(label):
   dict = {0: "positive", 1: "negative"}
   return dict[label]

def postprocess(str):
   str = str.replace(" ", "")
   str = str.replace("<pad>","")
   str = str.replace("</s>","")
   dict = {"negative": 1, "positive":0}
   return dict[str]

def predictor(texts): 
   tokenizer = T5Tokenizer.from_pretrained("t5-small")
   model = T5ForConditionalGeneration.from_pretrained("t5-small")
   enc = tokenizer.batch_encode_plus(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
   tokens = model.generate(**enc,output_scores=True,return_dict_in_generate=True)
   logits = tokens["scores"][0]
   selected_logits = logits[:, [1465, 2841]] 
   probs = F.softmax(selected_logits, dim=1).detach().cpu().numpy()
   return probs
   

   


def classify_t5(X_train, X_test, y_train, y_test):
    X_train_sst2 = list(map(map_sst2,X_train))
    #probs = predictor(X_train_sst2)
    explainer = LimeTextExplainer(class_names=["negative", "positive"])
    exp = explainer.explain_instance(X_train_sst2[0], predictor, num_features=6, labels=[0, 1])
    html_object = exp.as_html(labels=[0,1],predict_proba=True,show_predicted_value=True)
    with open(f"output/lime_t5.html", "w") as file:
        file.write(str(html_object))
    exit()

def main():
    sentences, classifications = get_data()
    X_train, X_test, y_train, y_test = train_test_split(sentences, classifications, test_size=0.2, random_state=42)
    X_train = X_train[:100]
    X_test = X_test[:100]
    y_train = y_train[:100]
    y_test = y_test[:100]
    classify_t5(X_train, X_test, y_train, y_test)
    exit()

if __name__ == "__main__":
    main()