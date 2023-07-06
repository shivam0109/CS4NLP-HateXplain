import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, concatenate_datasets
import torch.nn.functional as F

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
   dict = {0: "negative", 1: "positive"}
   return dict[label]

def postprocess(str):
   str = str.replace(" ", "")
   str = str.replace("<pad>","")
   str = str.replace("</s>","")
   dict = {"negative": 1, "positive":0}
   return dict[str]


def classify_t5(X_train, X_test, y_train, y_test):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    X_train_sst2 = list(map(map_sst2,X_train))
    enc = tokenizer.batch_encode_plus(
        X_train_sst2,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    tokens = model.generate(**enc,output_scores=True,return_dict_in_generate=True)
    logits = tokens["scores"][0]
    selected_logits = logits[:, [1465, 2841]] 
    probs = F.softmax(selected_logits, dim=1)
    decoded = list(map(postprocess,tokenizer.batch_decode(tokens["sequences"])))
    print(sum(1 for x,y in zip(decoded,y_train) if x == y) / len(decoded))
    print(probs)
    print(decoded)
    exit()
    model_name = 't5-base'
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
    t5_model = T5ForSequenceClassification.from_pretrained(model_name)
    classification_head = nn.Linear(t5_model.config.hidden_size, 2)

    train_inputs = t5_tokenizer.batch_encode_plus(
        X_train,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    train_outputs = t5_model(**train_inputs)
    last_hidden_state = train_outputs.last_hidden_state
    train_logits = classification_head(last_hidden_state[:, 0, :])

    test_inputs = t5_tokenizer.batch_encode_plus(
        X_test,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    test_outputs = t5_model(**test_inputs)
    last_hidden_state = test_outputs.last_hidden_state
    test_logits = classification_head(last_hidden_state[:, 0, :])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Perform forward pass and calculate loss
    logits = classification_head(last_hidden_state[:, 0, :])
    loss = loss_fn(logits, labels)

    # Perform backward pass and update model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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