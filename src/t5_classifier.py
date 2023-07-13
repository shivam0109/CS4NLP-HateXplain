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