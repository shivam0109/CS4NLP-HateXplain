import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from transformers import T5Tokenizer, T5ForSequenceClassification
from datasets import load_dataset, concatenate_datasets

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

def classify_t5(X_train, X_test, y_train, y_test):
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

def classify_t5() -> None:
    model_name = 'bert-base-uncased'
    model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    sentences, classifications = get_data(vectorize=False)
    prefix = "Is the following sentence normal or hate speech? "
    sentences = [prefix + sentence for sentence in sentences]

    X_train, X_test, y_train, y_test = train_test_split(sentences, classifications, test_size=0.2, random_state=42)

    train_inputs = tokenizer.batch_encode_plus(
        X_train,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    test_inputs = tokenizer.batch_encode_plus(
        X_test,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )


    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    # predictions = [0 if "0" in prediction else 1 for prediction in predictions]
    predictions = torch.argmax(outputs.logits, dim=1).tolist()
    print(classifications)
    print(predictions)
    print(outputs.logits)
    
    # accuracy = accuracy_score(classifications, predictions)
    # f1 = f1_score(classifications, predictions)

    # print("Accuracy:", accuracy)
    # print("F1-score:", f1)

def main():
    sentences, classifications = get_data()
    X_train, X_test, y_train, y_test = train_test_split(sentences, classifications, test_size=0.2, random_state=42)

if __name__ == "__main__":
    main()