from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words('english')
T5_MODEL = "t5-small"

# TODO: Sentences contain also emojis, unrecognised symbols and <user>. How to handle?

def preprocess_sample(x):
    # Sentences are currently saved as a list of words so we preprocess it into a sentence and remove stop words and add a prefix
    prefix = "Sentence: "
    x['sentence'] = prefix + " ".join([word for word in x['post_tokens'] if word not in STOP_WORDS])

    # Classify each sample based on the opinion of the annotators. We take the mean of opinions and classify it then.
    SCORE = {0: 1, 1: 0, 2: 0.5} # 0: Hate Speech, 1: Normal, 2: Offensive. We therefore have to rearange it.
    avg_score = sum(SCORE[score] for score in x['annotators']['label']) / len(x['annotators']['label'])
    x['classification'] = "Classification: " + "Hate Speech" if avg_score > 0.666 else "Offensive" if avg_score > 0.333 else "Normal"

    # Take the words that were highlighted by the annotators and try to create a rationale out of it
    rationale = [sum(elements) for elements in zip(*x['rationales'])]
    words = [x['post_tokens'][i] for i, rationale in enumerate(rationale) if rationale > 0]
    if len(words) > 0:
        explanation = "This sentence was classified as such because it contains harmful words such as: "
        x['rationale'] = explanation + ", ".join([f"'{word}'" for word in words if word not in STOP_WORDS])
    else:
        x['rationale'] = "This sentence was classified as such because it contains no harmful words."

    return x

def tokenize_sample(sample, max_sentence_length, max_classification_length, max_rationale_length, padding="max_length"):
    model_inputs = tokenizer(sample['sentence'], max_length=max_sentence_length, padding=padding, truncation=True)
    model_inputs['tokenized_classification'] = tokenizer(sample['classification'], max_length=max_classification_length, padding=padding, truncation=True)
    model_inputs['tokenized_rationale'] = tokenizer(sample['rationale'], max_length=max_rationale_length, padding=padding, truncation=True)
    return model_inputs

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")


if __name__ == "__main__":
    dataset = load_dataset('hatexplain')
    train_dataset = dataset["train"].map(preprocess_sample, remove_columns=["id", "annotators", "rationales", "post_tokens"])
    test_dataset = dataset["test"].map(preprocess_sample, remove_columns=["id", "annotators", "rationales", "post_tokens"])
    val_dataset = dataset["validation"].map(preprocess_sample, remove_columns=["id", "annotators", "rationales", "post_tokens"])

    tokenized_inputs = concatenate_datasets([train_dataset, test_dataset, val_dataset])
    max_s = max([len(x) for x in tokenized_inputs["sentence"]])
    max_c = max([len(x) for x in tokenized_inputs["classification"]])
    max_r = max([len(x) for x in tokenized_inputs["rationale"]])


    print(max_s, max_c, max_r)

    # train_dataset = train_dataset.map(lambda x: tokenize_sample(x, max_s, max_c, max_r), remove_columns=["sentence", "classification", "rationale"])
    # test_dataset = test_dataset.map(lambda x: tokenize_sample(x, max_s, max_c, max_r), remove_columns=["sentence", "classification", "rationale"])
    # val_dataset = val_dataset.map(lambda x: tokenize_sample(x, max_s, max_c, max_r), remove_columns=["sentence", "classification", "rationale"])
    # print(train_dataset[0])

    # tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
    # model = T5Model