import pandas as pd
from datasets import load_dataset, concatenate_datasets


def preprocess_sample(sample: dict):
    # Sentences are currently saved as a list of words so we preprocess it into a sentence and remove stop words and add a prefix
    sample['sentence'] = " ".join([word for word in sample['post_tokens']])

    # Classify each sample based on the opinion of the annotators. We take the mean of opinions and classify it then.
    SCORE = {0: 1, 1: 0, 2: 0.5} # 0: Hate Speech, 1: Normal, 2: Offensive. We therefore have to rearange it.
    avg_score = sum(SCORE[score] for score in sample['annotators']['label']) / len(sample['annotators']['label'])
    sample['classification'] = "Hate Speech" if avg_score >= 0.666 else "Offensive" if avg_score >= 0.333 else "Normal"

    # Take the words that were highlighted by the annotators and try to create a rationale out of it
    rationale = [sum(elements) for elements in zip(*sample['rationales'])]
    words = [sample['post_tokens'][i] for i, rationale in enumerate(rationale) if rationale > 0]
    if len(words) > 0 and sample['classification'] != "Normal":
        explanation = "This sentence was classified as such because it contains harmful words such as: "
        sample['rationale'] = explanation + ", ".join([f"'{word}'" for word in words])
    elif sample['classification'] == "Hate Speech" or sample['classification'] == "Offensive":
        sample['rationale'] =  "This sentence has a harmful tone."
    else:
        sample['rationale'] = "This sentence was classified as normal because it contains no harmful words."

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
    max_c = max([len(sample) for sample in tokenized_inputs["classification"]])
    max_r = max([len(sample) for sample in tokenized_inputs["rationale"]])

    return max_s, max_c, max_r

if __name__ == "__main__":
    train_dataset, test_dataset, val_dataset = load_datasets()

    print([x["sentence"] for x in train_dataset if x["classification"] == "Offensive"])
    # max_sentence, max_classification, max_rationale = get_max_sizes(train_dataset, test_dataset, val_dataset)