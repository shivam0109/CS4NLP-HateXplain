import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


from datasets import load_dataset, concatenate_datasets

SCORE = {0: 1, 1: 0, 2: 0.5} # 0: Hate Speech, 1: Normal, 2: Offensive

def preprocess_sample(sample: dict) -> dict:
    sample["sentence"] = " ".join([word for word in sample['post_tokens']])
    avg_score = sum(SCORE[score] for score in sample['annotators']['label']) / len(sample['annotators']['label'])
    sample["classification"] = 1 if avg_score > 0.5 else 0
    return sample

def vectorize_sentences(sample: dict, vectorizer: CountVectorizer) -> dict:
    print(sample["sentence"])
    sample["sentence"] = vectorizer.transform(sample["sentence"]).toarray()
    return sample

def get_data() -> tuple:
    dataset = load_dataset('hatexplain')
    concatenated_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    concatenated_dataset = concatenated_dataset.map(preprocess_sample, remove_columns=["id", "annotators", "rationales", "post_tokens"])

    vectorizer = CountVectorizer()
    sentences = vectorizer.fit_transform(concatenated_dataset["sentence"]).toarray()
    classifications = np.array(concatenated_dataset["classification"])

    return sentences, classifications, vectorizer

def decision_tree_classifier(X_train, X_test, y_train, y_test, vectorizer) -> int:
    print("Decision Tree Classifier")
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

def main() -> None:
    sentences, classifications, vectorizer = get_data()
    X_train, X_test, y_train, y_test = train_test_split(sentences, classifications, test_size=0.2, random_state=42)

    decision_tree_classifier(X_train, X_test, y_train, y_test, vectorizer)
    








if __name__ == "__main__":
    main()

# sorted_indices = feature_importances.argsort()[::-1]

# num_top_words = 5
# top_words = [feature_names[idx] for idx in sorted_indices[:5]]
# print("Top words with the most influence:")
# for word in top_words:
#     print(word)