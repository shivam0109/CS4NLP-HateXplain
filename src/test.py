import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

def vectorize_sentences(sample: dict, vectorizer: CountVectorizer) -> dict:
    sample["sentence"] = vectorizer.transform(sample["sentence"]).toarray()
    return sample

def get_data(vectorize: bool = True):
    dataset = load_dataset('hatexplain')
    concatenated_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    concatenated_dataset = concatenated_dataset.map(preprocess_sample, remove_columns=["id", "annotators", "rationales", "post_tokens"])

    if vectorize is True:
        vectorizer = CountVectorizer()
        sentences = vectorizer.fit_transform(concatenated_dataset["sentence"]).toarray()
        classifications = np.array(concatenated_dataset["classification"])

        percentage_of_zeros = np.count_nonzero(classifications == 0) / len(classifications)
        print("Percentage of zeros:", percentage_of_zeros)

        return sentences, classifications, vectorizer

    else:
        sentences = list(concatenated_dataset["sentence"])
        classifications = np.array(concatenated_dataset["classification"])

        return sentences, classifications

def analyze_svm(X_train, X_test, y_train, y_test, vectorizer, classifier: SVC) -> None:
    print("Current Classifier: ", type(classifier).__name__)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

def analyze_classifier(X_train, X_test, y_train, y_test, vectorizer, classifier) -> None:
    print("Current Classifier: ", type(classifier).__name__)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    vocabulary_list = [word for word, _ in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])]

    feature_importances = list(zip(vocabulary_list, classifier.feature_importances_))
    sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    print("Top words with the most influence (Decision Tree):")
    for i in range(10):
        print("Word: '", sorted_features[i][0], "'. Importance:", sorted_features[i][1])

    words = [item[0] for item in sorted_features[:10]]
    importance = [item[1] for item in sorted_features[:10]]

    fig, ax = plt.subplots()
    ax.bar(words, importance)
    ax.set_ylabel('Importance')
    ax.set_xlabel('Word')
    ax.set_title(f'Top 10 words with the most influence {type(classifier).__name__}')
    plt.xticks(rotation=45)
    plt.savefig(f'{type(classifier).__name__}.png')
    
    print("---------------------------------------------")

def main() -> None:
    sentences, classifications, vectorizer = get_data()
    X_train, X_test, y_train, y_test = train_test_split(sentences, classifications, test_size=0.2, random_state=42)

    analyze_svm(X_train, X_test, y_train, y_test, vectorizer, SVC(kernel='linear', random_state=42))
    analyze_classifier(X_train, X_test, y_train, y_test, vectorizer, DecisionTreeClassifier())
    analyze_classifier(X_train, X_test, y_train, y_test, vectorizer, RandomForestClassifier())

if __name__ == "__main__":
    main()