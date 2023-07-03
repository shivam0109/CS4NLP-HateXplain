import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from IPython.display import display, HTML

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from lime import lime_text
from lime.lime_text import LimeTextExplainer

from datasets import load_dataset, concatenate_datasets

SCORE = {0: 1, 1: 0, 2: 0.5} # 0: Hate Speech, 1: Normal, 2: Offensive
STOP_WORDS = stopwords.words('english') + ["<user>"]
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
        #vectorizer = CountVectorizer()
        vectorizer = TfidfVectorizer()
        vectorizer.fit(concatenated_dataset["sentence"])
        sentences = np.array(concatenated_dataset["sentence"])
        classifications = np.array(concatenated_dataset["classification"])

        percentage_of_zeros = np.count_nonzero(classifications == 0) / len(classifications)
        print("Percentage of zeros:", percentage_of_zeros)

        return sentences, classifications, vectorizer

    else:
        sentences = list(concatenated_dataset["sentence"])
        classifications = np.array(concatenated_dataset["classification"])

        return sentences, classifications

def lime_analysis(samples,labels, classifier, sample_id=0):
    explainer = LimeTextExplainer(class_names=["Normal", "Hate Speech"])
    exp = explainer.explain_instance(samples[sample_id], classifier.predict_proba, num_features=6, labels=[0, 1])
    print(f"Sentence: {samples[sample_id]}")
    print(f"Predicted class: {classifier.predict([samples[sample_id]])[0]}")
    print(f"True class: {labels[sample_id]}")
    print ('Explanation for class Normal')
    print ('\n'.join(map(str, exp.as_list(label=0))))
    print ()
    print ('Explanation for class Hateful')
    print ('\n'.join(map(str, exp.as_list(label=1))))
    html_object = exp.as_html(labels=[0,1],predict_proba=True,show_predicted_value=True)
    with open("output/lime.html", "w") as file:
        file.write(str(html_object))



def analyze_svm(X_train, X_test, y_train, y_test, vectorizer, classifier: SVC) -> None:
    print("Current Classifier: SVC")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    lime_analysis(X_test,y_test, classifier)

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
    NUM_SAMPLES = 1000
    sentences, classifications, vectorizer = get_data()
    sentences = sentences[:NUM_SAMPLES]
    classifications = classifications[:NUM_SAMPLES]
    X_train, X_test, y_train, y_test = train_test_split(sentences, classifications, test_size=0.2, random_state=42)

    analyze_svm(X_train, X_test, y_train, y_test, vectorizer, make_pipeline(vectorizer,SVC(random_state=42,kernel="sigmoid",C=1,probability=True)))
    #analyze_classifier(X_train, X_test, y_train, y_test, vectorizer, DecisionTreeClassifier())
    #analyze_classifier(X_train, X_test, y_train, y_test, vectorizer, RandomForestClassifier())

if __name__ == "__main__":
    main()