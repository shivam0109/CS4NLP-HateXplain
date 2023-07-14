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

def lime_analysis(samples,labels, classifier,name, sample_id=1):
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
    with open(f"output/lime_{name}.html", "w") as file:
        file.write(str(html_object))

def analyze(X_train, X_test, y_train, y_test, vectorizer, classifier: SVC,name) -> None:
    print(f"Current Classifier: {name}")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    lime_analysis(X_test,y_test, classifier,name)

def main() -> None:
    NUM_SAMPLES = 1000
    sentences, classifications, vectorizer = get_data()
    sentences = sentences[:NUM_SAMPLES]
    classifications = classifications[:NUM_SAMPLES]
    X_train, X_test, y_train, y_test = train_test_split(sentences, classifications, test_size=0.2, random_state=42)

    analyze(X_train, X_test, y_train, y_test, vectorizer, make_pipeline(vectorizer,SVC(random_state=42,kernel="sigmoid",C=1,probability=True)),name="SVC")
    analyze(X_train, X_test, y_train, y_test, vectorizer, make_pipeline(vectorizer,DecisionTreeClassifier()),name="Decision_tree")
    analyze(X_train, X_test, y_train, y_test, vectorizer, make_pipeline(vectorizer,RandomForestClassifier()),name= "Random Forest")

if __name__ == "__main__":
    main()