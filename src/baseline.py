from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from gensim import downloader

from transformers import AutoTokenizer, AutoModelForSequenceClassification


STOP_WORDS = stopwords.words('english')
SCORE = {0: 2, 1: 0, 2:1}

def preprocess_post_tokens(text):
    sentences = [word for word in text if word not in STOP_WORDS]
    # return sentences
    return " ".join(sentences)

def preprocess_annotators(text):
    scores = [SCORE[score] for score in text['label']]
    return sum(scores) / (len(scores) * 2) 

def tf_idf_embedding(train_x, test_x, validation_x):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_x + test_x + validation_x)

    train_X = vectorizer.transform(train_x)
    test_X = vectorizer.transform(test_x)
    validation_X = vectorizer.transform(validation_x)
    
    return train_X, test_X, validation_X, vectorizer

def support_vector_regression(train_X, train_Y):
    clf = SVR(kernel='linear', C=1.0, epsilon=0.2)
    clf.fit(train_X, train_Y)
    return clf

def decision_tree_regression(train_X, train_Y):
    clf = DecisionTreeRegressor(max_depth=25)
    clf.fit(train_X, train_Y)
    return clf

def knn_regression(train_X, train_Y):
    clf = KNeighborsRegressor(n_neighbors=25)
    clf.fit(train_X, train_Y)
    return clf

def eval(score) -> str:
    if (score < 0.333):
        return "Normal"
    elif (score < 0.666):
        return "Offensive"
    else:
        return "Hate speech"

def main():
    hatexplain = load_dataset('hatexplain')
    hatexplain = hatexplain.map(lambda x: {
        'post_tokens': preprocess_post_tokens(x['post_tokens']),
        'annotators': preprocess_annotators(x['annotators'])
        })
            
    train_sentences = hatexplain['train']['post_tokens']
    train_labels = hatexplain['train']['annotators']

    test_sentences = hatexplain['test']['post_tokens']
    test_labels = hatexplain['test']['annotators']

    validation_sentences = hatexplain['validation']['post_tokens']
    validation_labels = hatexplain['validation']['annotators']

    train_X, test_X, validation_X, vectorizer = tf_idf_embedding(train_sentences, test_sentences, validation_sentences)

    svr = support_vector_regression(train_X, train_labels)
    pred_y = svr.predict(test_X)
    mse = mean_squared_error(test_labels, pred_y)
    print(f"SVR - Mean squared error: {mse}")

    dtr = decision_tree_regression(train_X, train_labels)
    pred_y = dtr.predict(test_X)
    mse = mean_squared_error(test_labels, pred_y)
    print(f"DTR - Mean squared error: {mse}")

    knnR = knn_regression(train_X, train_labels)
    pred_y = knnR.predict(test_X)
    mse = mean_squared_error(test_labels, pred_y)
    print(f"knnR - Mean squared error: {mse}")
    print("----------------------------------------------------------")
    while(True):
        text = input("Enter text (or type 'q' to quit): ")
        if text == 'q':
            break

        embedding = vectorizer.transform([text])
        print("Support Vector Regression: ", eval(svr.predict(embedding)))
        print("Decision Tree Regression: ", eval(dtr.predict(embedding)))
        print("K-Nearest Neighbor Regression: ", eval(knnR.predict(embedding)))
        print("----------------------------------------------------------")

if __name__ == "__main__":
    main()