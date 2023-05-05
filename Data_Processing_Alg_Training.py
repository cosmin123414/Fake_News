import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def format_data(data):
    '''Load and format data'''

    real_news['label'] = 1
    fake_news['label'] = 0

    #Join title and text into a single column
    data['text'] = data['title'] + data['text']

    #Keep only the text and label columns
    data = data[['text','label']]
    return data


def clean_text(text):
    '''More formatting'''

    #convert to lowercase
    text = text.lower()

    #remove punctuation
    text = ''.join(ch for ch in text if ch not in string.punctuation)

    #remove stopwords
    stop = stopwords.words('english')
    text = ' '.join(word for word in text.split() if word not in stop)

    #remove non-ASCII characters
    text = ''.join(ch for ch in text if ord(ch)<128)

    #remove apostrophes
    text = text.replace("'", "")

    #remove single characters
    ' '.join(word for word in text.split() if len(word) > 1 )

    return text

def lemmatize(tokens):
    '''lemmatization'''

    lemmatizer = nltk.stem.WordNetLemmatizer()

    return [lemmatizer.lemmatize(token) for token in tokens]

def identity_function(text):
     return text

def prepare_input(text, algorithm):
    '''(string, string) -> array
    Processes and vectorizes text with the vectorizer that corresponds to the inputted algorithm
    Output can be fed directly into the classifiers'''

    ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '.'))

    if algorithm == 'LR':
        vec = open(os.path.join(ROOT_DIR, 'Vectorizers\VectorizerLR.sav'), 'rb')
        vectorizer = pickle.load(vec)
        vec.close()

    elif algorithm == 'PA':
        vec = open(os.path.join(ROOT_DIR, 'Vectorizers\VectorizerPA.sav'), 'rb')
        vectorizer = pickle.load(vec)
        vec.close()

    else:
        vec = open(os.path.join(ROOT_DIR, 'Vectorizers\VectorizerRF.sav'), 'rb')
        vectorizer = pickle.load(vec)
        vec.close()

    text = clean_text(text)
    text = nltk.word_tokenize(text)
    text = lemmatize(text)
    text = [text]
    text = pd.Series(text)
    text = vectorizer.transform(text)
    return text

if __name__ == '__main__':


    real_news = pd.read_csv('C:/Users/cosmi/PycharmProjects/Fake_News/Data/True.csv')
    fake_news = pd.read_csv('C:/Users/cosmi/PycharmProjects/Fake_News/Data/Fake.csv')

    #Format Data
    real_news = format_data(real_news)
    fake_news = format_data(fake_news)

    #Merge both datasets into a single dataset
    news_data = pd.concat([real_news, fake_news])

    #Fill na
    news_data['text'] = news_data['text'].fillna('')

    #More formatting and cleaning
    news_data['text'] = news_data['text'].apply(clean_text)

    #Tokenize text
    news_data['text'] = news_data['text'].apply(nltk.word_tokenize)

    #Lemmatize text
    news_data['text'] = news_data['text'].apply(lemmatize)

    #Split data into train/test
    x_train, x_test, y_train, y_test = train_test_split(news_data['text'], news_data['label'], test_size=0.3,
                                                        stratify=news_data['label'])

    #Train vectorizer
    vectorizer = TfidfVectorizer(tokenizer = identity_function, lowercase = False, ngram_range = (1,3))

    x_train_tfidf = vectorizer.fit_transform(x_train) #Only fit vectorizer to train data
    x_test_tfidf = vectorizer.transform(x_test)

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    #Train logistic regression model and save vectorizer
    logReg_model = LogisticRegression()
    logReg_model.fit(x_train_tfidf, y_train)

    vecfile = open('C:/Users/cosmi/PycharmProjects/Fake_News/Vectorizers/VectorizerLR.sav', 'wb')
    pickle.dump(vectorizer, vecfile)
    vecfile.close()

    #Train passive aggressive model and save vectorizer
    pa_model = PassiveAggressiveClassifier()
    pa_model.fit(x_train_tfidf, y_train)

    vecfile = open('C:/Users/cosmi/PycharmProjects/Fake_News/Vectorizers/VectorizerPA.sav', 'wb')
    pickle.dump(vectorizer, vecfile)
    vecfile.close()

    #Train random forest model and save vectorizer
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train_tfidf, y_train)

    vecfile = open('C:/Users/cosmi/PycharmProjects/Fake_News/Vectorizers/VectorizerRF.sav', 'wb')
    pickle.dump(vectorizer, vecfile)
    vecfile.close()

    #Saving Models
    lg = open('C:/Users/cosmi/PycharmProjects/Fake_News/Classifiers/LogReg_Classifier.sav', 'wb')
    pa = open('C:/Users/cosmi/PycharmProjects/Fake_News/Classifiers/PA_Classifier.sav', 'wb')
    rf = open('C:/Users/cosmi/PycharmProjects/Fake_News/Classifiers/RF_Classifier.sav', 'wb')

    pickle.dump(logReg_model, lg)
    pickle.dump(pa_model, pa)
    pickle.dump(rf_model, rf)

    lg.close()
    pa.close()
    rf.close()