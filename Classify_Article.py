import pickle
import os
import sklearn
import scipy
from Data_Processing_Alg_Training import *
import sys
import logging


def predict():
    '''Asks user for input and classifies their inputted text as real or fake'''

    input_text = input('Please paste the article you would like classified: ')

    print('\nLoading...')

    ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '.'))

    pa = open(os.path.join(ROOT_DIR, 'Classifiers\PA_Classifier.sav'), 'rb')
    lg = open(os.path.join(ROOT_DIR, 'Classifiers\LogReg_Classifier.sav'), 'rb')
    rf = open(os.path.join(ROOT_DIR, 'Classifiers\RF_Classifier.sav'), 'rb')

    pa_model = pickle.load(pa)
    logReg_model = pickle.load(lg)
    rf_model = pickle.load(rf)

    pa.close()
    lg.close()
    rf.close()


    input_LR = prepare_input(input_text, 'LR')
    input_PA = prepare_input(input_text, 'PA')
    input_RF = prepare_input(input_text, 'RF')

    LR_pred = logReg_model.predict(input_LR)[0]
    PA_pred = pa_model.predict(input_PA)[0]
    RF_pred = rf_model.predict(input_RF)[0]

    LR_pred = 'Real' if LR_pred == 1 else 'Fake'
    PA_pred = 'Real' if PA_pred == 1 else 'Fake'
    RF_pred = 'Real' if RF_pred == 1 else 'Fake'

    print('\nThe Logistic Regression algorithm classifies the article as: ' + LR_pred)
    print('The Passive Aggressive algorithm classifies the article as: ' + PA_pred)
    print('The Random Forest algorithm classifies the article as: ' + RF_pred + '\n')

def main():

    logger = logging.getLogger('nltk')
    logger.setLevel(logging.ERROR)

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    print('\n')

    ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '.'))

    pa = open(os.path.join(ROOT_DIR, 'Classifiers\PA_Classifier.sav'), 'rb')
    lg = open(os.path.join(ROOT_DIR, 'Classifiers\LogReg_Classifier.sav'), 'rb')
    rf = open(os.path.join(ROOT_DIR, 'Classifiers\RF_Classifier.sav'), 'rb')

    pa_model = pickle.load(pa)
    logReg_model = pickle.load(lg)
    rf_model = pickle.load(rf)

    pa.close()
    lg.close()
    rf.close()

    predict()

main()



