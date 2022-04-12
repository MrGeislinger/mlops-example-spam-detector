#!/usr/bin/env python
from typing import Optional
import requests
from zipfile import ZipFile
from io import BytesIO

import pandas as pd
import numpy as np

from joblib import dump, load
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, confusion_matrix,
)


# # Load Data
def load_data(
    url: str,
    filename: str,
    zip_filename: Optional[str] = None,
    **kwargs,
 ) -> pd.DataFrame:
    '''Download zip file with CSV from URL and returns a DataFrame.
    '''
    if zip_filename is None:
        zip_filename = url.split('/')[-1]

    r = requests.get(url)
    files = ZipFile(BytesIO(r.content))
    # No header
    return pd.read_csv(files.open(filename), **kwargs)

def load_spam_data() -> pd.DataFrame:
    '''Returns DataFrame for SMS spam detection (https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
    '''
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    filename = 'SMSSpamCollection'
    df = load_data(url, filename, sep='\t', names=['label', 'text'])
    return df

df = load_spam_data()
print(df.head())
print(df['label'].value_counts())


# ## Train-Test Split


def get_train_test_data(X, y) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=27,
    )
    return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = get_train_test_data(
                                        X=df['text'],
                                        y=df['label'],
)


# Check that this looks good (train & test should look "similar")
print(f'Train (n={y_train.shape[0]:,}):')
for label, count in zip(*np.unique(y_train, return_counts=True)):
    print(f'{label}\t{count/y_train.shape[0]:.2%}') # Normalize Counts

print()
print(f'Test (n={y_test.shape[0]:,}):')
for label, count in zip(*np.unique(y_test, return_counts=True)):
    print(f'{label}\t{count/y_test.shape[0]:.2%}') # Normalize Counts


# # Clean Data


def get_clean_data(data: pd.Series) -> np.ndarray:
    '''Lower text data
    '''
    # TODO: Actually clean the data with some process
    return data.str.lower().values # Turn it into a numpy array

def get_clean_labels(
    y: pd.Series,
    ham: Optional[str] ='ham',
    spam: Optional[str] ='spam',
) -> np.ndarray:
    '''Convert strings of 'ham' and 'spam' to 0 & 1 (respectively)
    '''
    assert (ham is not None) or (spam is not None), 'Define either ham or spam'
    
    if ham:
        y_clean = np.fromiter(
            iter=(0 if y_ == ham else 1 for y_ in y), 
            dtype=int,
        )
    elif spam:
        y_clean = np.fromiter(
            iter=(1 if y_ == spam else 0 for y_ in y), 
            dtype=int,
        )
    return y_clean

X_train_clean = get_clean_data(X_train)
X_test_clean = get_clean_data(X_test)


# Change ham → 0 & spam → 1
y_train = get_clean_labels(y_train, ham='ham')
y_test = get_clean_labels(y_test, ham='ham')


# # Simple Model

# A _very simple model_ is checking if "spammy" words are being used and mark it as spam
# > NOTE: We could have a more "realistic" simple model by looking at words that _only_ occur in training spam (or conversely words that only occur in training ham)


class SimpleModel(BaseEstimator, ClassifierMixin):
    ''' Very simple model that checks provided `spam_words` to label as "spam".
    Assumes that first class is "ham" and next class is "spam".
    
    Inspired by https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    '''
    
    def __init__(self, spam_words=('free')):
        self.spam_words = spam_words

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Spam if text contains any of the spam words
        preds = [
            1 if any(word in feat for word in self.spam_words)
            else 0
                for feat in X
        ]
        return preds
    


common_spam_words = (
    'free',
    'won',
    'win', # variation of "win"
    'money',
)

model = SimpleModel(spam_words=common_spam_words)


model.fit(X_train_clean.reshape(-1,1), y_train)
y_pred = model.predict(X_train_clean)


# # Evaluate

# ## Using Just Training Data

print(f'''Evaluation using Training Data:
    {accuracy_score(y_train, y_pred)=:.5f}
    {recall_score(y_train, y_pred)=:.5f}
    {precision_score(y_train, y_pred)=:.5f}
''')


# Percent we got correct for each class
print(confusion_matrix(y_train, y_pred, normalize='true'))


# ## Validation (w/ Cross-Validation)


scores = cross_val_score(model, X_train_clean, y_train, cv=3)
scores


# ## Evaluate with Test


y_pred = model.predict(X_test_clean)


print(f'''Evaluation using Test Data:
    {accuracy_score(y_test, y_pred)=:.5f}
    {recall_score(y_test, y_pred)=:.5f}
    {precision_score(y_test, y_pred)=:.5f}
''')

print(confusion_matrix(y_test, y_pred, normalize='true'))


# # Export Model

dump(model, 'simple_model_v0.joblib')

