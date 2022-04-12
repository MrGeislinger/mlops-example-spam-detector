#!/usr/bin/env python
import argparse
import numpy as np
from joblib import dump

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# # Simple Model
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create, train, and save simple model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-X', type=str, required=True, help='X train data'
    )
    parser.add_argument(
        '-y', type=str, required=True, help='y train data'
    )
    parser.add_argument(
        '-o', '--output', type=str, help='output filepathfor saved model',
    )

    args = parser.parse_args()
    
    # Load the data (assume file names)
    X_train = np.load(args.X, allow_pickle=True)
    y_train = np.load(args.y, allow_pickle=True)
    model_output_fname = args.output if args.output else 'model.joblib'

    # Create, train, save model
    model = SimpleModel(spam_words=common_spam_words)
    model.fit(X_train, y_train)
    dump(model, model_output_fname)