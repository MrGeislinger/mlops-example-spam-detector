from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

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