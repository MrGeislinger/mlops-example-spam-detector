#!/usr/bin/env python
from typing import Optional
import argparse
import pandas as pd
import numpy as np

# ## Clean Data
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean both train & test data (X &y)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Assume data are X_train, y_train, X_test, y_test(current directory)
    # Note assuming Series here
    X_train = pd.read_csv('X_train.csv', header=None).iloc[:,0]
    y_train = pd.read_csv('y_train.csv', header=None).iloc[:,0]
    X_test = pd.read_csv('X_test.csv', header=None).iloc[:,0]
    y_test = pd.read_csv('y_test.csv', header=None).iloc[:,0]

    # Clean
    X_train_clean = get_clean_data(X_train)
    X_test_clean = get_clean_data(X_test)
    # Change ham → 0 & spam → 1
    y_train_clean = get_clean_labels(y_train, ham='ham')
    y_test_clean = get_clean_labels(y_test, ham='ham')
    
    # Save data to files - Note data are Series saved with numpy
    np.save(f'X_train', X_train_clean)
    np.save(f'y_train', y_train_clean)
    np.save(f'X_test', X_test_clean)
    np.save(f'y_test', y_test_clean)