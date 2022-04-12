#!/usr/bin/env python
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split

# ## Train-Test Split
def get_train_test_data(X, y) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=27,
    )
    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split data from CSV to train and test data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-d', '--data', type=str, required=True, help='data filepath'
    )

    # Get a name for the CSV
    args = parser.parse_args()
    data_filename = args.data

    # Load data file
    df = pd.read_csv(data_filename)
    
    # Split data
    X_train, X_test, y_train, y_test = get_train_test_data(
                                            X=df['text'],
                                            y=df['label'],
    )

    # Save data to files - Note data are Series
    X_train.to_csv(f'X_train.csv', index=False)
    y_train.to_csv(f'y_train.csv', index=False)
    X_test.to_csv(f'X_test.csv', index=False)
    y_test.to_csv(f'y_test.csv', index=False)