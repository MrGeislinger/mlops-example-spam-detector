#!/usr/bin/env python
import argparse
import numpy as np
from joblib import load
from SimpleModel import SimpleModel
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, confusion_matrix,
)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='evaluate given model ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-m', '--model', type=str, help='load model',
    )
    parser.add_argument(
        '-X', type=str, required=True, help='X test data',
    )
    parser.add_argument(
        '-y', type=str, required=True, help='y test data',
    )

    args = parser.parse_args()
    
    # Load the data (assume file names)
    X_test = np.load(args.X, allow_pickle=True)
    y_test = np.load(args.y, allow_pickle=True)
    model = load(args.model)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    # Evaluate
    print(f'''Evaluation using Test Data:
        {accuracy_score(y_test, y_pred)=:.5f}
        {recall_score(y_test, y_pred)=:.5f}
        {precision_score(y_test, y_pred)=:.5f}
    ''')

    print(confusion_matrix(y_test, y_pred, normalize='true'))