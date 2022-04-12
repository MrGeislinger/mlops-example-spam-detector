#!/usr/bin/env python
import argparse
import numpy as np
from joblib import dump

from SimpleModel import SimpleModel

# # Simple Model


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