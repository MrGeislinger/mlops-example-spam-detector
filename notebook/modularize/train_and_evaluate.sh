#!/usr/bin/env bash
echo 'Training simple model...'
python train_simple_model.py -X X_train.npy -y y_train.npy -o model.joblib

echo
echo 'Evaluating model using the training data...'
python evaluate_model.py -X X_train.npy -y y_train.npy -m model.joblib

echo
echo 'Evaluating model using the test data...'
python evaluate_model.py -X X_test.npy -y y_test.npy -m model.joblib