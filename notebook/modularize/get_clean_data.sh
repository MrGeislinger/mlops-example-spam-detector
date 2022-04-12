#!/usr/bin/env bash
echo 'Loading data...'
python load_data.py
echo
echo 'Splitting data into train & test X & y sets...'
python train_test_split.py -d data.csv
echo
echo 'Cleaning training and testing data...'
python clean_data.py