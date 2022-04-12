#!/usr/bin/env python
import argparse
import sys
from typing import Optional
import requests
from zipfile import ZipFile
from io import BytesIO
import pandas as pd

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

# Runs when called from the command line
if __name__ == '__main__':
    # Very simple argument parser
    parser = argparse.ArgumentParser(
        description='Load data to CSV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-o', '--output', type=str, help='output filepath'
    )

    # Get a name for the CSV
    args = parser.parse_args()
    csv_name = args.output
    if csv_name is None:
        csv_name = 'data.csv'

    # Write data to file
    df = load_spam_data()
    df.to_csv(csv_name, index=False)
    # Print to screen the name of the data file
    sys.stdout.write(csv_name)
    sys.exit(0)