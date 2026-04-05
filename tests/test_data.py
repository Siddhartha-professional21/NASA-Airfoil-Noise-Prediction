import pandas as pd
import numpy as np

def test_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
    df = pd.read_csv(url, sep='\t', header=None)
    assert df.shape[0] == 1503
    assert df.shape[1] == 6
    assert df.isnull().sum().sum() == 0
    print('Data check passed - 1503 rows 6 columns no missing values')

if __name__ == '__main__':
    test_data()
