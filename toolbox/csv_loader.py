import pandas as pd


def load_data(csvfile):
    df = pd.read_csv(
        csvfile,
        names=['x', 'y', 'bits'],
        header=None,
        dtype={'x': float, 'y': float, 'bits': int}
    )
    return {
        "x": df['x'].values,
        "y": df['y'].values,
        "bits": df['bits'].values
    }
