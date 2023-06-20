from sklearn import preprocessing
import pandas as pd


def normalize(df, norm="l2"):
    """
    Normalize the dataset: scale individual samples to have unit norm
    Args:
      df: dataframe
      norm: the norm to use to normalize, ('l1', 'l2', 'max')
    Returns:
      normalized dataframe
    """

    normalizer = preprocessing.Normalizer(norm=norm).fit(df)
    return pd.DataFrame(normalizer.transform(df))
