from sklearn import preprocessing


def standardize(df):
    """
    Perform standardization on the dataset
    Args:
      df: dataframe
    Returns:
      Normalized dataframe
    """
    scaler = preprocessing.StandardScaler().fit(df)
    return scaler.transform(df)
