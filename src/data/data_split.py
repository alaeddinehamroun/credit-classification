from sklearn.model_selection import train_test_split


def data_split(df, label, split_size=0.2):
    """
    Split a dataframe into train, validation, and test sets.

    Args:
        df: DataFrame to split
        label: Column name of the label column
    Return:
        train, validation, test DataFrames
    """

    y = df[label]
    X = df.drop(columns=[label])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split_size)

    return x_train, x_test, y_train, y_test
