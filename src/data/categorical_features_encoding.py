from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def ordinal_encoding(df, cols):
    """
    Ordinal encode cols with categorical data: assign each unique value to a different integer
    Args:
        df: dataframe
        cols: columns to be ordinal encoded
    Returns:
        ordinal encoded dataframe
    """

    # Apply ordinal encoder to each column with categorical data
    ordinal_encoder = OrdinalEncoder()
    df[cols] = ordinal_encoder.fit_transform(df[cols])

    return df


def one_hot_encoding(df, cols):
    """
    One hot encode cols with categorical data: transform each categorical feature into binary features
    Args:
        df: dataframe
        cols: columns to be encoded
    Returns:
        on hot encoded dataframe
    """

    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(df[cols]))
    # One-hot encoding removed index; put it back
    OH_cols.index = df.index
    # Remove categorical columns (will replace with one-hot encoding)
    num_df = df.drop(cols, axis=1)
    # Add one-hot encoded columns to numerical features
    OH_df = pd.concat([num_df, OH_cols], axis=1)
    # Ensure all columns have string type
    OH_df.columns = OH_df.columns.astype(str)

    return OH_df
