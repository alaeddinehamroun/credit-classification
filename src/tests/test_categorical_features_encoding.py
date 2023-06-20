from data.categorical_features_encoding import ordinal_encoding, one_hot_encoding
import pandas as pd
import numpy as np


def test_ordinal_encoding():
    # Create a sample dataframe with categorical columns
    df = pd.DataFrame(
        {
            "Color": ["Red", "Green", "Blue", "Red"],
            "Size": ["Small", "Medium", "Large", "Small"],
        }
    )

    # Call the function to perform ordinal encoding
    encoded_df = ordinal_encoding(df, ["Color", "Size"])

    # Check if the categorical columns are encoded correctly
    expected_values = np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
    assert np.array_equal(encoded_df[["Color", "Size"]].values, expected_values)


def test_one_hot_encoding():
    # Create a sample dataframe with categorical columns
    df = pd.DataFrame(
        {
            "Color": ["Red", "Green", "Blue", "Red"],
            "Size": ["Small", "Medium", "Large", "Small"],
        }
    )

    # Call the function to perform one-hot encoding
    encoded_df = one_hot_encoding(df, ["Color", "Size"])

    print(encoded_df)
    # Check if the categorical columns are one-hot encoded correctly
    expected_values = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.array_equal(
        encoded_df,
        expected_values,
    )
