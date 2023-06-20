import pandas as pd
import numpy as np
from data.standardization import standardize


def test_standardize():
    # Create a sample dataframe
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": [100, 200, 300, 400, 500],
    }
    df = pd.DataFrame(data)

    # Call the standardize function
    standardized_df = standardize(df)

    # Calculate mean and standard deviation of each column in the standardized dataframe
    mean = np.mean(standardized_df, axis=0)
    std = np.std(standardized_df, axis=0)

    # Check if mean is close to 0 and standard deviation is close to 1 for each column
    for col in range(df.shape[1]):
        assert np.isclose(
            mean[col], 0, atol=1e-2
        ), f"Mean of column {df.columns[col]} is not close to 0."
        assert np.isclose(
            std[col], 1, atol=1e-2
        ), f"Standard deviation of column {df.columns[col]} is not close to 1."
