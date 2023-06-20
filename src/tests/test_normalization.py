import pandas as pd
from data.normalization import normalize
import numpy as np


def test_normalize():
    # Create a sample dataframe
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 4, 6, 8], "C": [3, 6, 9, 12]})

    # Call the function to perform normalization
    normalized_df = normalize(df, norm="l2")

    # Calculate the L2 norm manually
    norm_values = np.sqrt(df["A"] ** 2 + df["B"] ** 2 + df["C"] ** 2)

    # Divide each column by its respective L2 norm value
    expected_values = df.div(norm_values, axis="index")

    # Check if the normalized values are correct
    assert np.allclose(normalized_df.values, expected_values.values, atol=1e-6)
