import pandas as pd

from data.data_split import data_split


def test_data_split():
    # Create a sample DataFrame for testing
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [6, 7, 8, 9, 10],
        "label": ["A", "B", "A", "B", "A"],
    }
    df = pd.DataFrame(data)

    # Call the data_split function
    x_train, x_test, y_train, y_test = data_split(df, "label")

    # Check if the returned objects are DataFrames
    assert isinstance(x_train, pd.DataFrame), "x_train is not a DataFrame"
    assert isinstance(x_test, pd.DataFrame), "x_test is not a DataFrame"
    assert isinstance(y_train, pd.Series), "y_train is not a Series"
    assert isinstance(y_test, pd.Series), "y_test is not a Series"

    # Check if the sizes of the splits are correct
    assert len(x_train) + len(x_test) == len(
        df
    ), "Sum of train and test sizes does not match original size"
    assert len(x_train) == len(y_train), "Size of x_train does not match y_train"
    assert len(x_test) == len(y_test), "Size of x_test does not match y_test"
