import pandas as pd
import numpy as np

from data.handling_missing_values import (
    drop_cols_with_missing_values,
    drop_rows_with_missing_values,
    fill_missing_with_group_mode,
    mark_imputed_values,
    simple_multivariate_imputation,
    simple_univariate_imputation,
)


def test_drop_cols_with_missing_values():
    # Create a test DataFrame with missing values
    data = {"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8], "C": [9, 10, 11, 12]}
    df = pd.DataFrame(data)

    # Perform the function to drop columns with missing values
    df_cleaned = drop_cols_with_missing_values(df)

    # Check if the columns with missing values are dropped
    expected_columns = ["C"]
    assert list(df_cleaned.columns) == expected_columns

    # Check if the original DataFrame is not modified
    assert list(df.columns) == ["A", "B", "C"]

    # Create a test DataFrame with no missing values
    data_no_missing = {
        "D": [13, 14, 15, 16],
        "E": [17, 18, 19, 20],
        "F": [21, 22, 23, 24],
    }
    df_no_missing = pd.DataFrame(data_no_missing)

    # Perform the function on the DataFrame with no missing values
    df_cleaned_no_missing = drop_cols_with_missing_values(df_no_missing)

    # Check if the cleaned DataFrame remains the same
    assert df_cleaned_no_missing.equals(df_no_missing)

    # Check if the original DataFrame is not modified
    assert list(df_no_missing.columns) == ["D", "E", "F"]


def test_drop_rows_with_missing_values():
    # Create a test DataFrame with missing values
    data = {"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8], "C": [9, 10, np.nan, 12]}
    df = pd.DataFrame(data)

    # Perform the function to drop rows with missing values
    df_cleaned = drop_rows_with_missing_values(df)

    # Check if the rows with missing values are dropped
    expected_rows = 2
    assert len(df_cleaned) == expected_rows

    # Check if the original DataFrame is not modified
    assert len(df) == 4


def test_mark_imputed_values():
    # Create a test DataFrame with missing values
    data = {"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8], "C": [9, 10, 11, np.nan]}
    df = pd.DataFrame(data)

    # Perform the function to mark imputed values
    df_marked = mark_imputed_values(df)

    # Check if new columns indicating missing values are added
    expected_columns = ["A_was_missing", "B_was_missing", "C_was_missing"]
    assert list(df_marked.columns)[-3:] == expected_columns

    # Check if the values are marked correctly
    expected_values_A = [False, False, True, False]
    assert list(df_marked["A_was_missing"]) == expected_values_A

    expected_values_B = [False, True, False, False]
    assert list(df_marked["B_was_missing"]) == expected_values_B

    expected_values_C = [False, False, False, True]
    assert list(df_marked["C_was_missing"]) == expected_values_C

    # Check if the column names
    assert list(df.columns) == [
        "A",
        "B",
        "C",
        "A_was_missing",
        "B_was_missing",
        "C_was_missing",
    ]


def test_fill_missing_with_group_mode():
    # Create a sample dataframe with missing values
    df = pd.DataFrame(
        {
            "Group": ["A", "A", "B", "B", "C", "C"],
            "Value": [1, np.nan, 3, np.nan, np.nan, 6],
        }
    )

    # Call the function to fill missing values with group mode
    df_filled = fill_missing_with_group_mode(df, "Group", "Value")

    # Check if missing values are filled correctly
    expected_values = [1, 1, 3, 3, 6, 6]
    assert np.array_equal(df_filled["Value"], expected_values)

    # Create a sample dataframe with non-numerical values in the column to be filled
    df_non_numerical = pd.DataFrame(
        {
            "Group": ["A", "A", "B", "B", "C", "C"],
            "Value": ["One", np.nan, "Three", np.nan, np.nan, "Six"],
        }
    )


def test_simple_univariate_imputation():
    # Create a sample dataframe with missing values

    df = pd.DataFrame(
        {"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8], "C": [9, 10, 11, np.nan]}
    )

    # Call the function to perform simple univariate imputation
    imputed_df = simple_univariate_imputation(df)
    print(imputed_df)
    # Check if missing values are filled correctly
    expected_values = [
        [1.0, 5.0, 9.0],
        [2.0, 5.0, 10.0],
        [1.0, 7.0, 11.0],
        [4.0, 8.0, 9.0],
    ]
    assert np.array_equal(imputed_df.values, expected_values)


def test_simple_multivariate_imputation():
    # Create a sample dataframe with missing values
    df = pd.DataFrame(
        {"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8], "C": [9, 10, 11, np.nan]}
    )

    # Call the function to perform simple multivariate imputation
    imputed_df = simple_multivariate_imputation(df)

    # Check if missing values are filled correctly
    expected_values = [
        [1.0, 5.0, 9.0],
        [2.0, 5.0, 10.0],
        [5.0, 7.0, 11.0],
        [4.0, 8.0, 9.0],
    ]
    print(np.round(expected_values))
    print(np.round(imputed_df.values))
    assert np.array_equal(np.round(imputed_df.values), np.round(expected_values))
