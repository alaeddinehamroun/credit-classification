from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer

# IterativeImputer is experimental and the API might change without any deprecation cycle.
# To use it, you need to explicitly import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd


def drop_cols_with_missing_values(df):
    """
    Drop columns with missing values from df

    Args:
      df: dataframe
    Returns:
      df without missing values
    """
    # Get names of columns with missing values
    cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
    # Drop columns
    return df.drop(cols_with_missing, axis=1)


def drop_rows_with_missing_values(df):
    """
    Drop instances with missing values from df

    Args:
      df: dataframe
    Returns:
      df without missing values
    """
    return df.dropna()


def mark_imputed_values(df):
    """
    Indicate the presence of missing values in the dataset

    Args:
      df: dataframe

    Returns:
      new dataframe with new columns indicating if a value was missing
    """
    indicator = MissingIndicator()
    missing_values = indicator.fit_transform(df)
    cols_with_missing = df.columns[indicator.features_]
    # Make new columns indicating what will be imputed
    for col in cols_with_missing:
        df[col + "_was_missing"] = missing_values[:, cols_with_missing == col]

    return df


def fill_missing_with_group_mode(df, groupby, column):
    """
    Fill missing values of a column with group mode

    Args:
        df: dataframe
        groupby: the column to be grouped by
        column: the column with missing values, values must be numerical

    Returns:
        df without missing values

    Raises:
        ValueError: If not numerical values are group
    """
    mode_per_group = df.groupby(groupby)[column].transform(lambda x: x.mode().iat[0])
    if df[column].dtype != mode_per_group.dtype:
        raise ValueError(f"Column '{column}' has non-numerical values.")

    df[column] = df[column].fillna(mode_per_group)

    return df


def simple_univariate_imputation(df, strategy="most_frequent"):
    """
    Fill missing values using only non-missing values in that dimension

    Args:
        df: dataframe
    Returns:
        Imputed dataframe
    """

    my_imputer = SimpleImputer(strategy=strategy)
    imputed_df = pd.DataFrame(my_imputer.fit_transform(df))
    # Put back column names
    imputed_df.columns = df.columns

    return imputed_df


def simple_multivariate_imputation(
    df, max_iter=10, random_state=0, init_strategy="most_frequent"
):
    """
    Fill missing values using the entire set of available feature dimensions

    Args:
        df: dataframe
        max_iter: Maximum number of imputation rounds to perform, default is 10
        random_state: The seed of the pseudo random number generator to use, default is 0
        init_strategy: the initial strategy for imputation (mean, median, most_frequent, constant), for mean|median, numerical values are needed
    Returns:
        Imputed dataframe
    """

    my_imputer = IterativeImputer(
        max_iter=max_iter, random_state=random_state, initial_strategy=init_strategy
    )

    imputed_df = pd.DataFrame(my_imputer.fit_transform(df))

    return imputed_df
