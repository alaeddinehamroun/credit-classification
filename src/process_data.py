import pandas as pd

from data.handling_missing_values import drop_rows_with_missing_values


# Load in the data
df = pd.read_csv("data/data.csv")

# Drop the first column
df.drop("Unnamed: 0", axis=1, inplace=True)

# Handle missing values
df = drop_rows_with_missing_values(df)

# Convert non-numeric columns to numeric


# save the processed data
df.to_csv("data/processed_data.csv", index=False)
