import pandas as pd

# Sample data
data = {
    'Platform': ['LinkedIn', 'Canva'],
    'Value 0': [3958629.0, 73021.0],
    'Value 1': [14259464.0, 262482.0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set the 'Platform' column as index (if you want to keep it)
df.set_index('Platform', inplace=True)

# Divide each row by the sum of the row
df_normalized = df.div(df.sum(axis=1), axis=0)

print("Original DataFrame:")
print(df)
print("\nNormalized DataFrame:")
print(df_normalized)