import pandas as pd

df = pd.read_csv("/Users/ahmedmostafa/Downloads/linear_regression_from_scratch/diamonds.csv")
print(df.head())
print(df.shape)
df_subset = df[["x", "y", "z"]]
x = df.drop(columns="price")
y = df["price"]
print(f'df_subset: {df_subset.shape}')
print(f'x :{x.shape}')
print(f'y: {y.shape}')




