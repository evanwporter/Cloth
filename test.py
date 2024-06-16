import numpy as np
from sloth import DataFrame, Series

# Define numpy arrays for values, index, and columns
values_np = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])
index_np = np.array(['row1', 'row2', 'row3'])
columns_np = np.array(['col1', 'col2', 'col3'])

# Create the DataFrame using numpy arrays
df = DataFrame(values_np, index_np, columns_np)
print("DataFrame with numpy arrays:")
print(df)

# Create the DataFrame using Python lists (list of lists)
values_list = [[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]]
index_list = ['row1', 'row2', 'row3']
columns_list = ['col1', 'col2', 'col3']

df_list = DataFrame(values_list, index_list, columns_list)
print("DataFrame with Python lists:")
print(df_list)

print(dir(slice(1,5)))
print(slice(1,5).indices(1))
d = df.iloc[1:].iloc[0]
# print("D\n", d.to_dataframe())

print(d)

print("df.iloc[0]", df.iloc[0])

# print(d.mask)

print(df.sum(0))

print("df.loc[row1]", df.loc["row1"].index)
print(type(df.loc["row1"]))

print("THE DF", df)

s = Series(np.array([1., 2., 3.]), index_np)
print(s)
print(s.iloc[2])