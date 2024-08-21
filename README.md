# Cloth

![sloth](https://github.com/user-attachments/assets/86148566-91bf-4e42-a5a0-3e7cc6b29335)

Cloth is a high performance C++ implementation of the Pandas API.

In exchange for a fraction of the feautures of Pandas, Cloth has a fraction of the execution time and a fraction of the resources of Pandas.

To obtain such breakneck speeds Cloth is implemented using Eigen to hold the underlying data and Nanobinds for the fast C++ -> Python bindings.

All commands within Cloth are implemented and binded in such a way that is promises and idenitcal syntax to Pandas.

# Frames
* `cloth.Series`
* `cloth.DataFrame`

# Locational Indexing

* `cloth.Frame.loc[string]`
* `cloth.Frame.loc[slice]`
* `cloth.Frame.loc[cloth.slice]`

# Integer Indexing
* `cloth.Frame.iloc[int]`
* `cloth.Frame.iloc[slice]`
* `cloth.Frame.iloc[boolean array]`
* `cloth.Frame.iloc[cloth.slice]`

# Column Indexing
* `cloth.DataFrame[string]`

# DateTime
* `cloth.datetime(string)`


