# Cloth

![sloth](https://github.com/user-attachments/assets/86148566-91bf-4e42-a5a0-3e7cc6b29335)

Cloth is a high performance C++ implementation of the Pandas API.

In exchange for a fraction of the features of Pandas, Cloth has a fraction of the execution time and a fraction of the resources of Pandas.

To obtain such speeds Cloth is implemented using Eigen to hold the underlying data and Nanobinds for the fast C++ -> Python bindings. Note that because the data is stored within Eigen, it must be homogenous. This is akin to the `DataMatrix` in very early versions of Pandas.

Syntax within Cloth are implemented and binded in such a way that is (nearly) identical  to Pandas. Check out tests/test_cloth.py for an example of how to use Cloth in Python.

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
* `cloth.Frame.iloc[cloth.slice]`

# Boolean Indexing
* `cloth.Frame.where(boolean array)`
  * `boolean array -> cloth.Series > value`

# Column Indexing
* `cloth.DataFrame[string]`

# DateTime
* `cloth.datetime(ISO string)`

# Timedelta
* `cloth.timedelta(interval)`



