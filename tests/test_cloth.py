# tests/test_cloth.py

import pytest
import numpy as np
import cloth  # Import your module here

def test_slice():
    sl = cloth.slice(0, 10, 2)
    assert sl.length == 5
    assert sl.start == 0
    assert sl.stop == 10
    assert sl.step == 2

def test_slice_negative_indices():
    sl = cloth.slice(-10, -2, 2)
    sl.normalize(10)
    assert sl.length == 4
    assert sl.start == 0
    assert sl.stop == 8
    assert sl.step == 2

def test_slice_zero_step():
    with pytest.raises(ValueError):
        cloth.slice(0, 10, 0)

# def test_combine_slices():
#     sl1 = cloth.slice(0, 10, 2)
#     sl2 = cloth.slice(1, 4, 1)
#     combined = cloth.combine_slices(sl1, sl2)
#     assert combined.start == 2
#     assert combined.stop == 8
#     assert combined.step == 2

# def test_combine_slice_with_index():
#     sl1 = cloth.slice(0, 10, 2)
#     index = 3
#     combined_index = cloth.combine_slice_with_index(sl1, index)
#     assert combined_index == 6

def test_object_index():
    keys = ["a", "b", "c", "d"]
    obj_index = cloth.ObjectIndex(keys)
    assert obj_index.keys() == keys
    assert obj_index["a"] == 0
    assert obj_index["d"] == 3
    # assert str(obj_index) == "[a, b, c, d]"

def test_series():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    keys = ["a", "b", "c", "d"]
    series = cloth.Series(values, keys)
    assert series.length() == 4
    assert series.sum() == 10.0
    assert series.mean() == 2.5
    assert series.min() == 1.0
    assert series.max() == 4.0
    assert series.loc["a"] == 1.0
    assert series.iloc[1] == 2.0

def test_series_repr():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    keys = ["a", "b", "c", "d"]
    series = cloth.Series(values, keys)
    assert isinstance(str(series), str)

def test_series_mask():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    keys = ["a", "b", "c", "d"]
    series = cloth.Series(values, keys)
    mask = series.mask
    assert mask.start == 0
    assert mask.stop == 4
    assert mask.step == 1

def test_series_view():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    keys = ["a", "b", "c", "d"]
    series = cloth.Series(values, keys)
    head = series.head(2)
    assert head.length() == 2
    assert head.sum() == 3.0

    tail = series.tail(2)
    assert tail.length() == 2
    assert tail.sum() == 7.0

def test_series_iloc_slice():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    keys = ["a", "b", "c", "d"]
    series = cloth.Series(values, keys)
    sl = cloth.slice(1, 3, 1)
    subseries = series.iloc[sl]
    assert subseries.length() == 2
    assert subseries.sum() == 5.0

def test_dataframe():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    index = ["row1", "row2", "row3", "row4"]
    columns = ["col1", "col2"]
    df = cloth.DataFrame(values, index, columns)
    assert df.rows() == 4
    assert df.cols() == 2
    assert df["col1"].sum() == 16.0
    assert df["col2"].mean() == 5.0

def test_dataframe_repr():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    index = ["row1", "row2", "row3", "row4"]
    columns = ["col1", "col2"]
    df = cloth.DataFrame(values, index, columns)
    assert isinstance(str(df), str)

def test_dataframe_mask():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    index = ["row1", "row2", "row3", "row4"]
    columns = ["col1", "col2"]
    df = cloth.DataFrame(values, index, columns)
    mask = df.mask
    assert mask.start == 0
    assert mask.stop == 4
    assert mask.step == 1

def test_dataframe_iloc_slice():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    index = ["row1", "row2", "row3", "row4"]
    columns = ["col1", "col2"]
    df = cloth.DataFrame(values, index, columns)

    sl = cloth.slice(1, 3, 1)
    subdf = df.iloc[sl]
    assert subdf.rows() == 2
    assert subdf["col1"].sum() == 8.0
    assert subdf["col2"].sum() == 10.0

def test_dataframe_loc():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    index = ["row1", "row2", "row3", "row4"]
    columns = ["col1", "col2"]
    df = cloth.DataFrame(values, index, columns)

    series = df.loc["row2"]
    assert series.sum() == 7.0

    sliced_df = df.loc["row2":"row4"]
    assert sliced_df.rows() == 3
    assert sliced_df["col1"].sum() == 15.0

if __name__ == "__main__":
    pytest.main()