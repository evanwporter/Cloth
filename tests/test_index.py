# import pytest

# @pytest.mark.parametrize("index_cls, data, expected_keys", [
#     (cloth.StringIndex, ["a", "b", "c", "d"], ["a", "b", "c", "d"]),
#     (cloth.DateTimeIndex, ["2023-08-14T15:23:45", "2023-08-15T16:24:50", "2023-08-16T17:25:55"], ["2023-08-14T15:23:45", "2023-08-15T16:24:50", "2023-08-16T17:25:55"]),
#     (cloth.RangeIndex, [0, 10, 1], ["0", "10", "1"]),
# ])

# def test_indexes(index_cls, data, expected_keys):
#     test_index_operations(index_cls, data, expected_keys)


# def test_index_operations(index_cls, data, expected_keys):
#     idx = index_cls(data)
    
#     assert idx.length() == len(expected_keys)
    
#     for key, expected in zip(expected_keys, data):
#         assert idx[key] == expected_keys.index(key)
    
#     assert str(idx) == f"{index_cls.__name__}({', '.join(expected_keys)})"
    
#     mask = idx.mask
#     assert mask.start == 0
#     assert mask.stop == len(expected_keys)
#     assert mask.step == 1
    
#     assert idx.keys() == expected_keys
