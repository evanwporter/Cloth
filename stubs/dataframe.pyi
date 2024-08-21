# dataframe.pyi

from typing import List
from series import Series
from index import StringIndex, Index_

class DataFrame:
    """
    A two-dimensional, size-mutable, and potentially heterogeneous tabular data structure.

    Parameters
    ----------
    values : List[List[float]]
        A 2D array of data values in the DataFrame.
    index : Index_
        The row labels for the DataFrame.
    columns : StringIndex
        The column labels for the DataFrame.

    Methods
    -------
    iloc() -> 'DataFrameIlocProxy'
        Returns an index-based selection proxy.
    loc() -> 'DataFrameLocProxy'
        Returns a label-based selection proxy.
    sum() -> Series
        Returns the sum of each column in the DataFrame.
    values() -> List[List[float]]
        Returns the values of the DataFrame as a 2D list.
    length() -> int
        Returns the number of rows in the DataFrame.
    rows() -> int
        Returns the number of rows in the DataFrame.
    cols() -> int
        Returns the number of columns in the DataFrame.
    index() -> Index_
        Returns the index (row labels) of the DataFrame.
    columns() -> StringIndex
        Returns the column labels of the DataFrame.
    __getitem__(key: str) -> Series
        Retrieves the column associated with the given key as a Series.
    """

    def __init__(self, values: List[List[float]], index: Index_, columns: StringIndex):
        """
        Initializes a DataFrame with values, index, and columns.

        Parameters
        ----------
        values : List[List[float]]
            A 2D array of data values.
        index : Index_
            The row labels for the DataFrame.
        columns : StringIndex
            The column labels for the DataFrame.
        """
        ...

    def iloc(self) -> 'DataFrameIlocProxy':
        """
        Returns an index-based selection proxy.

        Returns
        -------
        DataFrameIlocProxy
            A proxy for index-based selection.
        """
        ...

    def loc(self) -> 'DataFrameLocProxy':
        """
        Returns a label-based selection proxy.

        Returns
        -------
        DataFrameLocProxy
            A proxy for label-based selection.
        """
        ...

    def sum(self) -> Series:
        """
        Returns the sum of each column in the DataFrame.

        Returns
        -------
        Series
            A series containing the sum of each column.
        """
        ...

    def values(self) -> List[List[float]]:
        """
        Returns the values of the DataFrame as a 2D list.

        Returns
        -------
        List[List[float]]
            A 2D list containing the data values.
        """
        ...

    def length(self) -> int:
        """
        Returns the number of rows in the DataFrame.

        Returns
        -------
        int
            The number of rows.
        """
        ...

    def rows(self) -> int:
        """
        Returns the number of rows in the DataFrame.

        Returns
        -------
        int
            The number of rows.
        """
        ...

    def cols(self) -> int:
        """
        Returns the number of columns in the DataFrame.

        Returns
        -------
        int
            The number of columns.
        """
        ...

    def index(self) -> Index_:
        """
        Returns the index (row labels) of the DataFrame.

        Returns
        -------
        Index_
            The row labels of the DataFrame.
        """
        ...

    def columns(self) -> StringIndex:
        """
        Returns the column labels of the DataFrame.

        Returns
        -------
        StringIndex
            The column labels of the DataFrame.
        """
        ...

    def __getitem__(self, key: str) -> Series:
        """
        Retrieves the column associated with the given key as a Series.

        Parameters
        ----------
        key : str
            The column label to lookup.

        Returns
        -------
        Series
            The column data as a Series.
        """
        ...

class DataFrameIlocProxy:
    """
    A proxy for index-based selection on a DataFrame.

    Methods
    -------
    __getitem__(idx: int) -> Series
        Retrieves the row at the specified index position as a Series.
    __getitem__(overlay: slice) -> DataFrame
        Retrieves a slice of the DataFrame.
    """

    def __getitem__(self, idx: int) -> Series:
        """
        Retrieves the row at the specified index position as a Series.

        Parameters
        ----------
        idx : int
            The index position of the row.

        Returns
        -------
        Series
            The row data as a Series.
        """
        ...

    def __getitem__(self, overlay: slice) -> DataFrame:
        """
        Retrieves a slice of the DataFrame.

        Parameters
        ----------
        overlay : slice
            The slice to retrieve.

        Returns
        -------
        DataFrame
            The sliced DataFrame.
        """
        ...

class DataFrameLocProxy:
    """
    A proxy for label-based selection on a DataFrame.

    Methods
    -------
    __getitem__(key: str) -> Series
        Retrieves the row associated with the specified label as a Series.
    """

    def __getitem__(self, key: str) -> Series:
        """
        Retrieves the row associated with the specified label as a Series.

        Parameters
        ----------
        key : str
            The row label to lookup.

        Returns
        -------
        Series
            The row data as a Series.
        """
        ...
