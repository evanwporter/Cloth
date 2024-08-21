# series.pyi

from typing import List, Union
from index import StringIndex, Index_
from datetime64 import timedelta, datetime

class Series:
    """
    A one-dimensional array-like object containing a sequence of values.

    Parameters
    ----------
    values : List[float]
        The data values in the series.
    index : StringIndex
        The index labels for the series.

    Methods
    -------
    sum() -> float
        Returns the sum of the values in the series.
    mean() -> float
        Returns the mean of the values in the series.
    min() -> float
        Returns the minimum value in the series.
    max() -> float
        Returns the maximum value in the series.
    values() -> List[float]
        Returns the values of the series as a list.
    length() -> int
        Returns the number of elements in the series.
    iloc() -> 'SeriesIlocProxy'
        Returns an index-based selection proxy.
    loc() -> 'SeriesLocProxy'
        Returns a label-based selection proxy.
    head(n: int) -> 'Series'
        Returns the first `n` rows of the series.
    tail(n: int) -> 'Series'
        Returns the last `n` rows of the series.
    index() -> Index_
        Returns the index of the series.
    __getitem__(key: Union[str, int]) -> float
        Retrieves the value associated with the given key.
    """

    def __init__(self, values: List[float], index: StringIndex):
        """
        Initializes a Series with values and an index.

        Parameters
        ----------
        values : List[float]
            The data values in the series.
        index : StringIndex
            The index labels for the series.
        """
        ...

    def sum(self) -> float:
        """
        Returns the sum of the values in the series.

        Returns
        -------
        float
            The sum of the values.
        """
        ...

    def mean(self) -> float:
        """
        Returns the mean of the values in the series.

        Returns
        -------
        float
            The mean of the values.
        """
        ...

    def min(self) -> float:
        """
        Returns the minimum value in the series.

        Returns
        -------
        float
            The minimum value.
        """
        ...

    def max(self) -> float:
        """
        Returns the maximum value in the series.

        Returns
        -------
        float
            The maximum value.
        """
        ...

    def values(self) -> List[float]:
        """
        Returns the values of the series as a list.

        Returns
        -------
        List[float]
            The values in the series.
        """
        ...

    def length(self) -> int:
        """
        Returns the number of elements in the series.

        Returns
        -------
        int
            The number of elements in the series.
        """
        ...

    def iloc(self) -> 'SeriesIlocProxy':
        """
        Returns an index-based selection proxy.

        Returns
        -------
        SeriesIlocProxy
            A proxy for index-based selection.
        """
        ...

    def loc(self) -> 'SeriesLocProxy':
        """
        Returns a label-based selection proxy.

        Returns
        -------
        SeriesLocProxy
            A proxy for label-based selection.
        """
        ...

    def head(self, n: int) -> 'Series':
        """
        Returns the first `n` rows of the series.

        Parameters
        ----------
        n : int
            Number of rows to return.

        Returns
        -------
        Series
            The first `n` rows of the series.
        """
        ...

    def tail(self, n: int) -> 'Series':
        """
        Returns the last `n` rows of the series.

        Parameters
        ----------
        n : int
            Number of rows to return.

        Returns
        -------
        Series
            The last `n` rows of the series.
        """
        ...

    def index(self) -> Index_:
        """
        Returns the index of the series.

        Returns
        -------
        Index_
            The index of the series.
        """
        ...

    def __getitem__(self, key: Union[str, int]) -> float:
        """
        Retrieves the value associated with the given key.

        Parameters
        ----------
        key : Union[str, int]
            The key (either a string label or an integer index).

        Returns
        -------
        float
            The value associated with the key.
        """
        ...

class SeriesIlocProxy:
    """
    A proxy for index-based selection on a Series.

    Methods
    -------
    __getitem__(idx: int) -> float
        Retrieves the value at the specified index position.
    __getitem__(overlay: slice) -> Series
        Retrieves a slice of the series.
    """

    def __getitem__(self, idx: int) -> float:
        """
        Retrieves the value at the specified index position.

        Parameters
        ----------
        idx : int
            The index position.

        Returns
        -------
        float
            The value at the specified index position.
        """
        ...

    def __getitem__(self, overlay: slice) -> Series:
        """
        Retrieves a slice of the series.

        Parameters
        ----------
        overlay : slice
            The slice to retrieve.

        Returns
        -------
        Series
            The sliced series.
        """
        ...

class SeriesLocProxy:
    """
    A proxy for label-based selection on a Series.

    Methods
    -------
    __getitem__(key: str) -> float
        Retrieves the value associated with the specified label.
    __getitem__(key: datetime) -> float
        Retrieves the value associated with the specified datetime label.
    """

    def __getitem__(self, key: str) -> float:
        """
        Retrieves the value associated with the specified label.

        Parameters
        ----------
        key : str
            The label to lookup.

        Returns
        -------
        float
            The value associated with the label.
        """
        ...

    def __getitem__(self, key: datetime) -> float:
        """
        Retrieves the value associated with the specified datetime label.

        Parameters
        ----------
        key : datetime
            The datetime label to lookup.

        Returns
        -------
        float
            The value associated with the datetime label.
        """
        ...
