# index.pyi

from typing import List, Union, Optional
from datetime64 import datetime

class Index_:
    """
    Base class for all index types.

    Methods
    -------
    length() -> int
        Returns the length of the index.
    mask() -> slice
        Returns a slice representing the mask applied to the index.
    __getitem__(key: Union[str, datetime, int]) -> int
        Retrieves the index position for a given key.
    keys() -> List[str]
        Returns the list of keys in the index.
    clone(mask: slice) -> 'Index_'
        Creates a copy of the index with the given mask applied.
    apply(view: 'BoolView') -> 'Index_'
        Applies a boolean view to the index and returns a new index.
    """

    def length(self) -> int:
        """
        Returns the length of the index.

        Returns
        -------
        int
            The number of elements in the index.
        """
        ...

    def mask(self) -> slice:
        """
        Returns a slice representing the mask applied to the index.

        Returns
        -------
        slice
            The mask as a slice object.
        """
        ...

    def __getitem__(self, key: Union[str, datetime, int]) -> int:
        """
        Retrieves the index position for a given key.

        Parameters
        ----------
        key : Union[str, datetime, int]
            The key to lookup in the index.

        Returns
        -------
        int
            The position of the key in the index.
        """
        ...

    def keys(self) -> List[str]:
        """
        Returns the list of keys in the index.

        Returns
        -------
        List[str]
            A list of keys in the index.
        """
        ...

    def clone(self, mask: slice) -> 'Index_':
        """
        Creates a copy of the index with the given mask applied.

        Parameters
        ----------
        mask : slice
            The mask to apply to the new index.

        Returns
        -------
        Index_
            A new index with the mask applied.
        """
        ...

    def apply(self, view: 'BoolView') -> 'Index_':
        """
        Applies a boolean view to the index and returns a new index.

        Parameters
        ----------
        view : BoolView
            The boolean view to apply.

        Returns
        -------
        Index_
            A new index with the boolean view applied.
        """
        ...

class StringIndex(Index_):
    """
    Index class for string keys.

    Parameters
    ----------
    keys : List[str]
        A list of string keys for the index.

    Methods
    -------
    __getitem__(key: str) -> int
        Retrieves the index position for a given string key.
    keys() -> List[str]
        Returns the list of string keys in the index.
    """

    def __init__(self, keys: List[str]):
        """
        Initializes the StringIndex with a list of keys.

        Parameters
        ----------
        keys : List[str]
            A list of string keys for the index.
        """
        ...

    def __getitem__(self, key: str) -> int:
        """
        Retrieves the index position for a given string key.

        Parameters
        ----------
        key : str
            The string key to lookup in the index.

        Returns
        -------
        int
            The position of the key in the index.
        """
        ...

    def keys(self) -> List[str]:
        """
        Returns the list of string keys in the index.

        Returns
        -------
        List[str]
            A list of string keys in the index.
        """
        ...

class DateTimeIndex(Index_):
    """
    Index class for datetime keys.

    Parameters
    ----------
    keys : List[datetime]
        A list of datetime keys for the index.

    Methods
    -------
    __getitem__(key: datetime) -> int
        Retrieves the index position for a given datetime key.
    keys() -> List[datetime]
        Returns the list of datetime keys in the index.
    """

    def __init__(self, keys: List[datetime]):
        """
        Initializes the DateTimeIndex with a list of datetime keys.

        Parameters
        ----------
        keys : List[datetime]
            A list of datetime keys for the index.
        """
        ...

    def __getitem__(self, key: datetime) -> int:
        """
        Retrieves the index position for a given datetime key.

        Parameters
        ----------
        key : datetime
            The datetime key to lookup in the index.

        Returns
        -------
        int
            The position of the key in the index.
        """
        ...

    def keys(self) -> List[str]:
        """
        Returns the list of datetime keys in the index.

        Returns
        -------
        List[datetime]
            A list of datetime keys in the index.
        """
        ...

class RangeIndex(Index_):
    """
    Index class for a range of integer keys.

    Parameters
    ----------
    start : int
        The start of the range.
    stop : int
        The end of the range.
    step : int, optional
        The step size of the range (default is 1).

    Methods
    -------
    __getitem__(index: int) -> int
        Retrieves the index position for a given integer key.
    keys() -> List[str]
        Returns the list of keys in the index.
    """

    def __init__(self, start: int, stop: int, step: int = 1):
        """
        Initializes the RangeIndex with a range of integer keys.

        Parameters
        ----------
        start : int
            The start of the range.
        stop : int
            The end of the range.
        step : int, optional
            The step size of the range (default is 1).
        """
        ...

    def __getitem__(self, index: int) -> int:
        """
        Retrieves the index position for a given integer key.

        Parameters
        ----------
        index : int
            The integer key to lookup in the index.

        Returns
        -------
        int
            The position of the key in the index.
        """
        ...

    def keys(self) -> List[str]:
        """
        Returns the list of keys in the index.

        Returns
        -------
        List[str]
            A list of keys in the index.
        """
        ...
