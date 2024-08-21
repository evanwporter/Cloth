# datetime64.pyi

from typing import Union

class timedelta:
    """
    A class representing a duration, the difference between two dates or times.

    Parameters
    ----------
    data : Union[int, str], optional
        The duration value as an integer (default is 0).

    Methods
    -------
    __add__(other: 'timedelta') -> 'timedelta'
        Adds another timedelta to this one.
    __sub__(other: 'timedelta') -> 'timedelta'
        Subtracts another timedelta from this one.
    __mul__(other: int) -> 'timedelta'
        Multiplies this timedelta by an integer.
    data() -> int
        Returns the internal data representation of the timedelta.
    """

    def __init__(self, data: Union[int, str] = 0):
        """
        Initializes the timedelta with a given duration.

        Parameters
        ----------
        data : Union[int, str], optional
            The duration value as an integer or string (default is 0).
        """
        ...

    def __add__(self, other: 'timedelta') -> 'timedelta':
        """
        Adds another timedelta to this one.

        Parameters
        ----------
        other : timedelta
            The other timedelta to add.

        Returns
        -------
        timedelta
            The result of the addition.
        """
        ...

    def __sub__(self, other: 'timedelta') -> 'timedelta':
        """
        Subtracts another timedelta from this one.

        Parameters
        ----------
        other : timedelta
            The other timedelta to subtract.

        Returns
        -------
        timedelta
            The result of the subtraction.
        """
        ...

    def __mul__(self, other: int) -> 'timedelta':
        """
        Multiplies this timedelta by an integer.

        Parameters
        ----------
        other : int
            The integer to multiply the timedelta by.

        Returns
        -------
        timedelta
            The result of the multiplication.
        """
        ...

    def data(self) -> int:
        """
        Returns the internal data representation of the timedelta.

        Returns
        -------
        int
            The internal data as an integer.
        """
        ...

class datetime:
    """
    A class representing a point in time.

    Parameters
    ----------
    data : Union[int, str]
        The datetime value as an integer or string.

    Methods
    -------
    __add__(delta: timedelta) -> 'datetime'
        Adds a timedelta to this datetime.
    __sub__(other: Union['datetime', timedelta]) -> Union['timedelta', 'datetime']
        Subtracts another datetime or timedelta from this datetime.
    floor(delta: timedelta) -> 'datetime'
        Floors the datetime to the nearest multiple of a timedelta.
    ceil(delta: timedelta) -> 'datetime'
        Ceils the datetime to the nearest multiple of a timedelta.
    seconds() -> int
        Returns the number of seconds since the epoch.
    minutes() -> int
        Returns the number of minutes since the epoch.
    hours() -> int
        Returns the number of hours since the epoch.
    days() -> int
        Returns the number of days since the epoch.
    weeks() -> int
        Returns the number of weeks since the epoch.
    years(start_year: int) -> int
        Returns the number of years since a given start year.
    months(start_year: int, start_month: int) -> int
        Returns the number of months since a given start year and month.
    to_iso() -> str
        Returns the datetime in ISO 8601 format.
    """

    def __init__(self, data: Union[int, str]):
        """
        Initializes the datetime with a given point in time.

        Parameters
        ----------
        data : Union[int, str]
            The datetime value as an integer or string.
        """
        ...

    def __add__(self, delta: timedelta) -> 'datetime':
        """
        Adds a timedelta to this datetime.

        Parameters
        ----------
        delta : timedelta
            The timedelta to add.

        Returns
        -------
        datetime
            The resulting datetime after addition.
        """
        ...

    def __sub__(self, other: Union['datetime', timedelta]) -> Union['timedelta', 'datetime']:
        """
        Subtracts another datetime or timedelta from this datetime.

        Parameters
        ----------
        other : Union[datetime, timedelta]
            The other datetime or timedelta to subtract.

        Returns
        -------
        Union[timedelta, datetime]
            The result of the subtraction.
        """
        ...

    def floor(self, delta: timedelta) -> 'datetime':
        """
        Floors the datetime to the nearest multiple of a timedelta.

        Parameters
        ----------
        delta : timedelta
            The timedelta to floor to.

        Returns
        -------
        datetime
            The floored datetime.
        """
        ...

    def ceil(self, delta: timedelta) -> 'datetime':
        """
        Ceils the datetime to the nearest multiple of a timedelta.

        Parameters
        ----------
        delta : timedelta
            The timedelta to ceil to.

        Returns
        -------
        datetime
            The ceiled datetime.
        """
        ...

    def seconds(self) -> int:
        """
        Returns the number of seconds since the epoch.

        Returns
        -------
        int
            The number of seconds.
        """
        ...

    def minutes(self) -> int:
        """
        Returns the number of minutes since the epoch.

        Returns
        -------
        int
            The number of minutes.
        """
        ...

    def hours(self) -> int:
        """
        Returns the number of hours since the epoch.

        Returns
        -------
        int
            The number of hours.
        """
        ...

    def days(self) -> int:
        """
        Returns the number of days since the epoch.

        Returns
        -------
        int
            The number of days.
        """
        ...

    def weeks(self) -> int:
        """
        Returns the number of weeks since the epoch.

        Returns
        -------
        int
            The number of weeks.
        """
        ...

    def years(self, start_year: int) -> int:
        """
        Returns the number of years since a given start year.

        Parameters
        ----------
        start_year : int
            The year to calculate the difference from.

        Returns
        -------
        int
            The number of years.
        """
        ...

    def months(self, start_year: int, start_month: int) -> int:
        """
        Returns the number of months since a given start year and month.

        Parameters
        ----------
        start_year : int
            The year to calculate the difference from.
        start_month : int
            The month to calculate the difference from.

        Returns
        -------
        int
            The number of months.
        """
        ...

    def to_iso(self) -> str:
        """
        Returns the datetime in ISO 8601 format.

        Returns
        -------
        str
            The ISO 8601 formatted datetime string.
        """
        ...
