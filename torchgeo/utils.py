"""Common utilities."""

from typing import Tuple, TypeVar, Union

T = TypeVar("T", float, int)


def _to_tuple(value: Union[Tuple[T, T], T]) -> Tuple[T, T]:
    """Convert value to a tuple if it is not already a tuple.

    Args:
        value: input value

    Returns:
        value if value is a tuple, else (value, value)
    """
    if isinstance(value, (int, int)) or isinstance(value, (float, float)):
        return (value, value)
    else:
        return value
