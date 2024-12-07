from typing import cast, overload, Literal, Callable, TypeVar, ParamSpec, Any
from dataclasses import dataclass
from functools import wraps, lru_cache as _lru_cache


T = TypeVar('T')
P = ParamSpec('P')


class NotSet:
    pass


@dataclass(slots=True, init=False)
class Slot:
    expires: float
    value: Any


@overload
def lru_cache(  # noqa: F811
    maxsize: Callable[P, T],
    typed: bool = False,
) -> Callable[P, T]: ...


@overload
def lru_cache(  # noqa: F811
    maxsize: int | None,
    typed: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def lru_cache(  # noqa: F811
    maxsize: Callable[P, T] | int | None = 128,
    typed: bool = False,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    return cast(
        Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]],
        _lru_cache(maxsize, typed),
    )


def cache(f: Callable[P, T]) -> Callable[P, T]:
    'Simple lightweight unbounded cache.  Sometimes called "memoize".'
    return lru_cache(maxsize=None)(f)


@overload
def alru_cache(
    f: Literal[None],
    *,
    maxsize: int | None = 128,
    typed: bool = False,
    ttl: int = 0,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


@overload
def alru_cache(  # noqa: F811
    f: Callable[P, T],
    *,
    maxsize: int | None = 128,
    typed: bool = False,
    ttl: int = 0,
) -> Callable[P, T]: ...


def alru_cache(  # noqa: F811
    f=None,
    *,
    maxsize: int | None = 128,
    typed: bool = False,
    ttl: int = 0,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:  # noqa: F811
    """
    An async version of the functools.lru_cache().
    The decorator is written on top of the C-implemented built-in one,
    so it's 4-10 times faster than well-known async_lru.alru_cache.

    Parameters:
        f: Optional; the async function to be cached.
        maxsize: Optional int; the maximum size of the cache.
        typed: bool; whether to treat different types as different keys.
        ttl: Time-to-live for the cached value in seconds (off by default).

    Returns:
        The decorated async function with caching capabilities.
    """
    from time import monotonic

    @lru_cache(maxsize, typed)
    def getslot(*args, **kwargs):
        slot = Slot()
        slot.expires = 0
        slot.value = NotSet
        return slot

    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            slot = getslot(*args, **kwargs)

            if slot.value is NotSet:
                slot.value = await f(*args, **kwargs)

            return slot.value

        @wraps(f)
        async def wrapper_ttl(*args, **kwargs):
            slot = getslot(*args, **kwargs)

            if slot.value is NotSet or slot.expires < monotonic():
                slot.value = await f(*args, **kwargs)
                slot.expires = monotonic() + ttl

            return slot.value

        return wrapper_ttl if ttl else wrapper

    if f:
        return cast(Callable[P, T], decorator(f))
    else:
        return cast(Callable[[Callable[P, T]], Callable[P, T]], decorator)
