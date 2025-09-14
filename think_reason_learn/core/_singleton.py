from typing import Any, TypeVar, ClassVar, Dict

T = TypeVar("T")


class SingletonMeta(type):
    _instances: ClassVar[Dict[type, object]] = {}

    def __call__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        if cls not in cls._instances:  # type: ignore
            cls._instances[cls] = super().__call__(*args, **kwargs)  # type: ignore
        return cls._instances[cls]  # type: ignore


__all__ = ["SingletonMeta"]
