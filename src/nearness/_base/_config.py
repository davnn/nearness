from dataclasses import asdict, dataclass

from typing_extensions import Any, Callable

__all__ = ["Config", "config"]


@dataclass
class Config:
    """The global NearestNeighbors configuration.

    Each instance of ``NearestNeighbors`` creates a current (deep) copy of the global ``config`` object. It is,
    therefore, possible to adapt class behaviour before instantiation by modification of ``config``, and, because
    each class owns a copy of ``config``, it can be configured individually.

    We further provide a simple event interface using callbacks. You can observe ``__setattr__`` events using
    ``register_callback`` and provide a function receiving the newly set value. To enable the callback-based
    observability it's necessary that all fields are immutable, the only way to update a field is to set it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._callbacks = {}

    methods_require_fit: frozenset[str] = frozenset(
        {
            "query",
            "query_idx",
            "query_dist",
            "query_batch",
            "query_batch_idx",
            "query_batch_dist",
        },
    )
    save_protocol: int | None = None
    save_compression: int = 0

    def register_callback(self, field_name: str, callback: Callable[[Any], Any]) -> None:
        if field_name in asdict(self):
            # initialize the list of callbacks if there is no callback yet
            if field_name not in self._callbacks:
                self._callbacks[field_name] = []
            self._callbacks[field_name].append(callback)
        else:
            msg = (
                f"Cannot register a callback for attribute '{field_name}', attribute "
                f"does not exist on class {self}."
            )
            raise AssertionError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in self.__annotations__:
            callbacks = self._callbacks.get(name, [])
            for callback in callbacks:
                callback(value)


config = Config()
