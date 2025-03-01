import os
import tempfile
import warnings
from pathlib import Path
from uuid import uuid4

from typing_extensions import Any, Callable, Generic, TypeVar, get_args

__all__ = ["IndexWrapper", "load_index_from_temp_file", "save_index_to_temp_file"]

IS_WINDOWS = os.name == "nt"

T = TypeVar("T")  # Generic type for index wrapper


class IndexWrapper(Generic[T]):
    """A general purpose wrapper around indexing structures that require values observed from the data.

    For example, some indexing structures require an argument ``dim`` for the data dimensionality of the index,
    which might not be known at class instantiation time, but can automatically be inferred once data is available.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialization just stores the parameters.

        :param args: Any positional arguments to the underlying index.
        :param kwargs: Any keyword arguments to the underlying index.
        """
        super().__init__()
        if type(self) is IndexWrapper:
            msg = "Cannot instantiate 'IndexWrapper', subclass it with the correct generic index type."
            raise TypeError(msg)

        self.index: T | None = None  # might be set by subclass
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Initialize the underlying index using the original ``args`` and ``kwargs`` with additional parameters.

        :param kwargs: Keyword arguments to be supplied to the underlying index.
        :return: The instantiated underlying index.
        """
        # ``__orig_bases__`` bases returns the generic base class, e.g. ``IndexWrapper[usearch.Index]`` (should be one)
        # ``get_args`` extract the generic type arguments from the base, e.g. ``usearch.Index`` (should be one)
        index = self.index if self.index is not None else get_args(self.__class__.__orig_bases__[0])[0]  # type: ignore[reportAttributeAccessIssue]
        return index(*args, *self.args, **self.kwargs, **kwargs)  # type: ignore[reportCallIssue]


def save_index_to_temp_file(save_fn: Callable[..., Any], **save_kwargs: Any) -> bytes:
    # there might be cleanup errors in multiprocessing setting on windows
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=IS_WINDOWS) as tmp_dir:
        file_path = Path(tmp_dir) / uuid4().hex
        save_fn(str(file_path), **save_kwargs)
        return file_path.read_bytes()


def load_index_from_temp_file(index_bytes: bytes, load_fn: Callable[..., Any], **load_kwargs: Any) -> None:
    # cannot delete the file on windows, ``delete_on_close`` is only available with Python >= 3.12
    with tempfile.NamedTemporaryFile("wb", delete=not IS_WINDOWS) as file:
        file.write(index_bytes)
        file.flush()  # otherwise errors with 'Inappropriate ioctl for device'

        # it's not allowed to re-open the file on windows
        # https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
        if IS_WINDOWS:
            file.close()

        # open the file and load the index
        load_fn(file.name, **load_kwargs)

        # manually delete the file on windows
        if IS_WINDOWS:
            try:
                file.close()
                Path(file.name).unlink(missing_ok=True)
            except PermissionError:
                msg = f"Could not delete file '{file.name}', the file might be used by another process."
                warnings.warn(msg, stacklevel=1)
