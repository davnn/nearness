import warnings
from functools import wraps

__all__ = [
    "ExperimentalWarning",
    "experimental",
]

from typing_extensions import Any, Callable

EXPERIMENTAL_WARNING_HELP = (
    "To mute warnings for experimental functionality, invoke"
    ' warnings.filterwarnings("ignore", category=nearness.ExperimentalWarning) or use'
    " one of the other methods described at"
    " https://docs.python.org/3/library/warnings.html#describing-warning-filters."
)


def experimental(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        experimental_warning(func.__name__)
        return func(*args, **kwargs)

    return wrapper


def experimental_warning(
    subject: str,
    additional_warn_text: str | None = None,
    stacklevel: int = 3,
) -> None:
    extra_text = f" {additional_warn_text}" if additional_warn_text else ""
    warnings.warn(
        (
            f"{subject} is experimental. It may break in future versions, even between dot"
            f" releases.{extra_text} {EXPERIMENTAL_WARNING_HELP}"
        ),
        ExperimentalWarning,
        stacklevel=stacklevel,
    )


class ExperimentalWarning(Warning):
    pass
