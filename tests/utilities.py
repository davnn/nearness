import pytest
import numpy as np
from typing_extensions import Any


def approx_equal(x: np.ndarray, y: np.ndarray, *arrays: np.ndarray, rtol: float = 1e-2, atol: float = 1e-4) -> bool:
    """Ensure that all input arrays and their elements are approximately equal."""
    result = np.allclose(x, y, rtol=rtol, atol=atol)
    for array in arrays:
        result = result and np.allclose(x, array, rtol=rtol, atol=atol)
    return result


def array_equal(x: np.ndarray, y: np.ndarray, *arrays: np.ndarray, equal_nan: bool = False) -> bool:
    """Ensure that all input arrays and their elements are equal."""
    result = np.array_equal(x, y, equal_nan=equal_nan)
    for array in arrays:
        result = result and np.array_equal(x, array, equal_nan=equal_nan)
    return result


def pytest_param_if_value_available(key: str, lazy_value: Any) -> Any:  # type: ignore[ANN401]
    try:
        return pytest.param(lazy_value(), id=key)
    except NameError:
        reason = f"Skipping test, {key} is not available."
        return pytest.param(None, id=key, marks=pytest.mark.skip(reason=reason))


def value_is_missing(lazy_value: Any) -> bool:
    try:
        lazy_value()
        return False
    except NameError:
        return True
