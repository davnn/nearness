import inspect
import threading
import types
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import fields, make_dataclass
from functools import partial, wraps
from logging import getLogger
from pathlib import Path
from warnings import warn

import joblib
import numpy as np
from joblib import Parallel, delayed
from safecheck import typecheck
from typing_extensions import Any, Callable, Self

from ._config import Config, config
from ._experimental import experimental

logger = getLogger(__name__)

__all__ = [
    "InvalidSignatureError",
    "NearestNeighbors",
]


class InvalidSignatureError(ValueError): ...


class NearestNeighborsMeta(ABCMeta):
    """Metaclass for NearestNeighbors.

    This metaclass is used to:
    1. Check if the implemented ``__init__`` only uses keyword-only parameters.
    2. Dynamically set an attribute ``_parameters_`` containing all ``__init__`` parameters.
    3. Dynamically set an attribute ``_config_`` containing a copy of the global ``config`` object.

    The parameters are modeled as a ``dataclasses.dataclass``, therefore, allowing:
    1. Parameter access using ``instance.parameters.name`` or ``getattr(instance.parameters, name)``
    2. Parameter updates using ``instance.parameters.name = value`` or ``setattr(instance.parameters, name, value)``
    """

    def __new__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> type:
        """Check the signature of ``__init__`` to ensure keyword-only arguments."""
        if "__init__" in attrs:
            parameters = inspect.signature(attrs["__init__"]).parameters
            for param_name, parameter in parameters.items():
                if param_name != "self" and (kind := parameter.kind) is not inspect.Parameter.KEYWORD_ONLY:
                    msg = (
                        "Only keyword-only arguments are allowed for classes inheriting from 'nearness."
                        f"NearestNeighbors', but found parameter '{parameter}' of kind '{kind}'."
                        f" Hint: You can enforce keyword-only arguments using the 'single star' syntax before the"
                        f" parameters, e.g. '___init___(self, *, {parameter}, ...), see "
                        f"https://peps.python.org/pep-3102/."
                    )
                    raise InvalidSignatureError(msg)

        return super().__new__(cls, name, bases, attrs)

    def __call__(cls, *_: Any, **kwargs: Any) -> "NearestNeighbors":
        """Dynamically set the ``_parameters_`` attribute and wrap the fit method and query methods.

        :param _: We ensure that there are only keyword-only arguments in ``__new__``.
        :param kwargs: The mapping of parameters of the algorithm.
        :return: The modified object.
        """
        # filter the self parameter
        parameters = {k: v for k, v in inspect.signature(cls.__init__).parameters.items() if k != "self"}

        # We don't want threads to override class-bound ``_parameters_`` or ``_config_``.
        # Because we use thread local storage, the ``_config_`` and ``_parameters`` attribute on the class should
        # never be visible to another parallel class instantiation.
        thread_local = threading.local()

        # Set the default config, such that it can be manipulated in ``__init__``, and deepcopy the config such that
        # a manipulation of ``cls._config_`` does not manipulate the global ``config`` object.
        # We must set the attribute here before ``__call__``, because the config should be usable in ``__init__``.
        # Setting the class attributes enable thread-local access to the config and parameters, which is later on
        # bound to the object, it's a bit magical, but useful to manipulate the config and params in init.
        thread_local.config = deepcopy(config)
        cls._config_ = thread_local.config  # type: ignore[reportUninitializedInstanceVariable]
        logger.info("Determined config object as '%s'", cls._config_)

        # same for the parameters, we set the parameters before the call such that they are usable in ``__init__``
        thread_local.parameters = _create_parameter_class(parameters, kwargs)
        cls._parameters_ = thread_local.parameters  # type: ignore[reportUninitializedInstanceVariable]
        logger.info("Determined parameters as '%s'", cls._parameters_)

        # now we set all the relevant attributes on the ``instance``, as they should not be class-bound.
        # order is important here as ``_wrap_fit_method`` and ``_wrap_check_method`` depend on the set attributes
        obj = type.__call__(cls, **kwargs)
        obj._parameters_, obj._config_ = cls._parameters_, cls._config_
        # make sure that the wrapped methods are in sync when the config is changed after class instantiation
        obj._config_.register_callback("methods_require_fit", partial(_check_callback, obj=obj))
        if not hasattr(obj, "__fitted__"):
            msg = (
                f"Instantiated {obj}, but missing the '__fitted__' attribute, which is automatically set to False in "
                f"'NearestNeighbors.__init__', did you forget to call 'super().__init__()' in the '__init__' of "
                f"{obj}? Assuming that '__fitted__' is 'False'."
            )
            warn(msg, stacklevel=1)
            obj.__fitted__ = False

        # __fitted__ might be true if the index is pre-loaded in the ``__init__``.
        if not obj.__fitted__:
            _wrap_check_method(obj)

        # always wrap the fit method of the object
        _wrap_fit_method(obj)

        # delete the temporary class variables
        del cls._parameters_
        del cls._config_
        return obj


class NearestNeighbors(metaclass=NearestNeighborsMeta):
    """Abstract base class for nearest neighbors search algorithms.

    The minimal implementation consists of a ``fit`` and ``query`` method. The default ``query_batch`` method uses
    a thread pool to parallelize calls to ``query``, but all methods may be overridden for efficiency purposes.
    It is expected that ``query`` returns a tuple of indices and distances to the nearest neighbors.

    The default interface to all methods is based on NumPy N-dimensional arrays, but implementations might ``overload``
    the methods such that other data types can be implemented. An important consideration for all possible methods
    and overloads is that they should be type stable, preferably allowing only floating-point arrays as input
    and returning floating-point distances of equal type as output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.__fitted__: bool = False

    @abstractmethod
    def fit(self, data: np.ndarray) -> Self:
        """Learn an index structure based on a matrix of points.

        :param data: matrix of ``size x dim``.
        :return: reference to object (``self``).
        """
        ...

    @abstractmethod
    def query(self, point: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
        """Search ``n_neighbors`` for a single point, returning the indices and distances.

        :param point: vector of ``dim``.
        :param n_neighbors: number of neighbors to search.
        :return:  (vector of indices, vector of distances) of size ``n_neighbors``
        """
        ...

    @experimental
    def query_idx(self, point: np.ndarray, n_neighbors: int) -> np.ndarray:
        """Search ``n_neighbors`` for a single point, returning the indices.

        :param point: vector of ``dim``.
        :param n_neighbors: number of neighbors to search.
        :return: vector of indices of size ``n_neighbors``
        """
        idx, _ = self.query(point, n_neighbors)
        return idx

    @experimental
    def query_dist(self, point: np.ndarray, n_neighbors: int) -> np.ndarray:
        """Search ``n_neighbors`` for a single point (vector), returning the distances.

        :param point: vector of ``dim``.
        :param n_neighbors: number of neighbors to search.
        :return: vector of distances of size ``n_neighbors``
        """
        _, dist = self.query(point, n_neighbors)
        return dist

    def query_batch(self, points: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
        """Search ``n_neighbors`` for a batch of points, returning the indices and distances.

        :param points: matrix of ``batch x dim``.
        :param n_neighbors: number of neighbors to search.
        :return: (matrix of indices, matrix of distances) of size ``batch x n_neighbors``.
        """
        result = Parallel(prefer="threads")(delayed(self.query)(q, n_neighbors) for q in points)
        idx, dist = zip(*result)
        return np.stack(idx), np.stack(dist)

    @experimental
    def query_batch_idx(self, points: np.ndarray, n_neighbors: int) -> np.ndarray:
        """Search ``n_neighbors`` for a batch of points, returning the indices.

        :param points: matrix of ``batch x dim``.
        :param n_neighbors: number of neighbors to search.
        :return: matrix of indices ``batch x n_neighbors``.
        """
        idx, _ = self.query_batch(points, n_neighbors)
        return idx

    @experimental
    def query_batch_dist(self, points: np.ndarray, n_neighbors: int) -> np.ndarray:
        """Search ``n_neighbors`` for a batch of points, returning the distances.

        :param points: matrix of ``batch x dim``.
        :param n_neighbors: number of neighbors to search.
        :return: matrix of distances ``batch x n_neighbors``.
        """
        _, dist = self.query_batch(points, n_neighbors)
        return dist

    def add(self, data: np.ndarray) -> Self:
        """Partially add elements to the index.

        Not all methods implement ``add``, but this can be useful if the data does not fit into memory.

        :param data: Data to add to the index.
        :raises: NotImplementedError if ``add`` is not available for the class.
        """
        msg = (
            f"Method 'add' is not implemented for '{self}', use 'fit' to learn the entire database at once, "
            f"or use a different indexing structure that enables partial addition of data. "
            f"If you would like to implement 'add' for '{self}', please open a pull request at "
            f"https://github.com/davnn/nearness"
        )
        raise NotImplementedError(msg)

    def save(self, file: str | Path) -> None:
        """Save the state of the model using pickle such that it can be fully restored using ``load``.

        :param file: name or path of the file to save to.
        :return: nothing.
        """
        joblib.dump(
            self,
            filename=file,
            protocol=self.config.save_protocol,
            compress=self.config.save_compression,
        )

    @staticmethod
    def load(file: str | Path) -> "NearestNeighbors":
        """Load a model using pickle to fully restore the saved state.

        :param file: name or path of the file to load from.
        :return: the restored ``NearestNeighbors`` algorithm.
        """
        return joblib.load(file)

    def __setstate__(self, state: dict[str, Any]) -> None:
        parameter_types, parameter_values = state["_parameters_"]
        state["_parameters_"] = make_dataclass("Parameters", parameter_types)(**parameter_values)
        self.__dict__.update(state)
        _wrap_fit_method(self)  # re-wrap the fit method
        self.is_fitted = self.__fitted__  # wrap or unwrap the check methods depending on ``__fitted__``

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()

        # it is very important to modify the save state and not modify the methods, otherwise the
        # methods would be modified in-place and the object would be invalid after ``__getstate__``.
        for method in self.config.methods_require_fit:
            if method in state:
                state[method] = m.__wrapped__ if hasattr(m := state[method], "__check__") else m

        state["fit"] = (  # unwrap the fit method if wrapped
            self.fit.__wrapped__  # type: ignore[reportFunctionMemberAccess]
            if hasattr(self.fit, "__fit__")
            else self.fit
        )

        parameter_fields = fields(self._parameters_)
        parameter_types = [(f.name, f.type) for f in parameter_fields]
        parameter_values = {f.name: getattr(self._parameters_, f.name) for f in parameter_fields}
        state["_parameters_"] = (parameter_types, parameter_values)
        return state

    @property
    def is_fitted(self) -> bool:
        return self.__fitted__

    @is_fitted.setter
    @typecheck
    def is_fitted(self, value: bool) -> None:
        if value:
            _unwrap_check_method(self)
        else:
            _wrap_check_method(self)

        # this variable is initialized in the metaclass
        self.__fitted__ = value

    @property
    def config(self) -> "Config":
        return self._config_

    @property
    def parameters(self) -> Any:
        """The parameters are dynamically set on class creation due to the ``NearestNeighborsMeta`` metaclass.

        All ``__init__`` arguments are considered parameters, and it is suggested to type-hint every parameter.
        """
        return self._parameters_


def _create_parameter_class(
    parameters: dict[str, inspect.Parameter],
    kwargs: dict[str, Any],
) -> "types.Parameters":  # type: ignore[reportGeneralTypeIssues]
    empty = inspect.Parameter.empty
    parameter_types = [(k, Any if (a := v.annotation) is empty else a) for k, v in parameters.items()]
    parameter_values = {k: kwargs.get(k, v.default) for k, v in parameters.items()}
    return make_dataclass("Parameters", parameter_types)(**parameter_values)


def _create_check_wrapper(obj: NearestNeighbors, method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug("Called method with fit check enabled.")
        if not obj.is_fitted:
            msg = f"Attempted to call '{method.__qualname__}', but 'fit' has not yet been called."
            raise AssertionError(msg)
        return method(*args, **kwargs)

    # we set a ``__check__`` attribute, to be able to (easily) distinguish the wrapper from the bound method
    wrapper.__check__ = True  # type: ignore[reportFunctionMemberAccess]
    return wrapper


def _wrap_fit_method(obj: "NearestNeighbors") -> None:
    """Wrap the ``fit`` method to ensure it sets ``is_fitted`` to ``True``, which unwraps the query method checks.

    This feels a bit like magic, but the alternatives would be to:
    1. Add a decorator to every ``fit`` method that sets ``is_fitted`` to ``True``.
    2. Set ``is_fitted`` on every ``fit``-like method and check for ``is_fitted`` everywhere.
    """
    logger.debug("Wrapping fit method.")
    original_fit = obj.fit

    @wraps(original_fit)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original_fit(*args, **kwargs)
        logger.debug("Called wrapped fit method with successful fit result, setting '__fitted__'.")
        obj.is_fitted = True
        return result

    # we set a ``__fit__`` attribute to denote that the method has been decorated
    wrapper.__fit__ = True  # type: ignore[reportGeneralTypeIssues]
    obj.fit = wrapper  # override the default fit with the wrapped fit


def _wrap_check_method(obj: "NearestNeighbors") -> None:
    """Wrap or unwrap methods according to ``methods_require_fit``.

    For each wrapped method, we ensure that ``__fitted__`` is True, before the method can be called.
    Methods not in ``methods_require_fit`` are unwrapped.
    """
    logger.debug("Starting to wrap methods to enable fit checking.")
    available_attributes = dir(obj)
    methods_to_wrap = obj._config_.methods_require_fit
    for attribute_name in available_attributes:
        attribute = getattr(obj, attribute_name)
        has_check = hasattr(attribute, "__check__")
        if attribute_name in methods_to_wrap:
            if has_check:
                continue
            # check if the ``method`` attribute is a bound method
            if isinstance(attribute, types.MethodType):
                logger.debug("Wrapping method '%s'.", attribute_name)
                # update the method with the wrapped ``__fitted__`` check
                setattr(obj, attribute_name, wraps(attribute)(_create_check_wrapper(obj, attribute)))
            else:
                msg = (
                    f"Attempting to enable '__fitted__' check for invalid attribute '{attribute_name}', because "
                    f"'{attribute_name}' is not a bound method, instead it's an attribute of '{type(attribute)}'."
                )
                warn(msg, stacklevel=1)
        elif has_check:  # if an attribute has a ``__check__`` attribute, but is not in the set to check, we unwrap.
            logger.debug(
                "Attribute '%s' is not in '%s', unwrapping the attribute.",
                attribute_name,
                methods_to_wrap,
            )
            setattr(obj, attribute_name, attribute.__wrapped__)

    for attribute_name in methods_to_wrap:
        if attribute_name not in available_attributes:
            msg = (
                f"Attempting to enable '__fitted__' check for missing attribute '{attribute_name}', because "
                f"'{attribute_name}' does not exist on class {obj}."
            )
            warn(msg, stacklevel=1)


def _unwrap_check_method(obj: "NearestNeighbors") -> None:
    """Unwrap all existing methods in ``methods_require_fit`` to disable the ``__fitted__`` check.

    This is just a performance optimization, it retrieves the original method and removes the implicitly
    generated function wrapper (decorator). It should be safe to unwrap the methods if ``__fitted__``
    is set in ``fit``, but unsafe when ``__fitted__`` is manually set to ``False`` after ``fit``.
    """
    logger.debug("Starting to unwrap all fit checking methods.")
    for method_name in obj._config_.methods_require_fit:
        # we set an __requires_fit__ attribute on the wrapper, because using ``__wrapped__`` alone is not
        # safe (methods also use ``__wrapped__`` starting with Python 3.10)
        if hasattr(obj, method_name) and hasattr(method := getattr(obj, method_name), "__check__"):
            logger.debug("Unwrapping method '%s'.", method_name)
            setattr(obj, method_name, method.__wrapped__)


def _check_callback(_: Any, *, obj: "NearestNeighbors") -> None:
    """A callback to refresh the ``__fitted__`` attributes when the 'methods_require_fit' attribute is updated.

    :param _: The set attribute value (ignored).
    :param obj: The object, where the attribute is set.
    :return: None.
    """
    if not obj.is_fitted:  # if the object is already fitted there is no need to wrap any methods
        _wrap_check_method(obj)
