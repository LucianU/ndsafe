import numpy as np


class ndsafearray:
    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array)  # Ensure it's a NumPy array

    def __repr__(self):
        return f"safe({self._array})"

    def __getattr__(self, name):
        attr = getattr(self._array, name)

        if callable(attr):
            def wrapped_method(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, np.ndarray):  # Wrap ndarray outputs
                    return ndsafearray(result)
                return result  # Scalars and other types remain unchanged

            return wrapped_method

        return attr  # Non-callable attributes are returned directly

    def __getitem__(self, index):
        """Ensure indexing never drops dimensions below 2D"""
        result = self._array[index]

        if result.ndim < 2:  # Prevents silent reduction to 1D
            if isinstance(index, tuple) and any(isinstance(i, slice) for i in index):
                result = result.reshape(-1, 1)  # Column vector if slicing a single column
            else:
                result = result.reshape(1, -1)  # Row vector otherwise

        return ndsafearray(result)

    def unwrap(self):
        """Return the raw NumPy array"""
        return self._array

    @property
    def T(self):
        """Ensure 1D arrays transpose into column vectors"""
        if self._array.ndim == 1:
            self_array = self._array.reshape(1, -1)  # Convert to 2D row vector
        else:
            self_array = self._array

        return ndsafearray(self_array.T)

    def dot(self, other):
        """Ensure proper shape before performing dot product"""
        if self._array.ndim == 1:
            self_array = self._array.reshape(1, -1)  # Convert to 2D row vector
        else:
            self_array = self._array

        if isinstance(other, ndsafearray):
            other_array = other._array
        elif isinstance(other, np.ndarray):
            other_array = other
        else:
            raise TypeError("dot product requires an ndarray or ndsafearray")

        return ndsafearray(self_array @ other_array)  # Use matrix multiplication

    def __matmul__(self, other):
        """Ensure proper shape before performing matrix multiplication"""
        self_array = self._array.reshape(1, -1) if self._array.ndim == 1 else self._array

        if isinstance(other, ndsafearray):
            other_array = other._array
        elif isinstance(other, np.ndarray):
            other_array = other
        else:
            raise TypeError("Matrix multiplication requires an ndarray or ndsafearray")

        return ndsafearray(self_array @ other_array)

    # Reduction methods
    def _reduce(self, func, axis=None):
        """Generic method for reduction operations"""
        array = self._array.ravel() if self._array.shape[0] == 1 else self._array  # Flatten if (1, n)
        result = func(array, axis=axis)
        return result if np.isscalar(result) else ndsafearray(result.ravel())

    def sum(self, axis=None):
        return self._reduce(np.sum, axis)

    def mean(self, axis=None):
        return self._reduce(np.mean, axis)

    def prod(self, axis=None):
        return self._reduce(np.prod, axis)

    def min(self, axis=None):
        return self._reduce(np.min, axis)

    def max(self, axis=None):
        return self._reduce(np.max, axis)

    def std(self, axis=None):
        return self._reduce(np.std, axis)

    def var(self, axis=None):
        return self._reduce(np.var, axis)

    def median(self, axis=None):
        return self._reduce(np.median, axis)

    def percentile(self, axis=None):
        return self._reduce(np.percentile, axis)

    def quantile(self, axis=None):
        return self._reduce(np.quantile, axis)

    def any(self, axis=None):
        return self._reduce(np.any, axis)

    def all(self, axis=None):
        return self._reduce(np.all, axis)

    def ptp(self, axis=None):
        return self._reduce(np.ptp, axis)

