import numpy as np
import pytest
from ndsafe import ndsafearray  # Adjust this if your package structure is different


def test_sum_scalar():
    """Test sum over all elements (should return a scalar)."""
    a = ndsafearray(np.array([[1, 2, 3], [4, 5, 6]]))
    assert a.sum() == 21  # Scalar, not wrapped

def test_sum_axis_0():
    """Test sum along axis 0 (should return a 1D array)."""
    a = ndsafearray(np.array([[1, 2, 3], [4, 5, 6]]))
    expected = np.array([5, 7, 9])
    np.testing.assert_array_equal(a.sum(axis=0).unwrap(), expected)

def test_sum_axis_1():
    """Test sum along axis 1 (should return a 1D array)."""
    a = ndsafearray(np.array([[1, 2, 3], [4, 5, 6]]))
    expected = np.array([6, 15])
    np.testing.assert_array_equal(a.sum(axis=1).unwrap(), expected)

def test_sum_shape_1_n():
    """Test sum on a (1, n) shaped array (should return a 1D array for performance)."""
    a = ndsafearray(np.array([[1, 2, 3]]))  # (1,3)
    expected = np.array([6])  # Optimized to 1D
    np.testing.assert_array_equal(a.sum(), expected)

def test_sum_preserves_ndsafearray():
    """Test that sum with an axis preserves the ndsafearray type."""
    a = ndsafearray(np.array([[1, 2, 3], [4, 5, 6]]))
    assert isinstance(a.sum(axis=0), ndsafearray)

