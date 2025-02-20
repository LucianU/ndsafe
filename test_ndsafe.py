import numpy as np
from ndsafe import ndsafearray


def test_getitem_preserves_dimensions():
    a = ndsafearray(np.array([[1, 2, 3], [4, 5, 6]]))

    np.testing.assert_array_equal(a[0].unwrap(), [[1, 2, 3]])
    np.testing.assert_array_equal(a[:, 1].unwrap(), [[2], [5]])

def test_sum_doesnt_wrap_scalar():
    """Test sum over all elements (should return a scalar)."""
    a = ndsafearray(np.array([[1, 2, 3], [4, 5, 6]]))
    assert a.sum() == 21  # Scalar, not wrapped

def test_mean_doesnt_wrap_scalar():
    a = ndsafearray(np.array([[1, 2, 3]]))
    c = a.mean()
    assert isinstance(c, float)

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

def test_reshape_preserves_ndsafearray():
    a = ndsafearray(np.array([[1, 2, 3]]))
    b = a.reshape(3, 1)  # Should return ndsafearray, not ndarray
    assert isinstance(b, ndsafearray)

def test_transpose_preserves_ndsafearray():
    a = ndsafearray(np.array([[1, 2, 3]]))
    d = a.T  # Should return an ndsafearray
    assert isinstance(d, ndsafearray)

