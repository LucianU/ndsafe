# ndsafe
A NumPy-compatible array that ensures shape safety and prevents silent errors.

## Why ndsafe?
NumPy operations behave differently depending on whether an array is **1D
(`(n,)`) or 2D (`(m, n)`)**.  This can cause **silent errors**, unexpected
behavior, or outright failures in numerical computations.

### 🚨 **Example: NumPy's Silent Shape Collapse**
```python
import numpy as np

a_unsafe = np.array([1, 2, 3])   # Shape (3,)
b_unsafe = np.array([[4], [5], [6]])  # Shape (3,1)

print(a_unsafe @ b_unsafe)  # ❌ Outputs: [32] (Unexpected 1D array)
```

### What went wrong?

We expected a proper matrix product that returns `(1,1)`, but NumPy collapses it
to `(1,)`. This can cause shape mismatches as some NumPy functions require
strictly 2D matrices and will fail on 1D inputs.

For example, np.linalg.pinv() raises an error when given a 1D array:

```
print(np.linalg.pinv(a_unsafe @ b_unsafe))  # ❌ LinAlgError: "Array must be at least two-dimensional"
```

### ✅ How `ndsafe` Fixes It

```python
from ndsafe import ndsafearray

a_safe = ndsafearray([1, 2, 3])  # Shape (1, 3)
b_safe = ndsafearray([[4], [5], [6]])  # Shape (3,1)

print(a_safe @ b_safe)  # ✅ Prints safe([[32]]) (Correct 1×1 matrix)
print(np.linalg.pinv(a_safe @ b_safe))  # ✅ Prints: safe([[0.03125]])
```

### `ndsafe`:

* Automatically reshapes 1D arrays to 2D when needed.
* Ensures reduction operations (sum, mean, etc.) remain efficient.
* Prevents silent shape mismatches in mathematical operations.

## Installation
```sh
pip install ndsafe
```

## Key Features
* **Shape Safety**: Converts `(n,) → (1, n)` where needed
* **Reduction Optimizations**: Converts `(1, n) → (n,)` for efficiency
* **Seamless NumPy Interop**: Works with `np.sum()`, `np.dot()`, etc.
* **Consistent Type Handling**: Always returns `ndsafearray` unless a scalar

## License
MIT License © Lucian Ursu

