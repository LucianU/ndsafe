# ndsafe
A NumPy-compatible array that ensures shape safety and prevents silent errors.

## Why ndsafe?
NumPy operations behave differently depending on whether an array is 1D (`(n,)`)
or 2D (`(m, n)`). This can lead to **unexpected errors** in operations like:
- **Matrix multiplication (`@`)**
- **Dot product (`np.dot`)**
- **Transpose (`.T`)**

`ndsafe` eliminates these issues by:

✅ **Automatically reshaping 1D arrays to 2D when needed**

✅ **Ensuring reduction operations (sum, mean, etc.) remain efficient**

✅ **Preventing silent shape mismatches in mathematical operations**

## Installation
```sh
pip install ndsafe
```

## Usage
```py
import numpy as np
from ndsafe import ndsafearray

# Avoid silent errors with matrix operations
a = ndsafearray([1, 2, 3])
b = ndsafearray([[4], [5], [6]])

print(a @ b)  # No more unexpected shape errors!
```

## Key Features
* **Shape Safety**: Converts `(n,) → (1, n)` where needed
* **Reduction Optimizations**: Converts `(1, n) → (n,)` for efficiency
* **Seamless NumPy Interop**: Works with `np.sum()`, `np.dot()`, etc.
* **Consistent Type Handling**: Always returns `ndsafearray` unless a scalar

## License
MIT License © Lucian Ursu

