JAX Crash Course - Accelerating Machine Learning code!

https://www.youtube.com/watch?v=juo5G3t4qAo


Machine Learning with JAX - From Zero to Hero | Tutorial #1

https://www.youtube.com/watch?v=SstuvS-tVc0

https://www.youtube.com/watch?v=CQQaifxuFcs

- - -

[NumPy with: differentiate, vectorize, JIT to GPU/TPU ...](https://github.com/google/jax#what-is-jax)

JAX uses XLA to compile and run your NumPy programs on GPUs and TPUs. Compilation happens under the hood by default, with library calls getting just-in-time compiled and executed.

JAX also lets you just-in-time compile your own Python functions into XLA-optimized kernels using a one-function API, `jit`. Compilation and autodiff can be composed arbitrarily, so you can express sophisticated algorithms and get `maximal performance` without leaving Python. You can even program multiple GPUs or TPU cores at once using `pmap`, and differentiate through the whole thing.

### Transformations

At its core, JAX is an extensible system for transforming numerical functions. Here are four transformations of primary interest: `grad`, `jit`, `vmap` (vectorized), and `pmap` (parallel).

```py
from jax import grad
import jax.numpy as jnp

def tanh(x):  # Define a function
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)  # Obtain its gradient function
print(grad_tanh(1.0))   # Evaluate it at x = 1.0
# prints 0.4199743
```
