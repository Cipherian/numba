from numba import njit, jit, types, typed
from numba.experimental import jitclass
from numba import prange
import time
import numpy as np
import timeit
import random

"""
https://numba.pydata.org/
Numba is an open source JIT compiler that translates python and numpy code into fast machine code

Numba translates Python functions to optimized machine code at runtime using the industry-standard LLVM compiler
library. Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN.
 
Numba is designed to be used with NumPy arrays and functions. Numba generates specialized code for different array data 
types and layouts to optimize performance. Special decorators can create universal functions that broadcast over NumPy arrays 
just like NumPy functions do.
 

The Numba @jit decorator fundamentally operates in two compilation modes, nopython mode and object mode. 
The behaviour of the nopython compilation mode is to essentially compile the decorated function so that it will run entirely 
without the involvement of the Python interpreter. This is the recommended and best-practice way to use the Numba jit decorator 
as it leads to the best performance.

Assuming Numba can operate in nopython mode, or at least compile some loops, it will target compilation to your specific CPU. 
Speed up varies depending on application but can be one to two orders of magnitude. Numba has a performance guide that covers 
common options for gaining extra performance.

Supported constructs:

conditional branch: if .. elif .. else

loops: while, for .. in, break, continue

basic generator: yield

assertion: assert

Partially supported constructs:

exceptions: try .. except, raise, else and finally (See details in this section)

context manager
list comprehension

Unsupported constructs:

async features: async with, async for and async def

class definition: class (except for @jitclass)

set, dict and generator comprehensions

generator delegation: yield from
 
Source: https://numba.readthedocs.io/en/stable/user/5minguide.html 

Causes segmentation fault
@njit(cache=True)
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
"""
@njit(cache=False)
def fibonacci_no_cache(n):
    if n < 2:
        return n
    else:
        a, b = 0, 1
        for _ in range(2, n+1):
            a, b = b, a + b
        return b

@njit(cache=True)
def fibonacci(n):
    if n < 2:
        return n
    else:
        a, b = 0, 1
        for _ in range(2, n+1):
            a, b = b, a + b
        return b



def no_njit_fibonacci(n):
    if n < 2:
        return n
    a, b = 0, 1
    for i in range(2, n+1):
        a, b = b, a + b
    return b


def time_fibonacci_sequences(n):
    with_njit_fibonacci = timeit.timeit(lambda: fibonacci(n), number=1)
    no_cache_fibonacci_time = timeit.timeit(lambda: fibonacci_no_cache(n), number=1)
    no_njit_fibonacci_time = timeit.timeit(lambda: no_njit_fibonacci(n), number=1)

    print(f"Fibonacci with njit took {with_njit_fibonacci:.6f} seconds")
    print(f"Fibonacci without cache took {no_cache_fibonacci_time:.6f} seconds")
    print(f"Fibonacci without njit took {no_njit_fibonacci_time:.6f} seconds")
    print(f"The function with cache was {no_cache_fibonacci_time / with_njit_fibonacci:.5f} times faster than without cache.")
    print(f"The function with njit was {no_njit_fibonacci_time / with_njit_fibonacci:.5f} times faster than without njit.")


def compare_performance(num:int) -> str:
    def python_loop(num:int):
        for i in range(num):
            i += 2**5

    start = time.time()
    python_loop(num)
    end = time.time()
    print(f'Python Time: {end - start:.5f}')
    python_time = end - start

    @jit(nopython=True)
    def jit_for_loop(num:int):
        for i in range(num):
           i += 2**5

    start = time.time()
    jit_for_loop(num)
    end = time.time()
    print(f'JIT Time: {end - start:.5f}')
    jit_time = end - start
    speedup = python_time / jit_time
    return f"The loop with the JIT compiler was {speedup:.5f} times faster than the plain Python loop."


def compare_performance_two():
    def np_array_sum(a:list[int]) -> int:
        return np.sum(a)

    start = time.time()
    array = np.arange(100000)
    result = np_array_sum(array)
    end = time.time()
    print(f"Without JIT compiler: {end - start:.6f} seconds")

    @jit(nopython=True)
    def jit_np_array_sum(a):
        return np.sum(a)

    start = time.time()
    array = np.arange(100000)
    result = jit_np_array_sum(array)
    end = time.time()
    print(f"With JIT compiler: {end - start:.6f} seconds")

def compare_sum_performance(arr:np.ndarray):
    def python_sum(arr:np.ndarray):
        result = 0
        for i in arr:
            result += (i**2 + np.sin(i) + np.cos(i))

    start = time.time()
    python_sum(arr)
    end = time.time()
    print(f'Python: {end - start:.6f}')
    python_time = end - start

    @jit(nopython=True)
    def jit_sum(arr:np.ndarray):
        result = 0
        for i in arr:
            result += (i**2 + np.sin(i) + np.cos(i))

    start = time.time()
    jit_sum(arr)
    end = time.time()
    print(f'JIT: {end - start:.6f}')
    jit_time = end - start
    speedup = python_time / jit_time
    print( f"The loop with the JIT compiler was {speedup:.2f} times faster than the plain Python loop.")


def sum_of_nested_arrays(arrays) -> int:
    result = 0
    for array in arrays:
        for i in numba.prange(len(array)):
            result += array[i]
    return result


def matrix_speed_test(A, B, C):

    @njit(parallel=True)
    def matrix_mult_numba(A, B, C):
        m, n = A.shape
        n, p = B.shape
        for i in prange(m):
            for j in prange(p):
                for k in prange(n):
                    C[i, j] += A[i, k] * B[k, j]

    # Function to perform matrix multiplication using regular Python
    def matrix_mult_python(A, B, C):
        m, n = A.shape
        n, p = B.shape
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]


    # Time Numba matrix multiplication

    start = time.time()
    matrix_mult_numba(A, B, C)
    end = time.time()
    jit_time = end - start

    print(f"Matrix multiplication using Numba took {end - start:.5f} seconds")


    # Time regular Python matrix multiplication
    start_time = time.time()
    matrix_mult_python(A, B, C)
    end_time = time.time()
    python_time = end_time - start_time
    print(f"Matrix multiplication using Numba took {end_time - start_time:.5f} seconds")
    speedup = python_time / jit_time
    print( f"The loop with the JIT compiler was {speedup:.2f} times faster")



@jit(forceobj=True)
def wrong_use(dictionary:dict, key:any):
    """
    https://numba.readthedocs.io/en/stable/glossary.html#term-object-mode
    """
    return dictionary[key]










if __name__ == "__main__":
    # time_fibonacci_sequences(1000000)
    # print(compare_performance(100000000))
    # compare_performance_two()
    # compare_sum_performance(np.arange(1000000))
    matrix_speed_test(
        A = np.random.rand(500, 500),
        B = np.random.rand(500, 500),
        C = np.random.rand(500, 500)
    )
    # print(wrong_use({'a': 1, 'b': 2, 'c': 3},'a'))




