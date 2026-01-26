"""
Fibonacci Sequence Using Recursion (Two-Branch Recursion)

Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
Each number is the sum of the two previous numbers.

Formula: fib(n) = fib(n-1) + fib(n-2)
Base cases: fib(0) = 0, fib(1) = 1

This is two-branch recursion because each call creates TWO recursive calls.
Without optimization, this is O(2^n) - exponential time!

Time (naive): O(2^n) - exponential growth
Time (memoized): O(n) - each value computed once
Space: O(n) - call stack depth
"""


def fibonacci_naive(n):
    """
    Naive recursive implementation - VERY SLOW for large n.
    
    Why slow? We recalculate the same values many times.
    For example, fib(5) calculates fib(3) multiple times.
    
    Time: O(2^n) - exponential!
    """
    # Base cases
    if n <= 1:
        return n
    
    # Two-branch recursion: call function twice
    # This creates a binary tree of function calls
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_memo(n, memo=None):
    """
    Optimized version using memoization (caching).
    
    Memoization stores results we've already computed so we don't
    recalculate them. This reduces time from O(2^n) to O(n).
    
    Time: O(n) - each value computed once
    Space: O(n) - memo dictionary + call stack
    """
    if memo is None:
        memo = {}
    
    # Base cases
    if n <= 1:
        return n
    
    # Check if we've already computed this value
    if n in memo:
        return memo[n]
    
    # Compute and store result
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


def fibonacci_iterative(n):
    """
    Iterative version - most efficient.
    
    We only need the last two values to compute the next one.
    This uses O(1) space instead of O(n).
    
    Time: O(n)
    Space: O(1)
    """
    if n <= 1:
        return n
    
    # Start with first two Fibonacci numbers
    prev2 = 0  # fib(0)
    prev1 = 1  # fib(1)
    
    # Build up from bottom
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


def demo():
    """
    Demonstrate Fibonacci calculation with different approaches.
    """
    print("=== Fibonacci Demo ===\n")
    
    n = 10
    print(f"Computing fib({n}):")
    
    # Naive (slow for large n)
    print(f"  Naive recursive: {fibonacci_naive(n)}")
    
    # Memoized (fast)
    print(f"  Memoized: {fibonacci_memo(n)}")
    
    # Iterative (fastest, most space-efficient)
    print(f"  Iterative: {fibonacci_iterative(n)}")
    
    print("\n=== Why Memoization Matters ===")
    print("Without memoization, fib(5) recalculates:")
    print("  fib(3) multiple times")
    print("  fib(2) many times")
    print("  fib(1) and fib(0) many, many times")
    print("\nWith memoization, each value is computed once and reused!")


if __name__ == "__main__":
    demo()
