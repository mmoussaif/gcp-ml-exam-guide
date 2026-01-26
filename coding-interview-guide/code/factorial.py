"""
Factorial Using Recursion (One-Branch Recursion)

Recursion is when a function calls itself. For factorial, we break the problem
into smaller sub-problems: n! = n * (n-1)!

Key components:
- Base case: stops recursion (n <= 1 returns 1)
- Recursive case: calls itself with smaller input (n * factorial(n-1))

Time: O(n) - we make n function calls
Space: O(n) - call stack depth is n
"""


def factorial(n):
    """
    Calculate n! using recursion.
    
    Example: 5! = 5 * 4 * 3 * 2 * 1 = 120
    
    How it works:
    1. Base case: if n <= 1, return 1 (0! = 1, 1! = 1)
    2. Recursive case: return n * factorial(n-1)
    
    The call stack builds up, then unwinds:
    factorial(5) calls factorial(4) calls factorial(3) ...
    Then results multiply back: 1 * 2 * 3 * 4 * 5 = 120
    """
    # Base case: smallest problem we can solve directly
    # Both 0! and 1! equal 1
    if n <= 1:
        return 1
    
    # Recursive case: break problem into smaller sub-problem
    # n! = n * (n-1)!
    # We trust that factorial(n-1) will give us the right answer
    return n * factorial(n - 1)


def factorial_iterative(n):
    """
    Iterative version for comparison.
    
    Sometimes iteration is simpler and more space-efficient.
    This uses O(1) space instead of O(n) space.
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def demo():
    """
    Demonstrate factorial calculation.
    """
    print("=== Factorial Demo ===\n")
    
    for n in [0, 1, 5, 7]:
        result = factorial(n)
        print(f"{n}! = {result}")
    
    print("\n=== Call Stack Visualization ===")
    print("factorial(5) calls:")
    print("  factorial(4) calls:")
    print("    factorial(3) calls:")
    print("      factorial(2) calls:")
    print("        factorial(1) returns 1")
    print("      factorial(2) returns 2 * 1 = 2")
    print("    factorial(3) returns 3 * 2 = 6")
    print("  factorial(4) returns 4 * 6 = 24")
    print("factorial(5) returns 5 * 24 = 120")


if __name__ == "__main__":
    demo()
