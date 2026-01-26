"""
Binary Search - Search Range

Instead of searching an array, we search a range of numbers using a comparison
function that tells us if our guess is too big, too small, or correct.

This pattern appears in problems like:
- Guess Number Higher or Lower
- First Bad Version
- Koko Eating Bananas

Time: O(log n) where n is the size of the range
Space: O(1)
"""


def binary_search_range(low, high, is_correct_func):
    """
    Binary search on a range using a comparison function.
    
    Parameters:
    - low: start of range
    - high: end of range
    - is_correct_func: function that takes a number and returns:
        - 1 if the number is too big
        - -1 if the number is too small
        - 0 if the number is correct
    
    Returns the correct number, or -1 if not found.
    
    Key insight: The comparison function acts as a "black box" - we don't
    need to know how it works internally, just how to interpret its return values.
    """
    while low <= high:
        mid = low + (high - low) // 2
        result = is_correct_func(mid)
        
        if result > 0:
            # Guess is too big → search left half (smaller numbers)
            high = mid - 1
        elif result < 0:
            # Guess is too small → search right half (larger numbers)
            low = mid + 1
        else:
            # Found the correct number!
            return mid
    
    return -1  # Not found (shouldn't happen if range is valid)


# Example 1: Simple number guessing
def make_is_correct_simple(target):
    """
    Creates a comparison function for a simple target number.
    
    This is like the "Guess Number Higher or Lower" problem.
    """
    def is_correct(n):
        if n > target:
            return 1   # Too big
        elif n < target:
            return -1  # Too small
        else:
            return 0   # Correct
    return is_correct


# Example 2: First Bad Version pattern
def make_is_bad_version(first_bad):
    """
    Creates a comparison function for "First Bad Version" problem.
    
    All versions >= first_bad are bad, all versions < first_bad are good.
    We want to find the first bad version.
    """
    def is_bad(version):
        if version >= first_bad:
            return 0  # This version is bad (or first bad)
        else:
            return -1  # This version is good, need to search right
    return is_bad


# Example 3: Feasibility check pattern (like Koko Eating Bananas)
def make_is_feasible(piles, h):
    """
    Creates a comparison function for feasibility problems.
    
    Checks if eating speed 'k' is feasible (can finish in h hours).
    This pattern appears in "Koko Eating Bananas" type problems.
    """
    def is_feasible(k):
        hours_needed = sum((pile + k - 1) // k for pile in piles)
        if hours_needed <= h:
            return 0   # Feasible (or could be faster)
        else:
            return 1   # Too slow, need faster speed
    return is_feasible


def demo():
    """
    Demonstrate binary search on range with different examples.
    """
    print("=== Binary Search on Range Demo ===\n")
    
    # Example 1: Simple number guessing (target = 10)
    print("Example 1: Guess Number (target = 10)")
    print("Range: 1 to 100")
    is_correct = make_is_correct_simple(10)
    result1 = binary_search_range(1, 100, is_correct)
    print(f"Found: {result1}\n")
    
    # Example 2: First Bad Version (first bad = 4)
    print("Example 2: First Bad Version (first bad = 4)")
    print("Range: 1 to 10")
    print("Versions: [G, G, G, B, B, B, B, B, B, B]")
    is_bad = make_is_bad_version(4)
    result2 = binary_search_range(1, 10, is_bad)
    print(f"First bad version: {result2}\n")
    
    # Example 3: Feasibility check
    print("Example 3: Koko Eating Bananas")
    print("Piles: [3, 6, 7, 11], Hours: 8")
    piles = [3, 6, 7, 11]
    is_feasible = make_is_feasible(piles, 8)
    # Find minimum feasible speed (search for first feasible)
    result3 = binary_search_range(1, max(piles), is_feasible)
    print(f"Minimum feasible speed: {result3}\n")
    
    print("=== Key Insights ===")
    print("1. Search a range instead of an array")
    print("2. Use a comparison function as a 'black box'")
    print("3. Function returns: 1 (too big), -1 (too small), 0 (correct)")
    print("4. Adjust search space based on function return value")
    print("5. Still O(log n) time complexity")
    print("\n=== Common Patterns ===")
    print("• Guess Number: Compare guess to secret")
    print("• First Bad Version: Find first version where condition is true")
    print("• Feasibility: Find minimum/maximum value that satisfies condition")
    print("\n⚠️  Important: Comparison function must be monotonic!")


if __name__ == "__main__":
    demo()
