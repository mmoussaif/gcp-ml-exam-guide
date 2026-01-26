"""
Binary Search

Binary search efficiently finds elements in a sorted array by repeatedly dividing
the search space in half. Like searching a dictionary - open to the middle, decide
if target is left or right, repeat.

Key requirements:
- Array must be sorted
- We eliminate half the search space each iteration

Time: O(log n) - eliminate half each step
Space: O(1) - only using pointers
"""


def binary_search(arr, target):
    """
    Search for target in sorted array using binary search.
    
    Returns index of target if found, -1 otherwise.
    
    Algorithm:
    1. Set left (L) and right (R) pointers to array bounds
    2. While L <= R:
       a. Calculate mid = L + (R - L) // 2
       b. Compare arr[mid] to target
       c. If target > arr[mid]: search right (L = mid + 1)
       d. If target < arr[mid]: search left (R = mid - 1)
       e. If target == arr[mid]: found! Return mid
    3. If loop ends, target not found (return -1)
    
    Why L + (R - L) // 2 instead of (L + R) // 2?
    - Prevents integer overflow when L and R are very large
    - Example: L = 2^31 - 1, R = 2^31 - 1
      - (L + R) // 2 could overflow
      - L + (R - L) // 2 = L + 0 = L (safe!)
    """
    L, R = 0, len(arr) - 1
    
    while L <= R:
        # Calculate middle index (prevents overflow)
        mid = L + (R - L) // 2
        
        if target > arr[mid]:
            # Target is greater → search right half
            L = mid + 1
        elif target < arr[mid]:
            # Target is smaller → search left half
            R = mid - 1
        else:
            # Found target!
            return mid
    
    # Target not found
    return -1


def binary_search_recursive(arr, target, L=0, R=None):
    """
    Recursive version of binary search.
    
    Same logic, but uses recursion instead of iteration.
    Space complexity becomes O(log n) due to call stack.
    """
    if R is None:
        R = len(arr) - 1
    
    # Base case: search space exhausted
    if L > R:
        return -1
    
    mid = L + (R - L) // 2
    
    if target > arr[mid]:
        return binary_search_recursive(arr, target, mid + 1, R)
    elif target < arr[mid]:
        return binary_search_recursive(arr, target, L, mid - 1)
    else:
        return mid


def demo():
    """
    Demonstrate binary search with examples.
    """
    print("=== Binary Search Demo ===\n")
    
    # Example 1: Target found
    arr1 = [1, 2, 3, 4, 5, 6, 7, 8]
    target1 = 5
    print(f"Array: {arr1}")
    print(f"Searching for: {target1}")
    result1 = binary_search(arr1, target1)
    print(f"Result: Found at index {result1}\n")
    
    # Example 2: Target not found
    arr2 = [1, 2, 3, 4, 5, 6, 7, 8]
    target2 = 9
    print(f"Array: {arr2}")
    print(f"Searching for: {target2}")
    result2 = binary_search(arr2, target2)
    print(f"Result: {result2} (not found)\n")
    
    # Example 3: Single element
    arr3 = [5]
    target3 = 5
    print(f"Array: {arr3}")
    print(f"Searching for: {target3}")
    result3 = binary_search(arr3, target3)
    print(f"Result: Found at index {result3}\n")
    
    print("=== How It Works ===")
    print("1. Start with entire array as search space")
    print("2. Calculate middle index")
    print("3. Compare middle element to target")
    print("4. Eliminate half the search space")
    print("5. Repeat until found or search space exhausted")
    print("\nEach step eliminates half → O(log n) time!")
    print("\n=== Why O(log n)? ===")
    print("n elements → n/2 → n/4 → n/8 → ... → 1")
    print("How many times can we divide n by 2?")
    print("Answer: log₂(n) times")
    print("Therefore: O(log n) time complexity")


if __name__ == "__main__":
    demo()
