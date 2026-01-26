"""
Bucket Sort (Counting Sort)

Bucket sort works when values are in a limited, known range.
It counts how many times each value appears, then overwrites
the array in sorted order based on those counts.

IMPORTANT: Even though there's a nested loop, this is O(n) NOT O(n²)!
The inner loop runs exactly as many times as elements exist (sum of all counts = n).

Time: O(n) - nested loops don't always mean O(n²)!
Space: O(k) - where k is number of distinct values
Stability: Not stable (overwrites original array)
"""


def bucket_sort(arr):
    """
    Sort array using bucket sort (counting sort).
    
    This version assumes array only contains 0, 1, or 2 (Sort Colors problem).
    
    Algorithm:
    1. Count frequency of each value (0, 1, 2)
    2. Overwrite array in sorted order based on counts
    
    Time Complexity Analysis:
    - First loop: O(n) - iterate through array once
    - Outer loop: O(k) where k = number of distinct values (3 in this case)
    - Inner loop: runs counts[val] times for each value
    - Total inner loop iterations: counts[0] + counts[1] + counts[2] = n
    - Therefore: O(n) + O(n) = O(n), NOT O(n²)!
    
    Key insight: Nested loops don't always mean O(n²). You must analyze
    how many times the inner loop actually executes.
    """
    # Step 1: Count frequency of each value
    # Create buckets (one for each possible value: 0, 1, 2)
    counts = [0, 0, 0]
    
    # Count how many times each value appears
    # This is O(n) - single pass through array
    for n in arr:
        counts[n] += 1
    
    # Step 2: Overwrite array in sorted order
    i = 0  # Pointer tracking next insertion position in arr
    
    # Outer loop: iterate over possible values (0, 1, 2)
    for val in range(len(counts)):
        # Inner loop: write value 'val' as many times as it appeared
        # This runs counts[val] times
        for j in range(counts[val]):
            arr[i] = val
            i += 1
    
    # Why is this O(n)? Because:
    # - Inner loop runs counts[0] + counts[1] + counts[2] times total
    # - counts[0] + counts[1] + counts[2] = n (total elements)
    # - So inner loop runs exactly n times total
    
    return arr


def bucket_sort_general(arr, max_val):
    """
    General bucket sort for any range 0 to max_val.
    
    Same O(n) time complexity - nested loops still sum to n iterations.
    """
    # Count frequencies
    counts = [0] * (max_val + 1)
    
    for val in arr:
        counts[val] += 1
    
    # Overwrite in sorted order
    i = 0
    for val in range(len(counts)):
        for _ in range(counts[val]):
            arr[i] = val
            i += 1
    
    return arr


def bucket_sort_optimized(arr, max_val):
    """
    Optimized version using single pass.
    
    Instead of nested loops, we can use a single loop
    with a running index for each value.
    """
    counts = [0] * (max_val + 1)
    
    # Count frequencies
    for val in arr:
        counts[val] += 1
    
    # Overwrite array
    i = 0
    for val in range(max_val + 1):
        count = counts[val]
        # Fill count consecutive positions with val
        arr[i:i + count] = [val] * count
        i += count
    
    return arr


def demo():
    """
    Demonstrate bucket sort.
    """
    print("=== Bucket Sort Demo ===\n")
    
    # Example: Sort Colors (0, 1, 2 only) - Classic LeetCode problem
    arr = [2, 0, 2, 1, 1, 0]
    print(f"Original: {arr}")
    sorted_arr = bucket_sort(arr.copy())
    print(f"Sorted:   {sorted_arr}\n")
    
    print("=== How It Works ===")
    print("Array: [2, 0, 2, 1, 1, 0]")
    print("\nStep 1: Count frequencies")
    print("  counts[0] = 2  (appears 2 times)")
    print("  counts[1] = 1  (appears 1 time)")
    print("  counts[2] = 3  (appears 3 times)")
    print("\nStep 2: Overwrite array")
    print("  Write 0 twice:  [0, 0, ...]")
    print("  Write 1 once:   [0, 0, 1, ...]")
    print("  Write 2 three times: [0, 0, 1, 2, 2, 2]")
    
    print("\n=== Time Complexity Analysis ===")
    print("You might see nested loops and think O(n²), but it's O(n)!")
    print("\nWhy?")
    print("  - Outer loop: 3 iterations (for values 0, 1, 2)")
    print("  - Inner loop: runs counts[val] times for each value")
    print("  - Total inner iterations: counts[0] + counts[1] + counts[2]")
    print("  - This equals n (total elements in array)")
    print("  - So: O(n) + O(n) = O(n)")
    print("\nKey lesson: Analyze how many times inner loop runs, not just")
    print("that it's nested!")
    
    print("\n=== When to Use Bucket Sort ===")
    print("✓ Values are in a known, limited range (e.g., 0-2, 0-255)")
    print("✓ Need O(n) time complexity")
    print("✓ Don't need stability")
    print("\n✗ Only works with limited value range")
    print("✗ Requires knowing max value upfront")
    print("✗ Not stable (doesn't preserve order of equal elements)")
    print("\nFor general sorting, merge sort or quick sort are safer choices.")


if __name__ == "__main__":
    demo()
