"""
Quick Sort

Quick sort uses divide-and-conquer with partitioning:
1. Pick a pivot element
2. Partition: elements < pivot go left, >= pivot go right
3. Recursively sort left and right sides

The partition step does the sorting work - no merge needed!

Time: O(n log n) average, O(n²) worst case (bad pivot choice)
Space: O(log n) - recursion stack depth
Stability: Not stable (swaps non-adjacent elements)
"""


def quick_sort(arr, s=0, e=None):
    """
    Sort array using quick sort.
    
    Parameters:
    - arr: array to sort
    - s: start index
    - e: end index
    
    Algorithm:
    1. Base case: array of size 1 is sorted
    2. Pick pivot (we use last element)
    3. Partition around pivot
    4. Recursively sort left and right sides
    """
    if e is None:
        e = len(arr) - 1
    
    # Base case: array of size 1 or less is sorted
    if e - s + 1 <= 1:
        return arr
    
    # Partition and get pivot's final position
    pivot_idx = partition(arr, s, e)
    
    # Recursively sort left side (elements < pivot)
    quick_sort(arr, s, pivot_idx - 1)
    
    # Recursively sort right side (elements >= pivot)
    quick_sort(arr, pivot_idx + 1, e)
    
    return arr


def partition(arr, s, e):
    """
    Partition array around pivot.
    
    Strategy:
    - Use last element as pivot
    - Move all elements < pivot to left
    - Place pivot in correct position
    - Return pivot's final index
    
    Returns: final index of pivot
    """
    # Choose pivot (last element)
    pivot = arr[e]
    
    # 'left' tracks where to place next element < pivot
    left = s
    
    # Iterate through array (excluding pivot)
    for i in range(s, e):
        if arr[i] < pivot:
            # Swap element to left side
            arr[left], arr[i] = arr[i], arr[left]
            left += 1
    
    # Place pivot in correct position (between left and right sides)
    arr[left], arr[e] = arr[e], arr[left]
    
    return left  # Return pivot's final position


def quick_sort_randomized(arr, s=0, e=None):
    """
    Randomized quick sort - better average performance.
    
    Randomly chooses pivot to avoid worst case O(n²).
    This makes worst case O(n log n) with high probability.
    """
    import random
    
    if e is None:
        e = len(arr) - 1
    
    if e - s + 1 <= 1:
        return arr
    
    # Randomly choose pivot and swap with last element
    pivot_idx = random.randint(s, e)
    arr[pivot_idx], arr[e] = arr[e], arr[pivot_idx]
    
    # Now partition as usual
    pivot_idx = partition(arr, s, e)
    quick_sort_randomized(arr, s, pivot_idx - 1)
    quick_sort_randomized(arr, pivot_idx + 1, e)
    
    return arr


def demo():
    """
    Demonstrate quick sort.
    """
    print("=== Quick Sort Demo ===\n")
    
    arr = [38, 27, 43, 3, 9, 82, 10]
    print(f"Original: {arr}")
    
    sorted_arr = quick_sort(arr.copy())
    print(f"Sorted:   {sorted_arr}\n")
    
    print("=== How Partition Works ===")
    print("Array: [38, 27, 43, 3, 9, 82, 10]")
    print("Pivot: 10 (last element)")
    print("\nPartition steps:")
    print("1. Compare 38 < 10? No → skip")
    print("2. Compare 27 < 10? No → skip")
    print("3. Compare 43 < 10? No → skip")
    print("4. Compare 3 < 10? Yes → swap to left")
    print("5. Compare 9 < 10? Yes → swap to left")
    print("6. Compare 82 < 10? No → skip")
    print("\nAfter partition: [3, 9, 43, 38, 27, 82, 10]")
    print("Left side: [3, 9] (all < 10)")
    print("Right side: [43, 38, 27, 82] (all >= 10)")
    print("Pivot at index 2")
    print("\nThen recursively sort left and right sides")
    
    print("\n=== Why Quick Sort? ===")
    print("✓ Average O(n log n) - very fast in practice")
    print("✓ In-place - O(log n) extra space")
    print("✓ Cache-friendly")
    print("\n✗ Worst case O(n²) if pivot is always min/max")
    print("✗ Not stable")
    print("✗ Performance depends on pivot choice")


if __name__ == "__main__":
    demo()
