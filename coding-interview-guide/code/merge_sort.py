"""
Merge Sort

Merge sort uses divide-and-conquer:
1. Split array in half
2. Recursively sort each half
3. Merge the sorted halves together

The merge step uses two pointers to combine two sorted arrays efficiently.

Time: O(n log n) - always (best, average, worst)
Space: O(n) - temporary arrays for merging
Stability: Stable (preserves relative order of equal elements)
"""


def merge_sort(arr, s=0, e=None):
    """
    Sort array using merge sort.
    
    Parameters:
    - arr: array to sort
    - s: start index (default 0)
    - e: end index (default len(arr) - 1)
    
    Algorithm:
    1. Base case: array of size 1 is already sorted
    2. Find middle index
    3. Recursively sort left half
    4. Recursively sort right half
    5. Merge the two sorted halves
    """
    if e is None:
        e = len(arr) - 1
    
    # Base case: array of size 1 or less is already sorted
    if e - s + 1 <= 1:
        return arr
    
    # Find middle index
    m = (s + e) // 2
    
    # Recursively sort left and right halves
    merge_sort(arr, s, m)      # Sort left half [s...m]
    merge_sort(arr, m + 1, e)  # Sort right half [m+1...e]
    
    # Merge the two sorted halves
    merge(arr, s, m, e)
    
    return arr


def merge(arr, s, m, e):
    """
    Merge two sorted subarrays: [s...m] and [m+1...e]
    
    Uses two-pointer technique:
    - Compare elements from both halves
    - Place smaller element in result
    - Move pointer forward
    - Copy remaining elements
    
    Time: O(n) where n is total elements to merge
    """
    # Copy left and right halves to temporary arrays
    # This allows us to overwrite original array safely
    left = arr[s:m + 1]   # Left half: [s...m]
    right = arr[m + 1:e + 1]  # Right half: [m+1...e]
    
    # Two pointers for left and right arrays
    i = 0  # Pointer for left array
    j = 0  # Pointer for right array
    k = s  # Pointer for original array (where to write)
    
    # Merge: compare and place smaller element
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            # Use <= to maintain stability (equal elements keep order)
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    
    # Copy remaining elements from left half (if any)
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    
    # Copy remaining elements from right half (if any)
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1


def demo():
    """
    Demonstrate merge sort.
    """
    print("=== Merge Sort Demo ===\n")
    
    arr = [38, 27, 43, 3, 9, 82, 10]
    print(f"Original: {arr}")
    
    sorted_arr = merge_sort(arr.copy())
    print(f"Sorted:   {sorted_arr}\n")
    
    print("=== How It Works ===")
    print("1. Split: [38,27,43,3,9,82,10]")
    print("   Left:  [38,27,43]")
    print("   Right: [3,9,82,10]")
    print("\n2. Recursively sort each half")
    print("   Left sorted:  [27,38,43]")
    print("   Right sorted: [3,9,10,82]")
    print("\n3. Merge using two pointers:")
    print("   Compare 27 vs 3 → place 3")
    print("   Compare 27 vs 9 → place 9")
    print("   Compare 27 vs 10 → place 10")
    print("   Compare 27 vs 82 → place 27")
    print("   ... continue until done")
    print("\n=== Why Merge Sort? ===")
    print("✓ Always O(n log n) - predictable performance")
    print("✓ Stable - preserves order of equal elements")
    print("✓ Good for linked lists")
    print("✓ Parallelizable")
    print("\n✗ Requires O(n) extra space")
    print("✗ Not in-place")


if __name__ == "__main__":
    demo()
