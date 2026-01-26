"""
Insertion Sort

Insertion sort builds a sorted portion from left to right. For each element,
it inserts it into the correct position in the sorted portion by shifting
larger elements to the right.

Think of it like sorting cards in your hand - you pick up each card and
insert it where it belongs in your sorted hand.

Time: O(n²) worst/average case, O(n) best case (already sorted)
Space: O(1) - sorts in-place
Stability: Stable (preserves relative order of equal elements)
"""


def insertion_sort(arr):
    """
    Sort array using insertion sort.
    
    Algorithm:
    1. Start from index 1 (first element is "sorted")
    2. For each element, compare with elements to its left
    3. Shift larger elements right until correct position found
    4. Insert current element in correct position
    
    Best case: Array already sorted - O(n) time
    Worst case: Array reversed - O(n²) time
    """
    # Start from index 1 - first element is already "sorted"
    for i in range(1, len(arr)):
        # Current element to insert
        current = arr[i]
        
        # j points to the last element in sorted portion
        j = i - 1
        
        # Shift elements right until we find correct position
        # Stop when we find element smaller than current, or reach start
        while j >= 0 and arr[j] > current:
            arr[j + 1] = arr[j]  # Shift element right
            j -= 1
        
        # Insert current element in correct position
        arr[j + 1] = current
    
    return arr


def insertion_sort_verbose(arr):
    """
    Same algorithm with print statements to visualize the process.
    """
    print(f"Initial array: {arr}")
    
    for i in range(1, len(arr)):
        current = arr[i]
        j = i - 1
        
        print(f"\nInserting {current} at position {i}")
        print(f"  Sorted portion: {arr[:i]}")
        
        while j >= 0 and arr[j] > current:
            arr[j + 1] = arr[j]
            j -= 1
            print(f"  Shifted {arr[j+1]} right")
        
        arr[j + 1] = current
        print(f"  Final: {arr[:i+1]}")
    
    return arr


def demo():
    """
    Demonstrate insertion sort.
    """
    print("=== Insertion Sort Demo ===\n")
    
    # Example 1: Random array
    arr1 = [5, 2, 4, 6, 1, 3]
    print(f"Original: {arr1}")
    sorted1 = insertion_sort(arr1.copy())
    print(f"Sorted:   {sorted1}\n")
    
    # Example 2: Already sorted (best case - O(n))
    arr2 = [1, 2, 3, 4, 5]
    print(f"Already sorted: {arr2}")
    sorted2 = insertion_sort(arr2.copy())
    print(f"Result: {sorted2}\n")
    
    # Example 3: Reversed (worst case - O(n²))
    arr3 = [5, 4, 3, 2, 1]
    print(f"Reversed: {arr3}")
    sorted3 = insertion_sort(arr3.copy())
    print(f"Result: {sorted3}\n")
    
    print("=== Why Insertion Sort? ===")
    print("✓ Simple to understand and implement")
    print("✓ Efficient for small arrays")
    print("✓ O(n) for nearly-sorted arrays")
    print("✓ Stable (preserves order of equal elements)")
    print("✓ In-place (O(1) extra space)")
    print("\n✗ O(n²) for large random arrays")
    print("✗ Not as fast as merge/quick sort for large datasets")


if __name__ == "__main__":
    demo()
