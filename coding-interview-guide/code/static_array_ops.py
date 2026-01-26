"""
Static Array Operations

Static arrays have a fixed size. These operations show how to work with
arrays when you need to insert/delete elements while maintaining order.
"""


def remove_end(arr, length):
    """
    Remove element from the end of the array.
    This is O(1) - we just overwrite the last element.
    """
    if length > 0:
        # Overwrite last element with 0 (marks it as empty)
        arr[length - 1] = 0
        length -= 1
    return length


def remove_middle(arr, i, length):
    """
    Remove element at index i from the middle of the array.
    This is O(n) - we must shift all elements after i to the left.
    
    Why shift? To keep elements contiguous (no gaps).
    """
    # Shift all elements after i one position to the left
    # Start from i+1 and move each element left
    for idx in range(i + 1, length):
        arr[idx - 1] = arr[idx]  # Move element left
    
    # Mark the last position as empty
    arr[length - 1] = 0
    return length - 1


def insert_end(arr, value, length, capacity):
    """
    Insert element at the end of the array.
    This is O(1) - we just write to the next empty slot.
    """
    if length < capacity:
        # Write to the next available position
        arr[length] = value
        length += 1
    return length


def insert_middle(arr, i, value, length, capacity):
    """
    Insert element at index i in the middle of the array.
    This is O(n) - we must shift all elements from i onward to the right.
    
    Why shift right? To make room for the new element without overwriting.
    """
    if length >= capacity:
        return length  # Array is full
    
    # Shift elements to the right, starting from the end
    # We go backwards to avoid overwriting values we need
    for idx in range(length - 1, i - 1, -1):
        arr[idx + 1] = arr[idx]  # Move element right
    
    # Now insert the new value at position i
    arr[i] = value
    return length + 1


def demo():
    capacity = 6
    arr = [0] * capacity
    length = 0

    for value in [4, 5, 6]:
        length = insert_end(arr, value, length, capacity)
    print("After inserts:", arr, "length:", length)

    length = insert_middle(arr, 1, 8, length, capacity)
    print("After insert middle:", arr, "length:", length)

    length = remove_middle(arr, 0, length)
    print("After remove middle:", arr, "length:", length)

    length = remove_end(arr, length)
    print("After remove end:", arr, "length:", length)


if __name__ == "__main__":
    demo()
