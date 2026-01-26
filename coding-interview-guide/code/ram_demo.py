"""
RAM (Random Access Memory) Demo

This demonstrates how arrays are stored in RAM. Each element is stored
at a specific memory address, and we can access any element instantly
by its index. This is why array access is O(1) - the computer knows
exactly where to look in memory.
"""


def access_demo():
    # Create an array with 3 elements
    # In RAM, these are stored contiguously (one after another)
    arr = [1, 3, 5]
    
    print("Array:", arr)
    
    # Accessing by index is instant - O(1) time
    # The computer calculates: start_address + (index * element_size)
    print("Index 0:", arr[0])  # Gets the first element
    print("Index 2:", arr[2])  # Gets the third element (0-indexed)


if __name__ == "__main__":
    access_demo()
