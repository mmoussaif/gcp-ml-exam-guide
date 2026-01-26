"""
Dynamic Array Implementation

Dynamic arrays can grow as needed. When they run out of space, they
double in size. This makes insertion O(1) amortized (average) time.

Why double? It ensures we don't resize too often, keeping average
insertion time constant.
"""


class DynamicArray:
    def __init__(self, capacity=2):
        """
        Initialize with a small capacity. We'll grow as needed.
        """
        self.capacity = capacity  # Maximum size before resizing
        self.length = 0  # Current number of elements
        self.arr = [0] * capacity  # The actual array

    def push_back(self, value):
        """
        Add element to the end. O(1) amortized time.
        
        Amortized means: if we add n elements, total time is O(n),
        so average per element is O(1).
        """
        # Check if we need more space
        if self.length == self.capacity:
            self._resize()  # Double the capacity
        
        # Add the new element
        self.arr[self.length] = value
        self.length += 1

    def _resize(self):
        """
        Double the array size. O(n) operation, but happens rarely.
        
        We create a new array twice the size, copy all elements over,
        then replace the old array. The old array gets garbage collected.
        """
        # Double the capacity
        self.capacity *= 2
        
        # Create new array with double the size
        new_arr = [0] * self.capacity
        
        # Copy all existing elements to the new array
        for i in range(self.length):
            new_arr[i] = self.arr[i]
        
        # Replace old array with new one
        self.arr = new_arr

    def to_list(self):
        """
        Return only the actual elements (not the empty slots).
        """
        return self.arr[: self.length]


def demo():
    da = DynamicArray()
    for value in [1, 2, 3, 4, 5]:
        da.push_back(value)
        print("Added", value, "->", da.to_list(), "cap:", da.capacity)


if __name__ == "__main__":
    demo()
