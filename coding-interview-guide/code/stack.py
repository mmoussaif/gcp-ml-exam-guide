"""
Stack Implementation

A stack is LIFO (Last In First Out) - like a stack of plates.
You can only add/remove from the top. All operations are O(1).

Common uses:
- Undo/redo functionality
- Expression evaluation
- Backtracking algorithms
"""


class Stack:
    def __init__(self):
        """
        Initialize empty stack using a list (dynamic array).
        """
        self.items = []

    def push(self, value):
        """
        Add element to the top of the stack. O(1) time.
        """
        self.items.append(value)

    def pop(self):
        """
        Remove and return the top element. O(1) time.
        Returns None if stack is empty.
        """
        if not self.items:
            return None
        return self.items.pop()  # Removes and returns last element

    def peek(self):
        """
        Look at the top element without removing it. O(1) time.
        Returns None if stack is empty.
        """
        if not self.items:
            return None
        return self.items[-1]  # Returns last element without removing


def demo():
    stack = Stack()
    for value in [1, 2, 3]:
        stack.push(value)
    print("Top:", stack.peek())
    print("Pop:", stack.pop())
    print("Pop:", stack.pop())


if __name__ == "__main__":
    demo()
