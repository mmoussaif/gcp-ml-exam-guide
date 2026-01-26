"""
Heap Properties

A heap is a specialized, tree-based data structure that implements a Priority Queue.
Unlike regular queues (FIFO), priority queues remove elements based on priority.

Two types:
- Min Heap: Smallest value at root (highest priority)
- Max Heap: Largest value at root (highest priority)

Heap Properties:
1. Structure Property: Complete binary tree (all levels filled except last, left to right)
2. Order Property: 
   - Min heap: All descendants ≥ ancestors
   - Max heap: All descendants ≤ ancestors

Implementation:
- Drawn as tree but implemented using arrays
- 1-based indexing (index 0 unused as sentinel)
- Formulas: leftChild = 2*i, rightChild = 2*i+1, parent = i//2

Time: O(1) to access root, O(1) to navigate with formulas
Space: O(n) for array storage
"""


class Heap:
    """
    Binary Heap implementation using array (1-based indexing).
    
    Structure:
    - Index 0: Unused (sentinel)
    - Index 1+: Heap elements in BFS order (level by level, left to right)
    
    Formulas (where i is node index):
    - leftChild(i) = 2 * i
    - rightChild(i) = 2 * i + 1
    - parent(i) = i // 2 (integer division)
    
    Why 1-based indexing?
    - If root at index 0: leftChild(0) = 0 (would point to itself!)
    - If root at index 1: leftChild(1) = 2, rightChild(1) = 3 ✓
    - Formulas work perfectly with index 1
    """
    def __init__(self):
        # Index 0 is unused (sentinel value)
        # This allows parent/child formulas to work correctly
        self.heap = [0]
    
    def left_child(self, i):
        """
        Get left child index of node at index i.
        
        Formula: leftChild = 2 * i
        """
        return 2 * i
    
    def right_child(self, i):
        """
        Get right child index of node at index i.
        
        Formula: rightChild = 2 * i + 1
        """
        return 2 * i + 1
    
    def parent(self, i):
        """
        Get parent index of node at index i.
        
        Formula: parent = i // 2 (integer division)
        """
        return i // 2
    
    def size(self):
        """Get number of elements in heap (excluding sentinel at index 0)."""
        return len(self.heap) - 1
    
    def is_empty(self):
        """Check if heap is empty."""
        return self.size() == 0
    
    def push(self, val):
        """
        Push value into heap and percolate up to maintain order property.
        
        Algorithm:
        1. Append value to end of array (maintains structure property - complete tree)
        2. Compare with parent using formula parent = i // 2
        3. If value < parent (min heap), swap with parent
        4. Repeat until parent is smaller or reach root (index 1)
        
        This is called "percolate up" or "bubble up" because the element moves up
        the tree until it finds its correct position.
        
        Time: O(log n) - height of tree (at most log n swaps)
        Space: O(1) - only swapping elements
        """
        self.heap.append(val)  # Add to end (next position in complete tree)
        i = len(self.heap) - 1  # Index of new element
        
        # Percolate up: swap with parent while violating order property
        # Stop when: reached root (i == 1) OR order property satisfied (val >= parent)
        while i > 1 and self.heap[i] < self.heap[i // 2]:
            # Swap with parent
            self.heap[i], self.heap[i // 2] = self.heap[i // 2], self.heap[i]
            i = i // 2  # Move to parent index
    
    def pop(self):
        """
        Pop and return root (minimum value for min-heap).
        
        Algorithm:
        1. If empty (only sentinel), return None
        2. If only one element, pop and return it
        3. Store root value (to return later)
        4. Replace root with last element (maintains structure property)
        5. Percolate down: swap with smaller child until order property satisfied
        
        This is called "percolate down" or "bubble down" because the element moves
        down the tree until it finds its correct position.
        
        Time: O(log n) - height of tree (at most log n swaps)
        Space: O(1) - only swapping elements
        """
        if len(self.heap) == 1:
            return None  # Empty heap (only sentinel at index 0)
        if len(self.heap) == 2:
            return self.heap.pop()  # Only one element, just remove it
        
        res = self.heap[1]  # Store root value (to return)
        # Move last element to root (maintains structure property - complete tree)
        self.heap[1] = self.heap.pop()
        i = 1
        
        # Percolate down: swap with smaller child while violating order property
        # Stop when: no left child OR order property satisfied
        while 2 * i < len(self.heap):  # While has left child (complete tree guarantees this)
            # Check if has right child and right child is smaller than left child
            if (2 * i + 1 < len(self.heap) and 
                self.heap[2 * i + 1] < self.heap[2 * i] and
                self.heap[i] > self.heap[2 * i + 1]):
                # Swap with right child (it's the smaller child)
                self.heap[i], self.heap[2 * i + 1] = self.heap[2 * i + 1], self.heap[i]
                i = 2 * i + 1
            elif self.heap[i] > self.heap[2 * i]:
                # Swap with left child (either no right child, or left is smaller)
                self.heap[i], self.heap[2 * i] = self.heap[2 * i], self.heap[i]
                i = 2 * i
            else:
                # Order property satisfied: current node <= both children
                break
        
        return res
    
    def peek(self):
        """
        Get root value without removing it.
        
        Time: O(1)
        """
        if len(self.heap) == 1:
            return None
        return self.heap[1]
    
    def heapify(self, arr):
        """
        Build heap from array in O(n) time (more efficient than pushing one-by-one).
        
        Algorithm:
        1. Move 0th element to end (for 1-based indexing compatibility)
        2. Start from first non-leaf node (n//2)
        3. Work backwards to root (index 1)
        4. At each node, percolate down (same logic as pop)
        
        Why O(n) instead of O(n log n)?
        - Only non-leaf nodes percolate (roughly n/2 nodes)
        - Nodes at level h percolate at most h levels down
        - Most nodes are leaves (no work needed)
        - Nodes that do work are closer to leaves (less percolation)
        - Mathematical sum of work simplifies to O(n)
        
        Time: O(n) - linear time! (vs O(n log n) for push one-by-one)
        Space: O(1) - in-place operation
        """
        # Move 0th element to end (for 1-based indexing)
        # This handles the case where input array is 0-based
        arr.append(arr[0])
        self.heap = arr
        
        # Start from first non-leaf node
        # First non-leaf = (n-1) // 2 in 1-based indexing
        # Since we have n elements (after moving 0th to end), first non-leaf = n // 2
        cur = (len(self.heap) - 1) // 2
        
        # Work backwards from first non-leaf to root (index 1)
        while cur > 0:
            # Percolate down from current node (same logic as pop)
            i = cur
            while 2 * i < len(self.heap):  # While has left child
                # Check if has right child and right child is smaller than left child
                if (2 * i + 1 < len(self.heap) and
                    self.heap[2 * i + 1] < self.heap[2 * i] and
                    self.heap[i] > self.heap[2 * i + 1]):
                    # Swap with right child (it's the smaller child)
                    self.heap[i], self.heap[2 * i + 1] = self.heap[2 * i + 1], self.heap[i]
                    i = 2 * i + 1
                elif self.heap[i] > self.heap[2 * i]:
                    # Swap with left child (either no right child, or left is smaller)
                    self.heap[i], self.heap[2 * i] = self.heap[2 * i], self.heap[i]
                    i = 2 * i
                else:
                    # Order property satisfied: current node <= both children
                    break
            
            cur -= 1  # Move to previous non-leaf node (work backwards)


def demo():
    """
    Demonstrate heap properties and array representation.
    """
    print("=== Heap Properties Demo ===\n")
    
    # Example: Min Heap
    # Tree representation:
    #        14
    #       /  \
    #     19   16
    #     / \  / \
    #   21 26 19 68
    #   / \
    # 65  30
    
    # Array representation (1-based indexing):
    # Index: 0  1  2  3  4  5  6  7  8  9
    # Value: [0, 14,19,16,21,26,19,68,65,30]
    
    heap = Heap()
    # Manually populate for demonstration
    heap.heap = [0, 14, 19, 16, 21, 26, 19, 68, 65, 30]
    
    print("Min Heap Example:")
    print("Tree structure:")
    print("        14")
    print("       /  \\")
    print("     19   16")
    print("     / \\  / \\")
    print("   21 26 19 68")
    print("   / \\")
    print(" 65  30")
    print()
    
    print("Array representation (1-based indexing):")
    print(f"  Index: {list(range(len(heap.heap)))}")
    print(f"  Value: {heap.heap}")
    print()
    
    print("Finding children and parent using formulas:")
    print()
    
    # Example 1: Root node (index 1, value 14)
    i = 1
    print(f"Node at index {i} (value {heap.heap[i]}):")
    print(f"  Left child: 2 * {i} = {heap.left_child(i)} → value {heap.heap[heap.left_child(i)]}")
    print(f"  Right child: 2 * {i} + 1 = {heap.right_child(i)} → value {heap.heap[heap.right_child(i)]}")
    print(f"  Parent: {i} // 2 = {heap.parent(i)} → sentinel (index 0)")
    print()
    
    # Example 2: Node at index 2 (value 19)
    i = 2
    print(f"Node at index {i} (value {heap.heap[i]}):")
    print(f"  Left child: 2 * {i} = {heap.left_child(i)} → value {heap.heap[heap.left_child(i)]}")
    print(f"  Right child: 2 * {i} + 1 = {heap.right_child(i)} → value {heap.heap[heap.right_child(i)]}")
    print(f"  Parent: {i} // 2 = {heap.parent(i)} → value {heap.heap[heap.parent(i)]}")
    print()
    
    # Example 3: Node at index 4 (value 21)
    i = 4
    print(f"Node at index {i} (value {heap.heap[i]}):")
    print(f"  Left child: 2 * {i} = {heap.left_child(i)} → value {heap.heap[heap.left_child(i)]}")
    print(f"  Right child: 2 * {i} + 1 = {heap.right_child(i)} → value {heap.heap[heap.right_child(i)]}")
    print(f"  Parent: {i} // 2 = {heap.parent(i)} → value {heap.heap[heap.parent(i)]}")
    print()
    
    print("=== Heap Properties ===")
    print("1. Structure Property:")
    print("   - Complete binary tree")
    print("   - All levels filled except last level")
    print("   - Last level filled left to right")
    print()
    print("2. Order Property (Min Heap):")
    print("   - All descendants ≥ ancestors")
    print("   - Root (14) ≤ all nodes ✓")
    print("   - 19 ≤ 21, 26 ✓")
    print("   - 16 ≤ 19, 68 ✓")
    print("   - Duplicates allowed (two 19s) ✓")
    print()
    print("=== Key Insights ===")
    print("1. Heap = Complete binary tree + Order property")
    print("2. Implemented as array (no pointers needed!)")
    print("3. 1-based indexing makes formulas work")
    print("4. BFS order fills array level by level")
    print("5. O(1) access to root (min/max)")
    print("6. O(1) navigation with formulas")
    print()


def demo_push_pop():
    """
    Demonstrate push and pop operations.
    """
    print("=== Heap Push and Pop Demo ===\n")
    
    # Create empty heap
    heap = Heap()
    
    # Push elements
    print("Pushing elements: 14, 19, 16, 21, 26, 19, 68, 65, 30")
    values = [14, 19, 16, 21, 26, 19, 68, 65, 30]
    for val in values:
        heap.push(val)
    
    print(f"Heap after pushes: {heap.heap[1:]}")
    print(f"Root (min): {heap.peek()}")
    print()
    
    # Push 17 (demonstrates percolate up)
    print("Pushing 17:")
    print("  Before: [14, 19, 16, 21, 26, 19, 68, 65, 30]")
    heap.push(17)
    print(f"  After:  {heap.heap[1:]}")
    print(f"  17 percolated up to correct position")
    print()
    
    # Pop elements
    print("Popping elements (removes minimum each time):")
    while not heap.is_empty():
        min_val = heap.pop()
        print(f"  Popped: {min_val}, Remaining: {heap.heap[1:]}")
    
    print()
    print("=== Time Complexity Summary ===")
    print("Operation    | Time Complexity")
    print("------------|----------------")
    print("Get Min/Max | O(1)")
    print("Push        | O(log n)")
    print("Pop         | O(log n)")


def demo_heapify():
    """
    Demonstrate heapify operation (building heap in O(n) time).
    """
    print("=== Heapify Demo ===\n")
    
    # Input array (0-based)
    arr = [14, 19, 16, 21, 26, 19, 68, 65, 30]
    print(f"Input array: {arr}")
    print()
    
    # Method 1: Push one-by-one (O(n log n))
    print("Method 1: Push one-by-one (O(n log n)):")
    heap1 = Heap()
    for val in arr:
        heap1.push(val)
    print(f"  Result: {heap1.heap[1:]}")
    print()
    
    # Method 2: Heapify (O(n))
    print("Method 2: Heapify (O(n)):")
    heap2 = Heap()
    heap2.heapify(arr.copy())  # Use copy to avoid modifying original
    print(f"  Result: {heap2.heap[1:]}")
    print()
    
    print("Both methods produce valid heaps!")
    print(f"Root (min): {heap2.peek()}")
    print()
    
    print("=== Time Complexity Comparison ===")
    print("Push one-by-one: O(n log n) - each push is O(log n)")
    print("Heapify:         O(n) - linear time!")
    print()
    print("Why heapify is faster:")
    print("- Only non-leaf nodes percolate (n/2 nodes)")
    print("- Most nodes are leaves (no work needed)")
    print("- Nodes that percolate are near leaves (less work)")
    print("- Mathematical sum simplifies to O(n)")


if __name__ == "__main__":
    demo()
    print("\n" + "="*50 + "\n")
    demo_push_pop()
    print("\n" + "="*50 + "\n")
    demo_heapify()