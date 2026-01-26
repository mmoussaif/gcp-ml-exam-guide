## Algorithms and Data Structures for Beginners

This main document includes short code snippets to teach the core ideas in
plain English. Full runnable code lives in `code/`.

---

## 📑 Table of Contents

### **Phase 1: Foundations**

- [0) Introduction](#0-introduction)
- [1) RAM (Memory)](#1-ram-memory)
- [2) Static Arrays](#2-static-arrays)
- [3) Dynamic Arrays](#3-dynamic-arrays)

### **Phase 2: Linear Data Structures**

- [4) Stacks](#4-stacks)
- [5) Singly Linked Lists](#5-singly-linked-lists)
- [6) Doubly Linked Lists](#6-doubly-linked-lists)
- [7) Queues](#7-queues)

### **Phase 3: Algorithms & Problem Solving**

- [8) Recursion - One Branch (Factorial)](#8-recursion---one-branch-factorial)
- [9) Recursion - Two Branch (Fibonacci)](#9-recursion---two-branch-fibonacci)
- [10) Insertion Sort](#10-insertion-sort)
- [11) Merge Sort](#11-merge-sort)
- [12) Quick Sort](#12-quick-sort)
- [13) Bucket Sort](#13-bucket-sort)
- [14) Binary Search](#14-binary-search)
- [15) Binary Search - Search a 2D Matrix](#15-binary-search---search-a-2d-matrix)
- [16) Binary Search - Search Range](#16-binary-search---search-range)

### **Phase 4: Trees**

- [17) Binary Trees](#17-binary-trees)
- [18) Binary Search Trees (BST)](#18-binary-search-trees-bst)
- [19) Depth-First Search (DFS) - Tree Traversals](#19-depth-first-search-dfs---tree-traversals)
- [20) Breadth-First Search (BFS)](#20-breadth-first-search-bfs)
- [21) BST Sets and Maps](#21-bst-sets-and-maps)

### **Phase 5: Advanced Topics**

- [22) Tree Maze (Backtracking)](#22-tree-maze-backtracking)
- [23) Heap Properties](#23-heap-properties)
- [24) Hash Usage](#24-hash-usage)
- [25) Introduction to Graphs](#25-introduction-to-graphs)
- [26) Dynamic Programming](#26-dynamic-programming)
- [27) Bit Manipulation](#27-bit-manipulation)

---

### 0) Introduction

**What this is**
This guide helps you learn DSA for interviews with beginner-friendly language,
clean code, and a clear interview narrative.

**What to say**
"I will explain my approach, justify trade-offs, and confirm complexity."

### 1) RAM (Memory)

**Concept in plain English**
RAM stores values at numeric addresses. Arrays use contiguous addresses so we
can jump to any index in constant time.

**Visual: How arrays are stored in RAM**

```
Memory Addresses:  1000    1004    1008    1012    1016
                    |       |       |       |       |
Array:            [  1  ] [  3  ] [  5  ] [  ?  ] [  ?  ]
                    |       |       |       |       |
Index:               0       1       2       3       4
                   🔵      🟢      🔴      ⚪      ⚪

🔵 = Index 0 (accessed in O(1))
🟢 = Index 1 (accessed in O(1))
🔴 = Index 2 (accessed in O(1))
⚪ = Empty/unused slots

Each integer takes 4 bytes, so addresses are 4 bytes apart.
To access arr[2], computer calculates: start_address + (2 * 4) = 1008
This is O(1) - instant access!
```

**Why it matters**
Understanding contiguous memory explains why arrays have O(1) access but
inserting in the middle is costly - we must shift elements to maintain
contiguity.

**Code snippet**

```python
arr = [1, 3, 5]
print(arr[0])  # constant-time access by index
```

**Full runnable code**
See `code/ram_demo.py`.

### 2) Static Arrays

**Concept in plain English**
A static array has a fixed size. You can read or write by index, but you cannot
grow it beyond its capacity.

**Visual: Reading from array (O(1))**

```
Array: [4, 5, 6]
Index:  0  1  2

Accessing arr[1]:
  Computer knows: start_address + (1 * element_size)
  Directly jumps to that memory location
  Result: 5
  Time: O(1) - instant!
```

**Visual: Inserting in middle (O(n))**

```
Before: [4, 5, 6, _, _]  (length=3, capacity=5)
        Want to insert 8 at index 1
        🔵 🟢 🔴 ⚪ ⚪

Step 1: Shift elements right to make room
        [4, 5, 5, 6, _]  (shifted 5 and 6 right)
        🔵 🟢 🟢 🔴 ⚪
              ↑
        Elements that moved

Step 2: Insert new value
        [4, 8, 5, 6, _]  (inserted 8 at index 1)
        🔵 🟡 🟢 🔴 ⚪
            ↑
        New element inserted

🟡 = New element being inserted
🟢🔴 = Elements that had to shift (costs O(n))

Why O(n)? We had to shift 2 elements (n-1 elements in worst case)
```

**Visual: Deleting from middle (O(n))**

```
Before: [4, 5, 6, _, _]  (length=3)
        Want to delete element at index 1 (value 5)
        🔵 🟢 🔴 ⚪ ⚪
            ↑
        Element to delete

Step 1: Shift elements left to fill gap
        [4, 6, 6, _, _]  (shifted 6 left)
        🔵 🔴 🔴 ⚪ ⚪
            ↑
        Element that moved

Step 2: Clear last position (optional)
        [4, 6, 0, _, _]  (length becomes 2)
        🔵 🔴 ⚪ ⚪ ⚪

🟢 = Element being deleted
🔴 = Elements that had to shift (costs O(n))
⚪ = Cleared/empty positions

Why O(n)? We had to shift 1 element (n-1 elements in worst case)
```

**Key operations**

- Read by index: O(1) - direct memory access
- Traverse: O(n) - must visit each element
- Insert/Delete middle: O(n) due to shifting

**Interview narrative**
"Arrays give me O(1) access, but insertions and deletions in the middle cost
O(n) because elements must shift to maintain contiguous memory layout."

**Code snippet**

```python
def remove_middle(arr, i, length):
    for idx in range(i + 1, length):
        arr[idx - 1] = arr[idx]  # Shift left
    arr[length - 1] = 0  # optional clear
```

**Full runnable code**
See `code/static_array_ops.py`.

### 3) Dynamic Arrays

**Concept in plain English**
A dynamic array grows by allocating a bigger array and copying elements when it
runs out of space. We double the capacity to keep amortized time O(1).

**Visual: Resizing process**

```
Initial: capacity=2, length=0
  [⚪, ⚪]

Add 1:   capacity=2, length=1
  [🟢, ⚪]

Add 2:   capacity=2, length=2  ← FULL! 🔴
  [🟢, 🟢]

Resize:  capacity=4, length=2  ← Doubled! 🟡
  [🟢, 🟢, ⚪, ⚪]  (copied from old array)
   ↑   ↑
  Copied elements

Add 3:   capacity=4, length=3
  [🟢, 🟢, 🔵, ⚪]

Add 4:   capacity=4, length=4  ← FULL again! 🔴
  [🟢, 🟢, 🔵, 🔵]

Resize:  capacity=8, length=4  ← Doubled again! 🟡
  [🟢, 🟢, 🔵, 🔵, ⚪, ⚪, ⚪, ⚪]
   ↑   ↑   ↑   ↑
  All copied elements

🟢 = Original elements
🔵 = New elements added
🟡 = Resize operation (O(n) cost)
🔴 = Array full (triggers resize)
⚪ = Empty slots
```

**Visual: Why doubling works (amortized O(1))**

```
To reach size 8, we did:
  - 1 copy (size 1→2)
  - 2 copies (size 2→4)
  - 4 copies (size 4→8)
  Total: 1+2+4 = 7 copies for 8 elements

Pattern: Last resize cost = sum of all previous costs
This means: amortized cost per element = O(1)
```

**Key operations**

- Append: O(1) amortized - occasionally O(n) resize, but average is O(1)
- Insert/Delete middle: O(n) - same as static arrays

**Interview narrative**
"Appends are usually O(1), but occasionally we pay O(n) to resize. By
doubling capacity, the resize cost is amortized across all operations, keeping
average append time at O(1) amortized."

**Code snippet**

```python
def push_back(arr, length, capacity, value):
    if length == capacity:
        capacity *= 2  # Double capacity
        new_arr = [0] * capacity
        for i in range(length):
            new_arr[i] = arr[i]  # Copy elements
        arr = new_arr
    arr[length] = value
    return arr, length + 1, capacity
```

**Full runnable code**
See `code/dynamic_array.py`.

### 4) Stacks

**Concept in plain English**
A stack is LIFO: last in, first out. Think of a stack of plates - you can only
add or remove from the top.

**Visual: Stack operations**

```
Empty stack:     []
                 |
                 TOP (empty) ⚪

Push 1:         [1]
                 |
                 TOP 🟢

Push 2:         [1, 2]
                    |
                   TOP 🔵

Push 3:         [1, 2, 3]
                       |
                      TOP 🔴

Pop:            [1, 2]  (returns 3)
                    |
                   TOP 🔵
                (🔴 removed)

Peek:           [1, 2]  (returns 2, doesn't remove)
                    |
                   TOP 🔵
                (🔵 still there)

🟢 = First element (bottom)
🔵 = Middle element
🔴 = Top element (last in, first out)
⚪ = Empty
```

**Visual: Stack of plates analogy**

```
        ┌───┐
        │ 3 │  ← Top (last added, first out)
        ├───┤
        │ 2 │
        ├───┤
        │ 1 │  ← Bottom (first added, last out)
        └───┘

Only the top plate is accessible!
```

**Key operations**

- Push: O(1) - add to top
- Pop: O(1) - remove from top
- Peek: O(1) - look at top without removing

**Interview narrative**
"A stack is ideal when I need to reverse order or match pairs, like parentheses.
The LIFO property makes it perfect for tracking nested structures or undoing
operations."

**Code snippet**

```python
stack = []
stack.append(10)  # push
top = stack.pop() # pop
```

**Full runnable code**
See `code/stack.py`.

### 5) Singly Linked Lists

**Concept in plain English**
Each node stores a value and a pointer to the next node. Nodes are not stored
contiguously in memory, so we can't jump directly to an index like arrays.

**Visual: Linked list structure**

```
Memory Layout (not contiguous!):
  Address 1000:  [val: "red", next: → 2000]    🔴
  Address 2000:  [val: "blue", next: → 3000]   🔵
  Address 3000:  [val: "green", next: → None]  🟢

Visual Representation:
  head → [🔴 red] → [🔵 blue] → [🟢 green] → None
          ↑         ↑          ↑
        Node1     Node2      Node3

🔴 = First node (head)
🔵 = Middle node
🟢 = Last node (tail)
→ = Pointer to next node
```

**Visual: Traversal**

```
Start: head → [🔴 red] → [🔵 blue] → [🟢 green] → None
       ↑
       cur (current pointer)

Step 1: cur = cur.next
        head → [🔴 red] → [🔵 blue] → [🟢 green] → None
                ↑
                cur

Step 2: cur = cur.next
        head → [🔴 red] → [🔵 blue] → [🟢 green] → None
                        ↑
                        cur

Step 3: cur = cur.next
        head → [🔴 red] → [🔵 blue] → [🟢 green] → None
                                ↑
                                cur

Step 4: cur = None (stop!) ⚪

🟡 = Current pointer position
🔴🔵🟢 = Nodes being traversed
⚪ = End of list
```

**Visual: Insertion (O(1) with node reference)**

```
Before:  [🔴 red] → [🔵 blue] → [🟢 green] → None
                    ↑
            Want to insert "🟡 yellow" after blue

After:   [🔴 red] → [🔵 blue] → [🟡 yellow] → [🟢 green] → None
                    ↑           ↑
            Just update pointers! O(1) time

🔴 = Existing node
🔵 = Node we're inserting after
🟡 = New node being inserted
🟢 = Node that was after blue (now after yellow)
```

**Key operations**

- Access/Search: O(n) - must traverse from head
- Insert/Delete with node reference: O(1) - just update pointers

**Interview narrative**
"Linked lists trade fast insertions for slower access. I can insert in O(1) if
I have a node reference, but accessing by index requires O(n) traversal from
the head."

**Code snippet**

```python
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

def traverse(head):
    cur = head
    while cur:
        cur = cur.next
```

**Full runnable code**
See `code/singly_linked_list.py`.

### 6) Doubly Linked Lists

**Concept in plain English**
Each node has both next and prev pointers, so we can traverse forward and back.

**Key operations**

- Insert/Delete with node reference: O(1)
- Access/Search: O(n)

**Interview narrative**
"A doubly linked list gives me efficient edits with the ability to traverse in
both directions."

**Code snippet**

```python
node.prev = prev_node
node.next = next_node
```

**Full runnable code**
See `code/doubly_linked_list.py`.

### 7) Queues

**Concept in plain English**
A queue is FIFO: first in, first out. It is great for level-order traversal.

**Key operations**

- Enqueue: O(1)
- Dequeue: O(1)

**Interview narrative**
"Queues process items in order and are perfect for BFS."

**Code snippet**

```python
from collections import deque
q = deque()
q.append(1)   # enqueue
q.popleft()  # dequeue
```

**Full runnable code**
See `code/queue.py`.

### 8) Recursion - One Branch (Factorial)

**Concept in plain English**
Recursion is when a function calls itself with a smaller input. It breaks problems
into smaller sub-problems and solves them in reverse order.

**Visual: Call stack for factorial(5)**

```
Call Stack (going down - building up):
┌─────────────────────────┐
│ factorial(5)            │ ← 🔴 Initial call
│ returns 5 * factorial(4)│
└─────────────────────────┘
         ↓ calls
┌─────────────────────────┐
│ factorial(4)            │ ← 🟠 Recursive call
│ returns 4 * factorial(3)│
└─────────────────────────┘
         ↓ calls
┌─────────────────────────┐
│ factorial(3)            │ ← 🟡 Recursive call
│ returns 3 * factorial(2)│
└─────────────────────────┘
         ↓ calls
┌─────────────────────────┐
│ factorial(2)            │ ← 🟢 Recursive call
│ returns 2 * factorial(1)│
└─────────────────────────┘
         ↓ calls
┌─────────────────────────┐
│ factorial(1)            │ ← 🔵 Base case!
│ returns 1               │
└─────────────────────────┘

Now unwinding (going back up):
🔵 factorial(1) = 1
🟢 factorial(2) = 2 * 1 = 2
🟡 factorial(3) = 3 * 2 = 6
🟠 factorial(4) = 4 * 6 = 24
🔴 factorial(5) = 5 * 24 = 120 ✓

🔴 = Initial call
🟠🟡🟢 = Recursive calls (going down)
🔵 = Base case (stops recursion)
Then we combine results going back up!
```

**Visual: Recursion tree**

```
factorial(5)
    │
    ├─ 5 * factorial(4)
           │
           ├─ 4 * factorial(3)
                  │
                  ├─ 3 * factorial(2)
                         │
                         ├─ 2 * factorial(1)
                                │
                                └─ 1 (base case)
```

**Key components**

- Base case: stops the recursion (e.g., n <= 1)
- Recursive case: calls itself with smaller input

**Interview narrative**
"I'll use recursion to break this into smaller sub-problems. The base case handles
the smallest problem, and the recursive case builds up the solution. We solve
sub-problems going down, then combine results going back up."

**Time and space**

- Time: O(n) - n function calls
- Space: O(n) - call stack depth

**Code snippet**

```python
def factorial(n):
    # Base case: smallest problem
    if n <= 1:
        return 1

    # Recursive case: break into smaller problem
    return n * factorial(n - 1)
```

**Full runnable code**
See `code/factorial.py`.

### 9) Recursion - Two Branch (Fibonacci)

**Concept in plain English**
Multi-branch recursion calls itself multiple times. Fibonacci sums the two previous
numbers, so each call spawns two more calls, creating a binary tree.

**Visual: Recursion tree for fibonacci(5) - WITHOUT memoization**

```
                    fib(5) 🔴
                   /      \
          fib(4) 🔵      fib(3) 🟡
             /      \    /      \
    fib(3) 🟢  fib(2)🟣 fib(2)🟣 fib(1)⚪
        /  \    /  \   /  \
fib(2)🟣 f(1)⚪ f(1)⚪f(0)⚪ f(1)⚪f(0)⚪
   /  \
fib(1)⚪ fib(0)⚪

🔴 = Calculated 1 time
🔵 = Calculated 1 time
🟡 = Calculated 2 times! ⚠️
🟢 = Calculated 2 times! ⚠️
🟣 = Calculated 3 times! ⚠️⚠️
⚪ = Calculated 5+ times! ⚠️⚠️⚠️

Notice: fib(3) is calculated TWICE!
        fib(2) is calculated THREE times!
        fib(1) is calculated FIVE times!
        This is why it's O(2^n) without memoization!
```

**Visual: With memoization (O(n))**

```
                    fib(5) 🔴
                   /      \
          fib(4) 🔵      fib(3) 🟢 ← ✅ Cached!
             /      \
    fib(3) 🟢  fib(2) 🟡 ← ✅ Cached!
        /  \
fib(2) 🟡  fib(1) ⚪ ← ✅ Cached!
   /  \
fib(1) ⚪ fib(0) ⚪

🔴 = Computed and cached
🔵 = Computed and cached
🟢 = Retrieved from memo (no recompute!)
🟡 = Retrieved from memo (no recompute!)
⚪ = Retrieved from memo (no recompute!)

Each value computed only ONCE = O(n) time!
Memo stores: {2: 1, 3: 2, 4: 3, 5: 5}
✅ = Cache hit (saves computation)
```

**Key insight**
The recursion tree grows exponentially. Without optimization, this is O(2^n) time
because we recalculate the same values many times. Memoization stores computed
values to avoid recalculation.

**Interview narrative**
"This is two-branch recursion. Each call creates two sub-problems, leading to
exponential growth. However, many sub-problems overlap. I can optimize with
memoization to cache results, reducing time from O(2^n) to O(n)."

**Time and space**

- Time: O(2^n) without memoization, O(n) with memoization
- Space: O(n) - call stack depth

**Code snippet**

```python
def fibonacci(n):
    # Base case
    if n <= 1:
        return n

    # Two-branch recursion
    return fibonacci(n - 1) + fibonacci(n - 2)

# Optimized with memoization
def fibonacci_memo(n, memo={}):
    if n <= 1:
        return n
    if n in memo:
        return memo[n]  # Use cached value
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]
```

**Full runnable code**
See `code/fibonacci.py`.

### 10) Insertion Sort

**Concept in plain English**
Insertion sort builds a sorted portion from left to right. For each element, it
inserts it into the correct position in the sorted portion by shifting larger
elements right.

**Visual: Step-by-step process**

```
Array: [5, 2, 4, 1, 3]
       🔴 🟡 🔵 🟢 ⚪

Step 1: i=1, element=2 🟡
  Sorted: [5] 🔴 | Unsorted: [2, 4, 1, 3] 🟡🔵🟢⚪
         ↑
  Compare 2 < 5? Yes → swap
  Result: [2, 5, 4, 1, 3]
          🟡 🔴 🔵 🟢 ⚪
          └─┘
        sorted ✅

Step 2: i=2, element=4 🔵
  Sorted: [2, 5] 🟡🔴 | Unsorted: [4, 1, 3] 🔵🟢⚪
              ↑
  Compare 4 < 5? Yes → swap → [2, 4, 5, 1, 3]
  Compare 4 < 2? No → stop
  Result: [2, 4, 5, 1, 3]
          🟡 🔵 🔴 🟢 ⚪
          └───┘
          sorted ✅

Step 3: i=3, element=1 🟢
  Sorted: [2, 4, 5] 🟡🔵🔴 | Unsorted: [1, 3] 🟢⚪
  Compare 1 < 5? Yes → swap → [2, 4, 1, 5, 3]
  Compare 1 < 4? Yes → swap → [2, 1, 4, 5, 3]
  Compare 1 < 2? Yes → swap → [1, 2, 4, 5, 3]
  Result: [1, 2, 4, 5, 3]
          🟢 🟡 🔵 🔴 ⚪
          └─────┘
          sorted ✅

Step 4: i=4, element=3 ⚪
  Sorted: [1, 2, 4, 5] 🟢🟡🔵🔴 | Unsorted: [3] ⚪
  Compare 3 < 5? Yes → swap → [1, 2, 4, 3, 5]
  Compare 3 < 4? Yes → swap → [1, 2, 3, 4, 5]
  Compare 3 < 2? No → stop
  Result: [1, 2, 3, 4, 5] ✓
          🟢 🟡 ⚪ 🔵 🔴
          └─────────┘
          sorted ✅

🔴🟡🔵🟢⚪ = Different elements being sorted
✅ = Sorted portion
↑ = Current element being inserted
```

**Visual: Card sorting analogy**

```
Your hand (sorted) | Deck (unsorted)
[5]                | [2, 4, 1, 3]
  ↑
Pick up 2, insert before 5

[2, 5]             | [4, 1, 3]
     ↑
Pick up 4, insert between 2 and 5

[2, 4, 5]          | [1, 3]
        ↑
And so on...
```

**Key insight**
Like sorting cards in your hand - pick up each card and insert it where it belongs.
If array is already sorted, inner loop never runs = O(n) best case!

**Interview narrative**
"Insertion sort is simple and works well for small or nearly-sorted arrays. It's
O(n²) worst case but O(n) for already-sorted input. It's stable and in-place,
making it useful when data is mostly sorted."

**Time and space**

- Time: O(n²) worst/average, O(n) best (already sorted)
- Space: O(1) - in-place sorting

**Code snippet**

```python
def insertion_sort(arr):
    # Start from index 1 (first element is "sorted")
    for i in range(1, len(arr)):
        j = i - 1
        # Shift elements right until correct position
        while j >= 0 and arr[j + 1] < arr[j]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
            j -= 1
    return arr
```

**Full runnable code**
See `code/insertion_sort.py`.

### 11) Merge Sort

**Concept in plain English**
Merge sort uses divide-and-conquer: split the array in half, sort each half
recursively, then merge the sorted halves together.

**Key insight**
The merge step uses two pointers to combine sorted arrays efficiently. This is
stable and always O(n log n).

**Interview narrative**
"Merge sort is a divide-and-conquer algorithm. I split in half, sort recursively,
then merge. It's O(n log n) in all cases and stable, but requires O(n) extra space."

**Time and space**

- Time: O(n log n) - always
- Space: O(n) - temporary arrays for merging

**Code snippet**

```python
def merge_sort(arr, s, e):
    # Base case: array of size 1 is sorted
    if e - s + 1 <= 1:
        return arr

    m = (s + e) // 2
    # Sort left and right halves
    merge_sort(arr, s, m)
    merge_sort(arr, m + 1, e)
    # Merge sorted halves
    merge(arr, s, m, e)
    return arr

def merge(arr, s, m, e):
    L = arr[s:m+1]
    R = arr[m+1:e+1]
    i = j = 0
    k = s

    # Merge using two pointers
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # Copy remaining elements
    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1
    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1
```

**Full runnable code**
See `code/merge_sort.py`.

### 12) Quick Sort

**Concept in plain English**
Quick sort picks a pivot, partitions the array so elements smaller than pivot are
on the left and larger on the right, then recursively sorts each side.

**Visual: Partition process**

```
Array: [6, 2, 4, 1, 3]
       🔴 🟢 🔵 🟡 🟠
Pivot: 3 🟠 (last element)
left pointer: tracks where to place next element < pivot

Step 1: Compare 6 🔴 < 3 🟠? No → skip
  [6, 2, 4, 1, 3]
   ↑
  left

Step 2: Compare 2 🟢 < 3 🟠? Yes → swap with left, move left
  [2, 6, 4, 1, 3]
   🟢 🔴 🔵 🟡 🟠
   ↑  ↑
  left i

Step 3: Compare 4 🔵 < 3 🟠? No → skip
  [2, 6, 4, 1, 3]
   🟢 🔴 🔵 🟡 🟠
   ↑     ↑
  left   i

Step 4: Compare 1 🟡 < 3 🟠? Yes → swap with left, move left
  [2, 1, 4, 6, 3]
   🟢 🟡 🔵 🔴 🟠
      ↑     ↑
     left   i

Step 5: Place pivot at left position
  [2, 1, 3, 6, 4]
   🟢 🟡 🟠 🔴 🔵
      ↑
    pivot
    Left: [2 🟢, 1 🟡] (all < 3 🟠) ✅
    Right: [6 🔴, 4 🔵] (all >= 3 🟠) ✅

🟢🟡 = Elements < pivot (moved left)
🟠 = Pivot
🔴🔵 = Elements >= pivot (stay right)
✅ = Partitioned correctly
```

**Visual: Recursive partitioning**

```
[6, 2, 4, 1, 3]
     │
     ├─ Partition around 3
     │
     ├─ Left: [2, 1]  Right: [6, 4]
     │    │                │
     │    ├─ Partition    ├─ Partition
     │    │                │
     │    ├─ [1, 2]        ├─ [4, 6]
     │    │                │
     └─── Combine: [1, 2, 3, 4, 6] ✓
```

**Key insight**
The partition step does the sorting work - no merge needed. Performance depends on
pivot choice. Best case: pivot is median (splits evenly). Worst case: pivot is
always min/max (one side empty).

**Interview narrative**
"Quick sort is divide-and-conquer with partitioning. I pick a pivot, partition
around it, then recurse on both sides. Average case is O(n log n), but worst case
is O(n²) if pivot is always smallest/largest. Randomized pivot selection helps
avoid worst case."

**Time and space**

- Time: O(n log n) average, O(n²) worst case
- Space: O(log n) - recursion stack depth

**Code snippet**

```python
def quick_sort(arr, s, e):
    # Base case
    if e - s + 1 <= 1:
        return arr

    pivot = arr[e]
    left = s

    # Partition: elements < pivot on left
    for i in range(s, e):
        if arr[i] < pivot:
            arr[left], arr[i] = arr[i], arr[left]
            left += 1

    # Place pivot in correct position
    arr[left], arr[e] = arr[e], arr[left]

    # Recursively sort left and right sides
    quick_sort(arr, s, left - 1)
    quick_sort(arr, left + 1, e)
    return arr
```

**Full runnable code**
See `code/quick_sort.py`.

### 13) Bucket Sort

**Concept in plain English**
Bucket sort works when values are in a limited, known range. Create a "bucket"
(count array) for each possible value, count frequencies, then overwrite the
array in sorted order.

**Visual: Step-by-step process**

```
Array: [2, 0, 2, 1, 1, 0]
       🔴 ⚪ 🔴 🟢 🟢 ⚪

Step 1: Count frequencies
  Value 0 ⚪: appears 2 times → counts[0] = 2
  Value 1 🟢: appears 2 times → counts[1] = 2
  Value 2 🔴: appears 2 times → counts[2] = 2

  counts = [2, 2, 2]
            ⚪ 🟢 🔴
            ↑  ↑  ↑
            0  1  2

Step 2: Overwrite array based on counts
  Write 0 ⚪ twice:  [0 ⚪, 0 ⚪, ...]
  Write 1 🟢 twice:   [0 ⚪, 0 ⚪, 1 🟢, 1 🟢, ...]
  Write 2 🔴 twice:   [0 ⚪, 0 ⚪, 1 🟢, 1 🟢, 2 🔴, 2 🔴] ✓

⚪ = Value 0 (appears 2 times)
🟢 = Value 1 (appears 2 times)
🔴 = Value 2 (appears 2 times)
✓ = Final sorted array
```

**Visual: Why nested loops are O(n), not O(n²)**

```
Outer loop: for val in range(3)  # 3 iterations (0, 1, 2)
  Inner loop: for j in range(counts[val])

Iteration 1: val=0, counts[0]=2 → inner runs 2 times
Iteration 2: val=1, counts[1]=2 → inner runs 2 times
Iteration 3: val=2, counts[2]=2 → inner runs 2 times

Total inner iterations: 2 + 2 + 2 = 6 = n (array size)

NOT 3 × 3 = 9! The inner loop doesn't run k times each iteration.
It runs exactly counts[val] times, and sum of all counts = n.
```

**Key insight**
Even though there's a nested loop, it's O(n) not O(n²). The inner loop only
runs as many times as the count for each value, and all counts sum to n.

**Interview narrative**
"Bucket sort is O(n) but only works when values are in a known range like 0-2.
I count frequencies in one pass, then write values back. The nested loop is O(n)
because the inner loop runs exactly as many times as elements exist - it's not
O(n²). It's not stable since we overwrite the array."

**Time and space**

- Time: O(n) - nested loops don't mean O(n²) here!
- Space: O(k) - where k is the number of distinct values

**Why nested loops are O(n) here**
The outer loop iterates over possible values (k iterations), and the inner loop
runs `counts[val]` times. Since all counts sum to n, total iterations = n.

**Code snippet**

```python
def bucket_sort(arr):
    # Assuming arr only contains 0, 1, or 2 (Sort Colors)
    counts = [0, 0, 0]

    # Count frequency of each value
    for n in arr:
        counts[n] += 1

    # Overwrite array in sorted order
    i = 0
    for val in range(len(counts)):
        for j in range(counts[val]):  # Inner loop runs counts[val] times
            arr[i] = val
            i += 1
    return arr
```

**Full runnable code**
See `code/bucket_sort.py`.

### 14) Binary Search

**Concept in plain English**
Binary search efficiently finds elements in a sorted array by repeatedly dividing
the search space in half. Like searching a dictionary - open to the middle, decide
if target is left or right, repeat.

**Visual: Dictionary analogy**

```
Dictionary: [A...M...Z]
            ↑   ↑   ↑
          Left Mid Right

Looking for "P":
1. Open to middle (M) → "P" comes after M → search right half
2. Open to middle of right half → find "P" ✓

Same idea with arrays!
```

**Visual: Binary search process (target found)**

```
Array: [1, 2, 3, 4, 5, 6, 7, 8]
       🔴 🟠 🟡 🟢 🔵 🟣 ⚪ ⚫
Target: 5 🔵

Initial: L=0, R=7
  [1, 2, 3, 4, 5, 6, 7, 8]
   🔴 🟠 🟡 🟢 🔵 🟣 ⚪ ⚫
   ↑               ↑
   L               R

Step 1: mid = (0+7)//2 = 3
  Compare: 5 🔵 > 4 🟢? Yes → search right
  [1, 2, 3, 4, 5, 6, 7, 8]
   🔴 🟠 🟡 🟢 🔵 🟣 ⚪ ⚫
            ↑   ↑       ↑
          mid  L        R
  Eliminate left half: 🔴🟠🟡🟢 (crossed out)

Step 2: L=4, R=7, mid = (4+7)//2 = 5
  Compare: 5 🔵 < 6 🟣? Yes → search left
  [1, 2, 3, 4, 5, 6, 7, 8]
   🔴 🟠 🟡 🟢 🔵 🟣 ⚪ ⚫
                ↑   ↑   ↑
                L  mid  R
  Eliminate right half: 🟣⚪⚫ (crossed out)

Step 3: L=4, R=4, mid = (4+4)//2 = 4
  Compare: 5 🔵 == 5 🔵? Yes → Found! ✓
  [1, 2, 3, 4, 5, 6, 7, 8]
   🔴 🟠 🟡 🟢 🔵 🟣 ⚪ ⚫
                    ↑
                  Found at index 4!
```

**Visual: Binary search process (target not found)**

```
Array: [1, 2, 3, 4, 5, 6, 7, 8]
       🔴 🟠 🟡 🟢 🔵 🟣 ⚪ ⚫
Target: 9 ⚠️ (doesn't exist)

Step 1: mid = 3, compare 9 > 4 → search right
Step 2: mid = 5, compare 9 > 6 → search right
Step 3: mid = 6, compare 9 > 7 → search right
Step 4: mid = 7, compare 9 > 8 → search right
Step 5: L=8, R=7 → L > R → exhausted search space

Result: Target not found, return -1 ⚠️
```

**Visual: Search space reduction**

```
Initial search space: [1, 2, 3, 4, 5, 6, 7, 8] (8 elements)
                      └───────────────────────┘

After 1st comparison: [5, 6, 7, 8] (4 elements) - eliminated half!
                      └───────────┘

After 2nd comparison: [5] (1 element) - eliminated half again!
                      └─┘

After 3rd comparison: Found or exhausted

Each step eliminates half → O(log n) time!
```

**Key insight**
Binary search requires a sorted array. At each step, we eliminate half the search
space by comparing the middle element to the target. This gives us O(log n) time
instead of O(n) linear search.

**Interview narrative**
"Binary search works on sorted arrays. I maintain left and right pointers, calculate
mid, and compare. If target is greater, search right half; if smaller, search left
half. Each comparison eliminates half the search space, giving O(log n) time. I use
L + (R-L)//2 to calculate mid to avoid overflow."

**Time and space**

- Time: O(log n) - eliminate half each iteration
- Space: O(1) - only using pointers

**Important formula**
Use `mid = L + (R - L) // 2` instead of `(L + R) // 2` to prevent integer overflow
when L and R are very large.

**Code snippet**

```python
def binary_search(arr, target):
    L, R = 0, len(arr) - 1

    while L <= R:
        mid = L + (R - L) // 2  # Prevents overflow

        if target > arr[mid]:
            L = mid + 1  # Search right half
        elif target < arr[mid]:
            R = mid - 1  # Search left half
        else:
            return mid  # Found!

    return -1  # Not found
```

**Full runnable code**
See `code/binary_search.py`.

### 15) Binary Search - Search a 2D Matrix

**Concept in plain English**
Search for a target in a 2D matrix where each row is sorted left-to-right, and
rows are sorted top-to-bottom. Multiple approaches exist, from brute force to
optimized binary search.

**Problem setup**
Given a sorted 2D matrix, find if target exists. The matrix has:

- Each row sorted left-to-right
- Rows sorted by their first/last elements

**Visual: Example matrix**

```
Matrix:
[ 1,  4,  7, 11]
[ 2,  5,  8, 12]
[ 3,  6,  9, 16]
[10, 13, 14, 17]

Target: 9
```

**Approach 1: Brute Force (O(m×n))**

**Concept**
Check every cell one by one. Simple but ignores the sorted structure.

**Visual: Brute force search**

```
[ 1,  4,  7, 11]  ← Check row 0: 1, 4, 7, 11
[ 2,  5,  8, 12]  ← Check row 1: 2, 5, 8, 12
[ 3,  6,  9, 16]  ← Check row 2: 3, 6, 9 ✓ Found!
[10, 13, 14, 17]

We check all 16 cells in worst case.
```

**Time and space**

- Time: O(m × n) - check every cell
- Space: O(1)

**Code snippet**

```python
def search_matrix_brute(matrix, target):
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            if matrix[r][c] == target:
                return True
    return False
```

**Approach 2: Staircase Search (O(m + n))**

**Concept**
Start at top-right corner. If value > target, move left (values decrease).
If value < target, move down (values increase). Like walking down stairs!

**Visual: Staircase search**

```
Start at top-right (0, 3):
[ 1,  4,  7, 11] ← Start here (11)
[ 2,  5,  8, 12]
[ 3,  6,  9, 16]
[10, 13, 14, 17]

Step 1: 11 > 9? Yes → move left
[ 1,  4,  7, 11]
[ 2,  5,  8, 12]
[ 3,  6,  9, 16] ← 7 < 9? Yes → move down
[10, 13, 14, 17]

Step 2: 8 < 9? Yes → move down
[ 1,  4,  7, 11]
[ 2,  5,  8, 12]
[ 3,  6,  9, 16] ← 9 == 9? Yes → Found! ✓
[10, 13, 14, 17]

Each step eliminates entire row or column!
```

**Time and space**

- Time: O(m + n) - at most m+n steps
- Space: O(1)

**Code snippet**

```python
def search_matrix_staircase(matrix, target):
    m, n = len(matrix), len(matrix[0])
    r, c = 0, n - 1  # Start top-right

    while r < m and c >= 0:
        if matrix[r][c] > target:
            c -= 1  # Move left
        elif matrix[r][c] < target:
            r += 1  # Move down
        else:
            return True
    return False
```

**Approach 3: Binary Search - Two Pass (O(log m + log n))**

**Concept**
First binary search over rows to find which row contains target. Then binary
search within that row.

**Visual: Two-pass binary search**

```
Pass 1: Find correct row
[ 1,  4,  7, 11] ← Check row 0: target 9 > 11? No, < 1? No → in this row!
[ 2,  5,  8, 12]
[ 3,  6,  9, 16]
[10, 13, 14, 17]

Actually, we check:
- Row 0: 9 > 11? No, but 9 < 1? No → 9 could be in row 0
- Row 1: 9 > 12? No, 9 < 2? No → 9 could be in row 1
- Row 2: 9 > 16? No, 9 < 3? No → 9 could be in row 2
- Row 3: 9 > 17? No, 9 < 10? Yes → 9 not in row 3

Better: Check if 9 is between first and last of each row
Row 2: 3 <= 9 <= 16 → Found candidate row!

Pass 2: Binary search within row 2
[ 3,  6,  9, 16]
  ↑   ↑   ↑   ↑
  L   m   m   R
  9 > 6? Yes → search right
  9 == 9? Yes → Found! ✓
```

**Time and space**

- Time: O(log m + log n) = O(log(m×n))
- Space: O(1)

**Code snippet**

```python
def search_matrix_two_pass(matrix, target):
    ROWS, COLS = len(matrix), len(matrix[0])

    # Pass 1: Find row
    top, bot = 0, ROWS - 1
    while top <= bot:
        row = (top + bot) // 2
        if target > matrix[row][-1]:
            top = row + 1
        elif target < matrix[row][0]:
            bot = row - 1
        else:
            break

    if not (top <= bot):
        return False

    # Pass 2: Search in row
    row = (top + bot) // 2
    l, r = 0, COLS - 1
    while l <= r:
        m = (l + r) // 2
        if target > matrix[row][m]:
            l = m + 1
        elif target < matrix[row][m]:
            r = m - 1
        else:
            return True
    return False
```

**Approach 4: Binary Search - One Pass (O(log(m×n)))**

**Concept**
Treat the entire matrix as one big sorted array. Convert 1D index to 2D
coordinates: `row = index // COLS`, `col = index % COLS`.

**Visual: One-pass binary search**

```
Matrix flattened conceptually:
Index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Value: [1, 4, 7,11, 2, 5, 8,12, 3, 6, 9,16,10,13,14,17]

Binary search on indices 0-15:
mid = 7 → row = 7 // 4 = 1, col = 7 % 4 = 3
matrix[1][3] = 12

12 > 9? Yes → search left (indices 0-6)
mid = 3 → row = 3 // 4 = 0, col = 3 % 4 = 3
matrix[0][3] = 11

11 > 9? Yes → search left (indices 0-2)
mid = 1 → row = 1 // 4 = 0, col = 1 % 4 = 1
matrix[0][1] = 4

4 < 9? Yes → search right (indices 2-2)
mid = 2 → row = 2 // 4 = 0, col = 2 % 4 = 2
matrix[0][2] = 7

7 < 9? Yes → search right (indices 3-2) → exhausted

Wait, let me recalculate...
Actually, we need to search in the right part after 7.
The key insight: treat as one sorted array!
```

**Time and space**

- Time: O(log(m×n))
- Space: O(1)

**Code snippet**

```python
def search_matrix_one_pass(matrix, target):
    ROWS, COLS = len(matrix), len(matrix[0])
    l, r = 0, ROWS * COLS - 1

    while l <= r:
        m = l + (r - l) // 2
        row, col = m // COLS, m % COLS

        if target > matrix[row][col]:
            l = m + 1
        elif target < matrix[row][col]:
            r = m - 1
        else:
            return True
    return False
```

**Common pitfalls**

1. **Wrong index conversion**: Use `row = m // COLS` not `m // ROWS`
2. **Off-by-one in row selection**: Recalculate row after first binary search
3. **Empty matrix**: Always check if matrix is empty before accessing

**Interview narrative**
"For a sorted 2D matrix, I can use staircase search starting top-right for O(m+n),
or binary search for O(log(m×n)). The one-pass binary search treats the matrix as
a flattened sorted array, converting indices using row = m // COLS and col = m % COLS."

**Full runnable code**
See `code/search_2d_matrix.py`.

### 16) Binary Search - Search Range

**Concept in plain English**
Instead of searching an array, we search a range of numbers (e.g., 1-100) using a
function that tells us if our guess is too big, too small, or correct. Like guessing
a number game!

**Visual: Number guessing game**

```
You think of a number: 10 (secret)
Friend guesses: 50
You say: "Too big!" → Friend searches 1-49

Friend guesses: 25
You say: "Too big!" → Friend searches 1-24

Friend guesses: 12
You say: "Too big!" → Friend searches 1-11

Friend guesses: 5
You say: "Too small!" → Friend searches 6-11

Friend guesses: 8
You say: "Too small!" → Friend searches 9-11

Friend guesses: 10
You say: "Correct!" ✓

This is binary search on a range!
```

**Key difference from array search**

- Array search: Compare `arr[mid]` with `target` directly
- Range search: Use a function `isCorrect(mid)` that returns:
  - `1` if guess is too big
  - `-1` if guess is too small
  - `0` if guess is correct

**Visual: Binary search on range**

```
Search range: 1 to 100
Target: 10 (hidden, we use isCorrect function)

Step 1: mid = (1 + 100) // 2 = 50
  isCorrect(50) → returns 1 (too big)
  Search space: 1-49

Step 2: mid = (1 + 49) // 2 = 25
  isCorrect(25) → returns 1 (too big)
  Search space: 1-24

Step 3: mid = (1 + 24) // 2 = 12
  isCorrect(12) → returns 1 (too big)
  Search space: 1-11

Step 4: mid = (1 + 11) // 2 = 6
  isCorrect(6) → returns -1 (too small)
  Search space: 7-11

Step 5: mid = (7 + 11) // 2 = 9
  isCorrect(9) → returns -1 (too small)
  Search space: 10-11

Step 6: mid = (10 + 11) // 2 = 10
  isCorrect(10) → returns 0 (correct!) ✓

Found: 10
```

**Visual: Search space reduction**

```
Initial: [1────────────────────────100] (100 numbers)
         └──────────────────────────┘

After guess 50: [1──────────49] (49 numbers)
                └──────────┘

After guess 25: [1────24] (24 numbers)
                └────┘

After guess 12: [1──11] (11 numbers)
                └──┘

After guess 6: [7──11] (5 numbers)
               └──┘

After guess 9: [10-11] (2 numbers)
               └─┘

Found: 10 ✓

Each step eliminates half → O(log n) time!
```

**Key insight**
The function `isCorrect()` acts as a "black box" - we don't need to know how it
works internally, just how to interpret its return values. This pattern appears in
many problems like "Guess Number Higher or Lower" and "First Bad Version".

**Interview narrative**
"This is binary search on a range instead of an array. I use a comparison function
that tells me if my guess is too big, too small, or correct. I adjust the search
space based on the function's return value. The time complexity is still O(log n)
where n is the size of the range."

**Time and space**

- Time: O(log n) - where n is the size of the search range
- Space: O(1) - only using pointers

**Code snippet**

```python
def is_correct(n, target):
    """Example comparison function."""
    if n > target:
        return 1   # Too big
    elif n < target:
        return -1  # Too small
    else:
        return 0   # Correct

def binary_search_range(low, high, is_correct_func):
    """
    Binary search on a range using a comparison function.

    The function is_correct_func(n) returns:
    - 1 if n is too big
    - -1 if n is too small
    - 0 if n is correct
    """
    while low <= high:
        mid = low + (high - low) // 2
        result = is_correct_func(mid)

        if result > 0:
            # Too big → search left half
            high = mid - 1
        elif result < 0:
            # Too small → search right half
            low = mid + 1
        else:
            # Found!
            return mid

    return -1  # Not found
```

**Common patterns**

1. **Guess Number Higher or Lower**: `isCorrect` compares guess to secret number
2. **First Bad Version**: `isBadVersion` tells if version is bad (monotonic)
3. **Koko Eating Bananas**: Function checks if eating speed is feasible

**Important note**
The comparison function must be monotonic - if `isCorrect(n)` returns "too big",
then all numbers > n are also "too big". This ensures binary search works correctly.

**Full runnable code**
See `code/search_range.py`.

### 17) Binary Trees

**Concept in plain English**
Similar to linked lists, binary trees use nodes and pointers. But instead of connecting
nodes in a straight line, binary trees connect nodes hierarchically with left and right
child pointers. The first node is called the root, and we draw pointers downward.

**Visual: Binary tree structure**

```
        🔴 1 (Root)
       /   \
    🟢 2   🔵 3
     / \   / \
  🟡 4 ⚪ 5 🟣 6
   /     \
⚪ 7     ⚫ 8

🔴 = Root node (no parent)
🟢🔵 = Children of root
🟡⚪🟣 = Grandchildren
⚪⚫ = Leaf nodes (no children)
```

**Visual: TreeNode class**

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None   # Left child pointer
        self.right = None  # Right child pointer
```

**Key properties**

- **At most 2 children**: Each node has left and/or right child (or neither)
- **No cycles**: Pointers only go downward (unlike linked lists)
- **Connected**: All nodes reachable from root
- **Undirected graph**: Mathematically, a tree is a connected, undirected graph with no cycles
- **Guaranteed leaves**: Leaf nodes always exist (nodes with no children)

**Visual: Tree vs Linked List**

```
Linked List:  1 → 2 → 3 → 4 → 5
              Linear, can have cycles

Binary Tree:     1
                / \
               2   3
              / \ / \
             4  5 6  7
             Hierarchical, no cycles!
```

**Properties and Terminology**

**Root Node**
The highest node in the tree with no parent. All nodes can be reached from the root.

```
        🔴 Root
       /   \
    🟢 2   🔵 3
     / \   / \
  🟡 4 ⚪ 5 🟣 6

🔴 = Root (no parent, top of tree)
```

**Leaf Nodes**
Nodes with no children. Found at the last level, but can exist on other levels too.

```
        🔴 1
       /   \
    🟢 2   🔵 3
     / \   / \
  🟡 4 ⚪ 5 🟣 6
   /     \
⚪ 7     ⚫ 8

⚪⚫ = Leaf nodes (4, 5, 7, 8 have no children)
```

**Children**
The left child and right child of a node.

```
        🔴 1
       /   \
    🟢 2   🔵 3
     ↑     ↑
  Left  Right
  child child
  of 1  of 1
```

**Height**
Distance from root to the lowest leaf node. Can be counted by nodes or edges.

```
        🔴 1  (Level 0)
       /   \
    🟢 2   🔵 3  (Level 1)
     / \   / \
  🟡 4 ⚪ 5 🟣 6  (Level 2)
   /     \
⚪ 7     ⚫ 8  (Level 3)

Height by nodes: 4 (1→2→4→7)
Height by edges: 3 (3 edges in longest path)

Note: Number of edges = n - 1 (where n = number of nodes)
```

**Depth**
Distance from a node up to the root (including the node itself).

```
        🔴 1  (Depth = 1)
       /   \
    🟢 2   🔵 3  (Depth = 2)
     / \   / \
  🟡 4 ⚪ 5 🟣 6  (Depth = 3)
   /     \
⚪ 7     ⚫ 8  (Depth = 4)

Depth increases as we go down the tree.
Root has depth = 1 (or 0 if counting from 0).
```

**Ancestor**
A node connected to all nodes below it. The root is ancestor to all nodes.

```
        🔴 1 (Ancestor of all)
       /   \
    🟢 2   🔵 3
     / \   / \
  🟡 4 ⚪ 5 🟣 6

🔴 1 is ancestor of: 2, 3, 4, 5, 6
🟢 2 is ancestor of: 4, 5
🔵 3 is ancestor of: 6
```

**Descendant**
A node that is a child or child of a descendant. All nodes below a node are its descendants.

```
        🔴 1
       /   \
    🟢 2   🔵 3
     / \   / \
  🟡 4 ⚪ 5 🟣 6

Descendants of 🔴 1: 2, 3, 4, 5, 6
Descendants of 🟢 2: 4, 5
Descendants of 🔵 3: 6
```

**Visual: Non-leaf vs Leaf nodes**

```
        🔴 1 (Non-leaf - has children)
       /   \
    🟢 2   🔵 3 (Non-leaf - has children)
     / \   / \
  🟡 4 ⚪ 5 🟣 6
   /     \
⚪ 7     ⚫ 8 (Leaf - no children)

Non-leaf nodes: 1, 2, 3, 4, 6 (have at least one child)
Leaf nodes: 5, 7, 8 (have no children)
```

**Interview narrative**
"A binary tree is a hierarchical structure where each node has at most two children:
left and right. The root has no parent, and leaf nodes have no children. Height is
measured from root to lowest leaf, depth from a node to root. Trees are connected
with no cycles, making them perfect for recursive algorithms."

**Time and space**

- Traversal: O(n) - must visit each node once
- Space: O(h) - where h is height (for recursion stack)

**Code snippet**

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None   # Left child pointer
        self.right = None  # Right child pointer
```

**Full runnable code**
See `code/binary_tree.py`.

### 18) Binary Search Trees (BST)

**Concept in plain English**
Binary Search Trees are a variation of binary trees with a sorted property: every
node in the left subtree is smaller than the root, and every node in the right subtree
is greater than the root. This property applies recursively to every node.

**Visual: Binary Tree vs Binary Search Tree**

```
Regular Binary Tree:
        🔴 5
       /   \
    🟢 3   🔵 8
     / \   / \
  🟡 1 ⚪ 7 🟣 9
     No ordering property

Binary Search Tree:
        🔴 5
       /   \
    🟢 3   🔵 8
     / \   / \
  🟡 1 ⚪ 7 🟣 9
     Sorted property: left < node < right
```

**Visual: BST property (recursive)**

```
Valid BST:
        🔴 5
       /   \
    🟢 3   🔵 8
     / \   / \
  🟡 1 ⚪ 7 🟣 9

Property check (recursive):
- Node 5: left subtree (3,1) all < 5 ✓
          right subtree (8,7,9) all > 5 ✓
- Node 3: left subtree (1) all < 3 ✓
          right subtree (4) all > 3 ✓
- Node 8: left subtree (7) all < 8 ✓
          right subtree (9) all > 8 ✓

This property applies to EVERY node!
```

**Motivation: Why use BST?**

**Visual: BST vs Sorted Array**

```
Sorted Array: [1, 3, 5, 7, 9]
- Search: O(log n) ✓
- Insert: O(n) ❌ (must shift elements)
- Delete: O(n) ❌ (must shift elements)

BST:
        🔴 5
       /   \
    🟢 3   🔵 8
     / \   / \
  🟡 1 ⚪ 7 🟣 9
- Search: O(log n) ✓
- Insert: O(log n) ✓ (just add node)
- Delete: O(log n) ✓ (just remove node)

BST allows O(log n) insert/delete, unlike arrays!
```

**BST Search Algorithm**

**Visual: Search process**

```
Tree: [2, 1, 3, null, null, null, 4]
        🔴 2
       /   \
    🟢 1   🔵 3
              \
            🟣 4

Search for target = 3:

Step 1: Compare 3 with root (2)
        3 > 2 → search right subtree
        Eliminate left subtree: 🟢 1 (crossed out)

Step 2: Compare 3 with node (3)
        3 == 3 → Found! ✓

Result: True
```

**Visual: Search when target doesn't exist**

```
Tree: [2, 1, 3, null, null, null, 4]
        🔴 2
       /   \
    🟢 1   🔵 3
              \
            🟣 4

Search for target = 5:

Step 1: Compare 5 with root (2)
        5 > 2 → search right subtree

Step 2: Compare 5 with node (3)
        5 > 3 → search right subtree

Step 3: Compare 5 with node (4)
        5 > 4 → search right subtree

Step 4: Reached null → target not found
        Return False ❌
```

**Key insight**
BST search is like binary search on a sorted array, but we navigate a tree structure
instead. At each step, we compare and eliminate half the remaining nodes.

**Interview narrative**
"A BST has a sorted property: left subtree < node < right subtree, applied recursively.
This enables O(log n) search by comparing and eliminating half the tree each step, just
like binary search. BSTs are preferred over sorted arrays because insert/delete are
also O(log n), not O(n)."

**Time and space**

- **Balanced BST**: O(log n) - height is log n, eliminate half each step
- **Skewed BST**: O(n) - worst case, tree becomes a linked list
- Space: O(h) where h is height (recursion stack)

**Visual: Balanced vs Skewed**

```
Balanced BST (O(log n)):
        🔴 4
       /   \
    🟢 2   🔵 6
     / \   / \
  🟡 1 ⚪ 3 5 🟣 7
  Height: 3, nodes: 7
  log₂(7) ≈ 2.8 ✓

Skewed BST (O(n)):
  🔴 1
   \
    🔵 2
     \
      🟢 3
       \
        🟡 4
  Height: 4, nodes: 4
  Degrades to O(n) ❌
```

**Code snippet**

```python
def search(root, target):
    """
    Search for target in BST using recursion.

    Algorithm:
    1. Base case: If root is None, return False
    2. If target > root.val, search right subtree
    3. If target < root.val, search left subtree
    4. If target == root.val, return True
    """
    if not root:
        return False  # Base case: target not found

    if target > root.val:
        return search(root.right, target)  # Search right (larger values)
    elif target < root.val:
        return search(root.left, target)   # Search left (smaller values)
    else:
        return True  # Base case: target found
```

**BST Insertion**

**Concept in plain English**
Insert a new node while maintaining the BST property. Traverse to find the correct
position (like search), then add the node as a leaf. This is O(log n) vs O(n) for
sorted arrays.

**Visual: Insertion process**

```
Initial BST: [4]
        🔴 4

Insert 6: [4, null, 6]
        🔴 4
            \
          🔵 6

Insert 5: [4, null, 6, 5, null]
        🔴 4
            \
          🔵 6
          /
      🟢 5

Step-by-step for inserting 5:
1. Compare 5 with 4 → 5 > 4 → go right
2. Compare 5 with 6 → 5 < 6 → go left
3. Reached null → insert 5 here ✓
```

**Visual: Insertion always adds as leaf**

```
Before:        🔴 4
                  \
                🔵 6

Insert 5:      🔴 4
                  \
                🔵 6
                /
            🟢 5 (new leaf)

We always insert at a leaf position to maintain BST property!
```

**Key insight**
Insertion always adds the new node as a leaf. We traverse until we find a null
position, then create a new node there. This maintains the BST property.

**Code snippet**

```python
def insert(root, val):
    """
    Insert new node and return root of BST.

    Algorithm:
    1. If root is None, create new node (base case)
    2. If val > root.val, insert into right subtree
    3. If val < root.val, insert into left subtree
    4. Return root (maintains tree structure)
    """
    if not root:
        return TreeNode(val)  # Base case: add new leaf node

    if val > root.val:
        root.right = insert(root.right, val)  # Insert right
    elif val < root.val:
        root.left = insert(root.left, val)   # Insert left

    return root  # Return current node (maintains structure)
```

**BST Removal**

**Concept in plain English**
Remove a node while maintaining BST property. Two cases: node has 0-1 children
(easy), or node has 2 children (replace with in-order successor).

**Visual: Case 1 - Node with 0 or 1 child**

```
Case 1a: Delete node with 0 children (leaf)
Before:        🔴 3
              /   \
          🟢 2   🔵 4

Delete 2:     🔴 3
                  \
                🔵 4
        (Simply remove the leaf)

Case 1b: Delete node with 1 child
Before:        🔴 3
              /   \
          🟢 2   🔵 4

Delete 3:     🟢 2
                  \
                🔵 4
        (Replace with child)
```

**Visual: Case 2 - Node with 2 children**

```
Before:        🔴 5
              /   \
          🟢 3   🔵 7
         / \     / \
      🟡 2 ⚪ 4 6 🟣 8

Delete 5 (has 2 children):

Step 1: Find in-order successor (leftmost in right subtree)
        Right subtree of 5: [7, 6, 8]
        Leftmost: 6 🟢

Step 2: Replace 5 with 6
        🔴 6 (was 5)
       /   \
   🟢 3   🔵 7
  / \     / \
🟡 2 ⚪ 4 null 🟣 8

Step 3: Delete the duplicate 6 from right subtree
        🔴 6
       /   \
   🟢 3   🔵 7
  / \       \
🟡 2 ⚪ 4   🟣 8

In-order successor = smallest value > target
This maintains BST property!
```

**Visual: Finding in-order successor**

```
Target node: 5 🔴
Right subtree:     🔵 7
                  / \
                🟢 6 🟣 8
                /
            ⚪ null

In-order successor = leftmost node in right subtree
Start at 7, go left until null → 6 ✓

Why this works:
- 6 is smallest value > 5
- Replacing 5 with 6 maintains: left < 6 < right
```

**Code snippet**

```python
def min_value_node(root):
    """Find leftmost (minimum) node in subtree."""
    curr = root
    while curr and curr.left:
        curr = curr.left
    return curr

def remove(root, val):
    """
    Remove node with value val and return root of BST.

    Algorithm:
    1. If root is None, return None (not found)
    2. If val > root.val, remove from right subtree
    3. If val < root.val, remove from left subtree
    4. If val == root.val (found node to delete):
       a. Case 1: 0 or 1 child → replace with child (or None)
       b. Case 2: 2 children → replace with in-order successor
    """
    if not root:
        return None

    if val > root.val:
        root.right = remove(root.right, val)
    elif val < root.val:
        root.left = remove(root.left, val)
    else:
        # Found node to delete
        if not root.left:
            return root.right  # Case 1a: 0 children, or only right child
        elif not root.right:
            return root.left   # Case 1b: Only left child
        else:
            # Case 2: 2 children
            min_node = min_value_node(root.right)  # Find successor
            root.val = min_node.val                # Replace value
            root.right = remove(root.right, min_node.val)  # Delete duplicate

    return root
```

**Time and space**

- **Time**: O(log n) balanced, O(n) worst (skewed tree)
- **Space**: O(log n) balanced, O(n) worst (recursion stack)

**Interview narrative**
"To insert, I traverse like search until finding a null position, then add the node
as a leaf - O(log n). For removal, if the node has 0-1 children, I replace it with
its child. If it has 2 children, I replace it with its in-order successor (leftmost
in right subtree), then delete the duplicate. This maintains BST property."

**Full runnable code**
See `code/binary_search_tree.py`.

### 19) Depth-First Search (DFS) - Tree Traversals

**Concept in plain English**
Depth-First Search goes as deep as possible before backtracking. Pick a direction
(left), follow pointers down until null, then backtrack to parent and go right.
Repeat until all nodes visited. Best implemented with recursion.

**Visual: DFS concept**

```
        🔴 4
       /   \
    🟢 3   🔵 6
     / \   / \
  🟡 2 ⚪ 5 🟣 7

DFS Process:
1. Start at root (4) 🔴
2. Go left → 3 🟢
3. Go left → 2 🟡 (can't go further)
4. Backtrack to 3, go right → null
5. Backtrack to 4, go right → 6 🔵
6. Go left → 5 ⚪
7. Backtrack to 6, go right → 7 🟣

We go DEEP before backtracking!
```

**Three DFS Traversal Methods**

**1. Inorder Traversal (Left → Root → Right)**

**Concept**
Recursively visit left subtree, then parent, then right subtree.

**Visual: Inorder traversal**

```
Tree:        🔴 4
            /   \
        🟢 3   🔵 6
         / \   / \
      🟡 2 ⚪ 5 🟣 7

Visit order (numbers show sequence):
        🔴 4 (4)
       /   \
    🟢 3   🔵 6 (6)
     / \   / \
  🟡 2 ⚪ 5 🟣 7
 (1) (2) (3) (5) (7)

Result: [2, 3, 4, 5, 6, 7] ✓ (sorted!)

Process:
1. Go left to 3 → go left to 2 → visit 2 (1)
2. Backtrack to 3 → visit 3 (2)
3. Backtrack to 4 → visit 4 (3)
4. Go right to 6 → go left to 5 → visit 5 (4)
5. Backtrack to 6 → visit 6 (5)
6. Go right to 7 → visit 7 (6)
```

**Key insight for BST**
Inorder traversal gives sorted order for BSTs! Because left < root < right, visiting
in left-root-right order naturally produces sorted sequence.

**Code snippet**

```python
def inorder(root):
    """
    Inorder: Left → Root → Right

    For BST: Returns nodes in sorted order!
    """
    if not root:
        return

    inorder(root.left)   # Visit left subtree first
    print(root.val)      # Visit root
    inorder(root.right)  # Visit right subtree last
```

**2. Preorder Traversal (Root → Left → Right)**

**Concept**
Visit parent first, then left subtree, then right subtree.

**Visual: Preorder traversal**

```
Tree:        🔴 4 (1)
            /   \
        🟢 3   🔵 6 (4)
         / \   / \
      🟡 2 ⚪ 5 🟣 7
     (2) (3) (5) (6) (7)

Result: [4, 3, 2, 6, 5, 7]

Process:
1. Visit root 4 (1)
2. Go left → visit 3 (2)
3. Go left → visit 2 (3)
4. Backtrack, go right → null
5. Backtrack to 4, go right → visit 6 (4)
6. Go left → visit 5 (5)
7. Backtrack, go right → visit 7 (6)
```

**Code snippet**

```python
def preorder(root):
    """
    Preorder: Root → Left → Right

    Useful for: Copying trees, prefix notation
    """
    if not root:
        return

    print(root.val)      # Visit root first
    preorder(root.left)  # Then left subtree
    preorder(root.right) # Then right subtree
```

**3. Postorder Traversal (Left → Right → Root)**

**Concept**
Visit left subtree, then right subtree, then parent last.

**Visual: Postorder traversal**

```
Tree:        🔴 4 (6)
            /   \
        🟢 3   🔵 6 (5)
         / \   / \
      🟡 2 ⚪ 5 🟣 7
     (1) (2) (4) (3) (7)

Result: [2, 3, 5, 7, 6, 4]

Process:
1. Go left to 3 → go left to 2 → visit 2 (1)
2. Backtrack to 3 → visit 3 (2)
3. Backtrack to 4 → go right to 6
4. Go left to 5 → visit 5 (3)
5. Backtrack to 6 → go right to 7 → visit 7 (4)
6. Backtrack to 6 → visit 6 (5)
7. Backtrack to 4 → visit 4 (6)
```

**Code snippet**

```python
def postorder(root):
    """
    Postorder: Left → Right → Root

    Useful for: Deleting trees, postfix notation, calculating expressions
    """
    if not root:
        return

    postorder(root.left)  # Visit left subtree first
    postorder(root.right) # Then right subtree
    print(root.val)       # Visit root last
```

**Visual: Comparison of all three**

```
Tree:        🔴 4
            /   \
        🟢 3   🔵 6
         / \   / \
      🟡 2 ⚪ 5 🟣 7

Inorder:   [2, 3, 4, 5, 6, 7] ← Sorted for BST!
Preorder:  [4, 3, 2, 6, 5, 7] ← Root first
Postorder: [2, 3, 5, 7, 6, 4] ← Root last
```

**Time and space**

- **Time**: O(n) - must visit every node, regardless of tree height
- **Space**: O(h) where h is height
  - Balanced tree: O(log n)
  - Skewed tree: O(n)

**Interview narrative**
"DFS traverses trees by going deep before backtracking. There are three orders:
inorder (left-root-right) gives sorted order for BSTs, preorder (root-left-right)
processes root first, and postorder (left-right-root) processes root last. All use
recursion naturally and visit each node once, giving O(n) time."

**Full runnable code**
See `code/tree_traversals.py`.

---

### 20) Breadth-First Search (BFS)

**Concept in plain English**
BFS prioritizes breadth - visit all nodes on one level before moving to the next.
Also called level-order traversal for trees. Implemented iteratively using a queue.

**Visual: DFS vs BFS**

```
Tree:        🔴 4
            /   \
        🟢 3   🔵 6
         / \   / \
      🟡 2 ⚪ 5 🟣 7

DFS (goes deep):     BFS (goes level by level):
4 → 3 → 2 → ...     4 (level 0)
                    3 → 6 (level 1)
                    2 → 5 → 5 → 7 (level 2)
```

**Visual: BFS process with queue**

```
Tree:        🔴 4
            /   \
        🟢 3   🔵 6
         / \   / \
      🟡 2 ⚪ 5 🟣 7

Level 0:
  Queue: [4] 🔴
  Process: 4
  Add children: [3, 6] 🟢🔵

Level 1:
  Queue: [3, 6] 🟢🔵
  Process: 3, add children → [6, 2, 5] 🔵🟡⚪
  Process: 6, add children → [2, 5, 5, 7] 🟡⚪⚪🟣

Level 2:
  Queue: [2, 5, 5, 7] 🟡⚪⚪🟣
  Process all (no children to add)

Result: [4, 3, 6, 2, 5, 5, 7] (level by level)
```

**Visual: Queue state at each level**

```
Initial: Queue = [4] 🔴
         Level 0

After level 0: Queue = [3, 6] 🟢🔵
              Level 1

After level 1: Queue = [2, 5, 5, 7] 🟡⚪⚪🟣
              Level 2

After level 2: Queue = [] (empty, done!)
```

**Key insight**
BFS uses a queue (FIFO) to process nodes level by level. We enqueue children as
we process parents, ensuring we visit all nodes at one level before the next.

**Interview narrative**
"BFS visits nodes level by level using a queue. I enqueue the root, then while the
queue isn't empty, I process all nodes at the current level and enqueue their children.
This gives level-order traversal. Time is O(n) to visit all nodes, space is O(n)
for the queue which can hold up to half the tree (last level)."

**Time and space**

- **Time**: O(n) - visit every node exactly once
- **Space**: O(n) - queue stores entire level, worst case last level is ~n/2 nodes

**Code snippet**

```python
from collections import deque

def bfs(root):
    """
    Breadth-First Search (Level-Order Traversal).

    Algorithm:
    1. Enqueue root
    2. While queue not empty:
       a. Process all nodes at current level
       b. Enqueue children (left, then right)
       c. Move to next level
    """
    queue = deque()

    if root:
        queue.append(root)  # Start with root

    level = 0
    while queue:
        print(f"Level {level}:")
        level_size = len(queue)  # Process all nodes at current level

        for _ in range(level_size):
            curr = queue.popleft()  # Remove from head (FIFO)
            print(curr.val)

            # Add children to tail (for next level)
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)

        level += 1
```

**Visual: Why queue works**

```
Queue (FIFO): [head] ... [tail]
              ↑          ↑
            Remove    Add

Level 0: Add 4 → [4]
         Process 4, add children → [3, 6]

Level 1: Process 3 → [6]
         Add 3's children → [6, 2, 5]
         Process 6 → [2, 5]
         Add 6's children → [2, 5, 5, 7]

Level 2: Process all → []

FIFO ensures we process level by level!
```

**Full runnable code**
See `code/tree_traversals.py`.

---

### 21) BST Sets and Maps

**Concept in plain English**
Sets and Maps are interfaces that can be implemented using BSTs. This gives O(log n) time
for insertion, deletion, and search operations. Sets store unique values in sorted order
(TreeSet). Maps store key-value pairs sorted by key (TreeMap).

**Visual: Set vs Map**

```
TreeSet (unique values, sorted):
        🔴 "Alice"
       /   \
  🟢 "Brad"  🔵 "Collin"

Values: {"Alice", "Brad", "Collin"}
- Unique: No duplicates ✓
- Sorted: Alphabetically ordered ✓

TreeMap (key-value pairs, sorted by key):
        🔴 "Alice" → 123
       /   \
  🟢 "Brad" → 345  🔵 "Collin" → 678

Mapping: {"Alice": 123, "Brad": 345, "Collin": 678}
- Keys: Sorted alphabetically ✓
- Values: Can be any type (phone numbers, objects, etc.)
```

**Visual: Phone book example**

```
Phone Book as TreeSet (names only):
        🔴 "Alice"
       /   \
  🟢 "Brad"  🔵 "Collin"

Phone Book as TreeMap (names → phone numbers):
        🔴 "Alice" → 123
       /   \
  🟢 "Brad" → 345  🔵 "Collin" → 678

Operations:
- Search "Brad": O(log n) - traverse tree
- Insert "David": O(log n) - add as leaf
- Delete "Alice": O(log n) - remove node
```

**Key insight: Why use BST?**

**Visual: BST vs Dynamic Array**

```
Dynamic Array for phone book:
["Alice", "Brad", "Collin"]
- Search: O(n) - must check each element ❌
- Insert: O(n) - must shift elements ❌
- Delete: O(n) - must shift elements ❌

BST (TreeSet/TreeMap):
        🔴 "Alice"
       /   \
  🟢 "Brad"  🔵 "Collin"
- Search: O(log n) - binary search ✓
- Insert: O(log n) - add leaf ✓
- Delete: O(log n) - remove node ✓
```

**Sets (TreeSet)**

**Concept in plain English**
A Set ensures unique values. When implemented with a BST, values are automatically
sorted. Perfect for maintaining a sorted collection of unique items.

**Visual: TreeSet operations**

```
Initial TreeSet: {}
        (empty)

Insert "Collin": {"Collin"}
        🔴 "Collin"

Insert "Alice": {"Alice", "Collin"}
        🔴 "Collin"
       /
  🟢 "Alice"

Insert "Brad": {"Alice", "Brad", "Collin"}
        🔴 "Collin"
       /
  🟢 "Alice"
       \
    🔵 "Brad"

Try insert "Alice" again: {"Alice", "Brad", "Collin"}
        (no change - duplicates not allowed!)
```

**Maps (TreeMap)**

**Concept in plain English**
A Map stores key-value pairs. When implemented with a BST, keys are sorted. Values
can be any type (numbers, objects, etc.). Only keys need to be comparable.

**Visual: TreeMap operations**

```
Initial TreeMap: {}
        (empty)

Insert ("Collin", 678): {"Collin": 678}
        🔴 "Collin" → 678

Insert ("Alice", 123): {"Alice": 123, "Collin": 678}
        🔴 "Collin" → 678
       /
  🟢 "Alice" → 123

Insert ("Brad", 345): {"Alice": 123, "Brad": 345, "Collin": 678}
        🔴 "Collin" → 678
       /
  🟢 "Alice" → 123
       \
    🔵 "Brad" → 345

Search "Brad": Returns 345 ✓
Update "Brad" to 999: {"Alice": 123, "Brad": 999, "Collin": 678}
        🔴 "Collin" → 678
       /
  🟢 "Alice" → 123
       \
    🔵 "Brad" → 999 (updated!)
```

**Visual: Key requirement**

```
TreeMap requires comparable keys:

✅ Valid keys:
- Strings: "Alice", "Brad", "Collin"
- Numbers: 1, 2, 3
- Dates: 2024-01-01, 2024-01-02

❌ Invalid keys:
- Objects without comparison (unless custom comparator)
- Unordered types

Values can be ANYTHING:
- Numbers: 123, 345, 678
- Objects: Person, Address, etc.
- Lists, arrays, other data structures
```

**Time and space**

- **Time**: O(log n) for all operations (search, insert, delete)
  - Balanced BST: O(log n)
  - Skewed BST: O(n) worst case
- **Space**: O(n) to store n key-value pairs

**Interview narrative**
"Sets and Maps can be implemented using BSTs to get O(log n) operations. A TreeSet
stores unique values in sorted order. A TreeMap stores key-value pairs sorted by key.
Both leverage the BST property: left < node < right, giving efficient search/insert/delete.
The key must be comparable, but values can be any type."

**Implementation in different languages**

**Python**

```python
from sortedcontainers import SortedDict

# TreeMap
treemap = SortedDict({'c': 3, 'a': 1, 'b': 2})
# Keys automatically sorted: {'a': 1, 'b': 2, 'c': 3}

# TreeSet (use SortedList or SortedSet)
from sortedcontainers import SortedSet
treeset = SortedSet(['c', 'a', 'b'])
# Values automatically sorted: SortedSet(['a', 'b', 'c'])
```

**Java**

```java
// TreeMap
TreeMap<String, Integer> treeMap = new TreeMap<>();
treeMap.put("Alice", 123);
treeMap.put("Brad", 345);
treeMap.put("Collin", 678);

// TreeSet
TreeSet<String> treeSet = new TreeSet<>();
treeSet.add("Alice");
treeSet.add("Brad");
treeSet.add("Collin");
```

**C++**

```cpp
// TreeMap (std::map)
map<string, int> treeMap;
treeMap["Alice"] = 123;
treeMap["Brad"] = 345;
treeMap["Collin"] = 678;

// TreeSet (std::set)
set<string> treeSet;
treeSet.insert("Alice");
treeSet.insert("Brad");
treeSet.insert("Collin");
```

**JavaScript**

```javascript
// TreeMap (requires external library)
const TreeMap = require('treemap-js')
let map = new TreeMap()
map.set('Alice', 123)
map.set('Brad', 345)
map.set('Collin', 678)
```

**Closing notes**
BSTs are one way to implement Sets and Maps. Hashing (HashSet, HashMap) is another
approach we'll see later, offering O(1) average time but without sorted order.

**Visual: BST vs Hash implementation**

```
BST (TreeSet/TreeMap):
- Sorted: Keys in order ✓
- Time: O(log n) operations
- Space: O(n)
- Use when: Need sorted order

Hash (HashSet/HashMap):
- Unsorted: Keys in random order ❌
- Time: O(1) average operations ✓
- Space: O(n)
- Use when: Don't need sorted order, want faster operations
```

**Full runnable code**
See `code/binary_search_tree.py` for BST operations that form the basis of TreeSet/TreeMap.

---

## Phase 5: Advanced Topics

### 22) Tree Maze (Backtracking)

**Concept in plain English**
Backtracking tries all possible solutions and backtracks when hitting a dead-end. Like
being trapped in a maze: try all paths, backtrack from dead-ends, find the correct path.
It overlaps with DFS but emphasizes constraint checking and undoing choices.

**Visual: Maze analogy**

```
Maze:
    🚪 START
    |
    ├─→ 🟢 Path A → ❌ Dead-end → Backtrack
    |
    └─→ 🔵 Path B → ✅ EXIT

Backtracking process:
1. Try Path A → Hit dead-end
2. Backtrack to START
3. Try Path B → Found exit!
```

**Motivation: Path Sum Problem**

Given a binary tree, determine if there exists a path from root to leaf without any
node having value 0. Return true if such a path exists, false otherwise.

**Visual: Valid path exists**

```
Tree: [4, 0, 1, null, 7, 2, 0]
        🔴 4
       /   \
    ❌ 0   🔵 1
         /   \
      🟢 7   🔴 2
              \
            ❌ 0

Valid path: 4 → 1 → 2 ✓
(No zeros in path)

Invalid paths:
- 4 → 0 ❌ (contains 0)
- 4 → 1 → 7 → ... (but 7 has no valid children)
- 4 → 1 → 2 → 0 ❌ (contains 0)
```

**Visual: No valid path exists**

```
Tree: [4, 0, 1, null, 0, 2, 0]
        🔴 4
       /   \
    ❌ 0   🔵 1
         /   \
      ❌ 0   🔴 2
              \
            ❌ 0

All paths contain 0:
- 4 → 0 ❌
- 4 → 1 → 0 ❌
- 4 → 1 → 2 → 0 ❌

Result: False (no valid path)
```

**Algorithm: Basic version (return true/false)**

**Visual: Decision tree**

```
At each node:
1. Check constraint: Is node.val == 0? → Return False
2. Base case: Is leaf? → Return True (valid path found!)
3. Try left subtree
4. If left returns True → Return True
5. Try right subtree
6. If right returns True → Return True
7. Both failed → Return False
```

**Key insight**
If a solution exists, it's in either the left or right subtree. Try left first, if it
succeeds return true. Otherwise try right. If both fail, backtrack (return false).

**Code snippet: Basic version**

```python
def canReachLeaf(root):
    """
    Check if valid path exists (no zeros) from root to leaf.

    Algorithm:
    1. Base case: If root is None or root.val == 0 → False
    2. Base case: If leaf node → True (valid path!)
    3. Try left subtree → if True, return True
    4. Try right subtree → if True, return True
    5. Both failed → return False
    """
    if not root or root.val == 0:
        return False  # Constraint violation or empty tree

    if not root.left and not root.right:
        return True  # Leaf node reached - valid path!

    # Try left subtree first
    if canReachLeaf(root.left):
        return True

    # Try right subtree
    if canReachLeaf(root.right):
        return True

    # Both subtrees failed
    return False
```

**Building the Path**

Now let's also build the actual path if it exists. We pass a `path` list to store
nodes in the valid path.

**Visual: Building path step-by-step**

```
Tree: [4, 0, 1, null, 7, 3, 2, null, null, null, 0]
        🔴 4
       /   \
    ❌ 0   🔵 1
         /   \
      🟢 7   🔴 3
              / \
          🟡 2  ❌ 0

Step 1: Add 4 → path = [4]
Step 2: Try left (0) → Invalid, backtrack
Step 3: Try right (1) → Valid, add 1 → path = [4, 1]
Step 4: Try 1's left (7) → Valid, add 7 → path = [4, 1, 7]
Step 5: 7 is leaf but no valid path → Remove 7 → path = [4, 1]
Step 6: Try 1's right (3) → Valid, add 3 → path = [4, 1, 3]
Step 7: Try 3's left (2) → Valid, add 2 → path = [4, 1, 3, 2]
Step 8: 2 is leaf → Found valid path! Return True
Step 9: Backtrack, remove 3 → path = [4, 1, 3] (but we already returned)

Final path: [4, 1, 2] ✓
```

**Visual: Backtracking with path**

```
Current path: [4, 1, 3]
              ↓
Try 3's left (2):
  - Add 2 → [4, 1, 3, 2]
  - 2 is leaf → Found! Return True ✓

If 2 wasn't valid:
  - Remove 2 → [4, 1, 3] (backtrack)
  - Try 3's right (0):
    - Invalid → Remove 3 → [4, 1] (backtrack)
    - Continue exploring...
```

**Key insight: Undo (pop) when backtracking**
When a path doesn't lead to a solution, we must remove the node from our path list
before trying another branch. This is the "undo" step in backtracking.

**Code snippet: Building path**

```python
def leafPath(root, path):
    """
    Find valid path (no zeros) and build the path list.

    Algorithm:
    1. Check constraint: If root is None or root.val == 0 → False
    2. Add current node to path
    3. Base case: If leaf → True (valid path found!)
    4. Try left subtree → if True, return True
    5. Try right subtree → if True, return True
    6. Backtrack: Remove current node from path (pop)
    7. Return False (no valid path in this branch)
    """
    if not root or root.val == 0:
        return False  # Constraint violation

    path.append(root.val)  # Add current node to path

    if not root.left and not root.right:
        return True  # Leaf reached - valid path!

    # Try left subtree
    if leafPath(root.left, path):
        return True

    # Try right subtree
    if leafPath(root.right, path):
        return True

    # Backtrack: remove current node (undo choice)
    path.pop()
    return False
```

**Visual: Why pop is needed**

```
Path so far: [4, 1, 3]
Current node: 3

Try 3's left (2):
  path = [4, 1, 3, 2]
  Found leaf! → Return True ✓
  (Path stays as [4, 1, 3, 2] - but actually we'd clean up)

If 2 wasn't valid:
  path = [4, 1, 3, 2]
  Backtrack: path.pop() → [4, 1, 3]
  Try 3's right (0):
    Invalid → Backtrack: path.pop() → [4, 1]
  Continue with other branches...
```

**Time and space**

- **Time**: O(n) - visit every node in worst case
- **Space**: O(h) where h is height
  - Recursion stack: O(h)
  - Path list: O(h) maximum (one path from root to leaf)

**Interview narrative**
"Backtracking tries all possible solutions and backtracks from dead-ends. For this tree
problem, I check constraints at each node (no zeros), try left subtree first, then right.
If a path works, return true. If not, backtrack by removing the node from the path and
trying other branches. Time is O(n) to visit all nodes, space is O(h) for recursion and
the path list."

**Closing notes**
Backtracking applies beyond trees. We'll see it in subsets, permutations, and
combination problems. The pattern: try a choice, explore recursively, undo (backtrack)
if it doesn't work.

**Full runnable code**
See `code/backtracking.py`.

---

### 23) Heap Properties

**Concept in plain English**
A heap is a specialized, tree-based data structure that implements a Priority Queue. Unlike
regular queues (FIFO), priority queues remove elements based on priority - highest priority
first, regardless of insertion order. Heaps maintain two key properties: complete binary
tree structure and heap order property.

**Visual: Queue vs Priority Queue**

```
Regular Queue (FIFO):
Enqueue: [A, B, C] → Dequeue: A, B, C (order matters)

Priority Queue:
Enqueue: [A(priority:3), B(priority:1), C(priority:2)]
Dequeue: B, C, A (priority order: 1, 2, 3)
```

**Two Types of Heaps**

**Min Heap**

- Smallest value at root (highest priority)
- Smallest value removed first
- All descendants ≥ ancestors

**Max Heap**

- Largest value at root (highest priority)
- Largest value removed first
- All descendants ≤ ancestors

**Visual: Min Heap vs Max Heap**

```
Min Heap:              Max Heap:
      🔴 14                   🔴 68
     /   \                   /   \
  🟢 19  🔵 16            🟢 65  🔵 30
   / \   / \              / \   / \
🟡21⚪26🟣19🔴68        🟡19⚪26🟣16🔴14

Root = smallest          Root = largest
Remove smallest first    Remove largest first
```

**Heap Properties**

For a binary tree to be a heap, it must satisfy two properties:

**1. Structure Property**
A complete binary tree: every level is completely filled, except possibly the lowest level,
which is filled contiguously from left to right.

**Visual: Complete vs Incomplete**

```
Complete Binary Tree (✓):    Incomplete (✗):
      🔴 14                        🔴 14
     /   \                       /   \
  🟢 19  🔵 16                🟢 19  🔵 16
   / \   / \                   / \     \
🟡21⚪26🟣19🔴68            🟡21⚪26    🔴68

All levels filled except      Missing nodes in
last level (left to right)    middle/right
```

**2. Order Property**

**Min Heap**: All descendants ≥ ancestors (recursive property)

- Root ≤ all nodes in left subtree
- Root ≤ all nodes in right subtree
- Applies recursively to every node

**Max Heap**: All descendants ≤ ancestors

- Root ≥ all nodes in left subtree
- Root ≥ all nodes in right subtree

**Visual: Order Property (Min Heap)**

```
Min Heap:
        🔴 14 (root)
       /   \
    🟢 19  🔵 16
     / \   / \
  🟡21⚪26🟣19🔴68

Order check:
- 14 ≤ 19, 16 ✓
- 14 ≤ 21, 26, 19, 68 ✓ (all descendants)
- 19 ≤ 21, 26 ✓
- 16 ≤ 19, 68 ✓

Unlike BST, duplicates allowed!
```

**Key Differences from BST**

- **BST**: Left < root < right (strict ordering)
- **Heap**: Parent ≤ children (min heap) or Parent ≥ children (max heap)
- **BST**: No duplicates typically
- **Heap**: Duplicates allowed ✓

**Implementation: Array-Based**

Heaps are drawn as trees but implemented using arrays! This is efficient and uses no
pointers.

**Visual: Tree to Array Mapping**

```
Binary Heap Tree:              Array (1-based indexing):
        🔴 14                        Index: 0  1  2  3  4  5  6  7  8  9
       /   \                        Value: [0, 14,19,16,21,26,19,68,65,30]
    🟢 19  🔵 16
     / \   / \                      Note: Index 0 unused (sentinel)
  🟡21⚪26🟣19🔴68                    Start at index 1!
   / \
🟢65🔴30

BFS order (level by level, left to right):
Level 0: 14 (index 1)
Level 1: 19 (index 2), 16 (index 3)
Level 2: 21 (index 4), 26 (index 5), 19 (index 6), 68 (index 7)
Level 3: 65 (index 8), 30 (index 9)
```

**One-Based Indexing**

We start at index 1 (not 0) to simplify parent/child calculations. Index 0 is unused
(sentinel value).

**Formulas (where i is node index)**

```
leftChild = 2 * i
rightChild = 2 * i + 1
parent = i // 2 (integer division)
```

**Visual: Finding Children and Parent**

```
Array: [0, 14, 19, 16, 21, 26, 19, 68, 65, 30]
        ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
Index:  0   1   2   3   4   5   6   7   8   9

Find children and parent of node at index 2 (value 19):

Tree view:
        🔴 14 (index 1)
       /   \
    🟢 19  🔵 16 (index 3)
(index 2)  / \
      🟡21⚪26 (index 4, 5)

For node at index 2 (value 19):
- Left child: 2 * 2 = 4 → value 21 ✓
- Right child: 2 * 2 + 1 = 5 → value 26 ✓
- Parent: 2 // 2 = 1 → value 14 ✓
```

**Visual: Why Index 1 (Not 0)?**

```
If root at index 0:
- Left child: 2 * 0 = 0 ❌ (would point to itself!)
- Right child: 2 * 0 + 1 = 1 ✓
- Parent: 0 // 2 = 0 ❌ (would point to itself!)

If root at index 1:
- Left child: 2 * 1 = 2 ✓
- Right child: 2 * 1 + 1 = 3 ✓
- Parent: 1 // 2 = 0 (sentinel, safe) ✓

Index 1 makes formulas work perfectly!
```

**Key insight**
Complete binary tree structure + contiguous array filling (BFS order) + one-based indexing
= simple parent/child formulas with no pointers needed!

**Code snippet: Basic Heap Structure**

```python
class Heap:
    """
    Binary Heap implementation using array (1-based indexing).

    Structure:
    - Index 0: Unused (sentinel)
    - Index 1+: Heap elements in BFS order

    Formulas:
    - leftChild(i) = 2 * i
    - rightChild(i) = 2 * i + 1
    - parent(i) = i // 2
    """
    def __init__(self):
        self.heap = [0]  # Index 0 unused (sentinel)

    def left_child(self, i):
        """Get left child index of node at index i."""
        return 2 * i

    def right_child(self, i):
        """Get right child index of node at index i."""
        return 2 * i + 1

    def parent(self, i):
        """Get parent index of node at index i."""
        return i // 2
```

**Time and space**

- **Structure**: O(n) space for array
- **Access**: O(1) to get root (min/max)
- **Formulas**: O(1) to calculate parent/children

**Interview narrative**
"A heap is a complete binary tree stored in an array. It maintains two properties: structure
(complete tree, filled left to right) and order (min heap: parent ≤ children, max heap:
parent ≥ children). We use 1-based indexing so parent/child formulas work: left = 2i,
right = 2i+1, parent = i//2. This gives O(1) access to root and O(1) navigation without
pointers."

**Push and Pop Operations**

**Concept in plain English**
We can read the min/max value in O(1) by accessing the root. Push and pop are more
complex but still efficient at O(log n). Push adds an element and "percolates up" to maintain
order. Pop removes the root, replaces it with the last element, then "percolates down" to
restore order.

**Push Operation**

**Visual: Pushing 17 into min heap**

```
Initial heap: [14,19,16,21,26,19,68,65,30]
        🔴 14
       /   \
    🟢 19  🔵 16
     / \   / \
  🟡21⚪26🟣19🔴68
   / \
🟢65🔴30

Step 1: Add 17 at next position (index 10)
        🔴 14
       /   \
    🟢 19  🔵 16
     / \   / \
  🟡21⚪26🟣19🔴68
   / \   /
🟢65🔴30🟠17 (new, violates order!)

Step 2: Compare 17 with parent (26)
        17 < 26 → Swap!
        🔴 14
       /   \
    🟢 19  🔵 16
     / \   / \
  🟡21🟠17🟣19🔴68
   / \   /
🟢65🔴30⚪26

Step 3: Compare 17 with new parent (19)
        17 < 19 → Swap!
        🔴 14
       /   \
    🟠 17  🔵 16
     / \   / \
  🟡21🟢19🟣19🔴68
   / \   /
🟢65🔴30⚪26

Step 4: Compare 17 with new parent (14)
        17 > 14 → Stop! ✓

Final heap: [14,17,16,21,19,19,68,65,30,26]
```

**Key insight: Percolate up (bubble up)**
When pushing, we add at the end (maintains structure), then swap up with parent until
order property is satisfied. This is O(log n) because we traverse at most the height.

**Code snippet: Push**

```python
def push(self, val):
    """
    Push value into heap and percolate up to maintain order.

    Algorithm:
    1. Append value to end of array (maintains structure property)
    2. Compare with parent
    3. If smaller (min heap), swap with parent
    4. Repeat until parent is smaller or reach root

    Time: O(log n) - height of tree
    """
    self.heap.append(val)  # Add to end (next position in complete tree)
    i = len(self.heap) - 1  # Index of new element

    # Percolate up: swap with parent while violating order property
    while i > 1 and self.heap[i] < self.heap[i // 2]:
        # Swap with parent
        self.heap[i], self.heap[i // 2] = self.heap[i // 2], self.heap[i]
        i = i // 2  # Move to parent
```

**Pop Operation**

**Visual: The wrong way**

```
Pop root (14):
        ❌ (empty)
       /   \
    🟢 19  🔵 16
     / \   / \
  🟡21⚪26🟣19🔴68
   / \
🟢65🔴30

Replace with min(19, 16) = 16:
        🔵 16
       /   \
    🟢 19  ❌ (empty) ← Structure violated!
     / \   / \
  🟡21⚪26🟣19🔴68
   / \
🟢65🔴30

Missing node at level 2! ❌
```

**Visual: The correct way**

```
Initial heap: [14,19,16,21,26,19,68,65,30]
        🔴 14 (to remove)
       /   \
    🟢 19  🔵 16
     / \   / \
  🟡21⚪26🟣19🔴68
   / \
🟢65🔴30

Step 1: Store root (14) to return
Step 2: Replace root with last element (30)
        🔴 30 (violates order!)
       /   \
    🟢 19  🔵 16
     / \   / \
  🟡21⚪26🟣19🔴68
   /
🟢65

Step 3: Percolate down: swap with min(19, 16) = 16
        🔵 16
       /   \
    🟢 19  🔴 30
     / \   / \
  🟡21⚪26🟣19🔴68
   /
🟢65

Step 4: Continue: swap 30 with min(19, 68) = 19
        🔵 16
       /   \
    🔴 30  🟢 19
     / \   / \
  🟡21⚪26🟣19🔴68
   /
🟢65

Step 5: Continue: swap 30 with min(21, 26) = 21
        🔵 16
       /   \
    🟡 21  🟢 19
     / \   / \
  🔴30⚪26🟣19🔴68
   /
🟢65

Step 6: 30 has no children → Stop! ✓

Final heap: [16,21,19,30,26,19,68,65]
```

**Key insight: Percolate down (bubble down)**
When popping, we replace root with last element (maintains structure), then swap down
with smaller child until order property is satisfied. This is O(log n).

**Code snippet: Pop**

```python
def pop(self):
    """
    Pop and return root (min for min-heap, max for max-heap).

    Algorithm:
    1. If empty, return None
    2. If only one element, pop and return
    3. Store root value
    4. Replace root with last element (maintains structure)
    5. Percolate down: swap with smaller child until order satisfied

    Time: O(log n) - height of tree
    """
    if len(self.heap) == 1:
        return None  # Empty heap
    if len(self.heap) == 2:
        return self.heap.pop()  # Only one element

    res = self.heap[1]  # Store root value
    # Move last element to root (maintains structure property)
    self.heap[1] = self.heap.pop()
    i = 1

    # Percolate down: swap with smaller child while violating order
    while 2 * i < len(self.heap):  # While has left child
        # Check if has right child and right < left
        if (2 * i + 1 < len(self.heap) and
            self.heap[2 * i + 1] < self.heap[2 * i] and
            self.heap[i] > self.heap[2 * i + 1]):
            # Swap with right child (smaller)
            self.heap[i], self.heap[2 * i + 1] = self.heap[2 * i + 1], self.heap[i]
            i = 2 * i + 1
        elif self.heap[i] > self.heap[2 * i]:
            # Swap with left child
            self.heap[i], self.heap[2 * i] = self.heap[2 * i], self.heap[i]
            i = 2 * i
        else:
            break  # Order property satisfied

    return res
```

**Visual: Pop cases**

```
Case 1: No children
  Node is leaf → Stop ✓

Case 2: Only left child (complete tree guarantees this)
  Compare with left child
  If node > left → Swap, continue
  Else → Stop ✓

Case 3: Two children
  Compare with min(left, right)
  If node > min → Swap with min, continue
  Else → Stop ✓
```

**Time Complexity Summary**

| Operation   | Time Complexity |
| ----------- | --------------- |
| Get Min/Max | O(1)            |
| Push        | O(log n)        |
| Pop         | O(log n)        |

**Interview narrative**
"To push, I add the element at the end to maintain structure, then percolate up by swapping
with parent until order is satisfied - O(log n). To pop, I replace root with last element
to maintain structure, then percolate down by swapping with smaller child until order is
satisfied - O(log n). Both operations maintain the complete binary tree structure and heap
order property."

**Heapify: Building Heap in O(n) Time**

**Concept in plain English**
Building a heap by pushing elements one-by-one takes O(n log n) time. Heapify builds a heap
from an array in O(n) time by starting from the first non-leaf node and percolating down
backwards to the root. We skip leaf nodes since they can't percolate down.

**Visual: Naive approach vs Heapify**

```
Naive approach (push one-by-one):
Push 14 → O(log 1)
Push 19 → O(log 2)
Push 16 → O(log 3)
...
Push n elements → O(n log n) total ❌

Heapify approach:
Start from first non-leaf (n/2)
Percolate down backwards to root
Total → O(n) ✓
```

**Visual: Why start from n/2?**

```
Complete binary tree with n elements (1-based indexing):

In a complete binary tree:
- Leaf nodes are at indices > n//2
- Non-leaf nodes are at indices ≤ n//2
- Last element's parent is at index n//2

Example with n=8 elements (indices 1-8):
        🔴 1
       /   \
    🟢 2  🔵 3
     / \   / \
  🟡4⚪5🟣6🔴7
   /
🟢8

Non-leaf nodes: 1, 2, 3, 4 (indices ≤ 8//2 = 4)
Leaf nodes: 5, 6, 7, 8 (indices > 4)

Start heapify from index 4 (first non-leaf)
Work backwards: 4 → 3 → 2 → 1 (root)
Skip indices 5-8 (they're leaves, can't percolate down)
```

**Algorithm: Heapify**

**Visual: Heapify process**

```
Input array: [14,19,16,21,26,19,68,65,30]
Step 1: Move 0th element to end
        [19,16,21,26,19,68,65,30,14]

Step 2: Find first non-leaf = n//2 = 8//2 = 4
        Start percolating from index 4 backwards

Index 4 (value 26): Already a leaf? No, check children
  - Left child: 2*4=8 (value 30)
  - Right child: 2*4+1=9 (out of bounds)
  - 26 < 30? Yes → No swap needed ✓

Index 3 (value 21): Check children
  - Left child: 2*3=6 (value 68)
  - Right child: 2*3+1=7 (value 65)
  - min(68,65) = 65
  - 21 < 65? Yes → No swap needed ✓

Index 2 (value 16): Check children
  - Left child: 2*2=4 (value 26)
  - Right child: 2*2+1=5 (value 19)
  - min(26,19) = 19
  - 16 < 19? Yes → No swap needed ✓

Index 1 (value 19): Check children
  - Left child: 2*1=2 (value 16)
  - Right child: 2*1+1=3 (value 21)
  - min(16,21) = 16
  - 19 > 16? Yes → Swap with left child!

After swap at index 1:
  [16,19,21,26,19,68,65,30,14]

Continue percolating 19 down from index 2...
```

**Key insight**
We start from the first non-leaf node (n/2) and work backwards to the root, percolating
down at each step. Leaf nodes are skipped because they have no children to swap with.
This builds the heap bottom-up, which is more efficient than top-down (push one-by-one).

**Code snippet: Heapify**

```python
def heapify(self, arr):
    """
    Build heap from array in O(n) time.

    Algorithm:
    1. Move 0th element to end (for 1-based indexing)
    2. Start from first non-leaf node (n//2)
    3. Work backwards to root (index 1)
    4. At each node, percolate down (same as pop)

    Why O(n)?
    - Only non-leaf nodes percolate (n/2 nodes)
    - Nodes at level h percolate at most h levels
    - Mathematical sum simplifies to O(n)

    Time: O(n) - linear time!
    Space: O(1) - in-place
    """
    # Move 0th element to end (for 1-based indexing)
    arr.append(arr[0])
    self.heap = arr

    # Start from first non-leaf node (n//2)
    # Work backwards to root (index 1)
    cur = (len(self.heap) - 1) // 2

    while cur > 0:
        # Percolate down from current node
        i = cur
        while 2 * i < len(self.heap):  # While has left child
            # Check if has right child and right < left
            if (2 * i + 1 < len(self.heap) and
                self.heap[2 * i + 1] < self.heap[2 * i] and
                self.heap[i] > self.heap[2 * i + 1]):
                # Swap with right child (smaller)
                self.heap[i], self.heap[2 * i + 1] = self.heap[2 * i + 1], self.heap[i]
                i = 2 * i + 1
            elif self.heap[i] > self.heap[2 * i]:
                # Swap with left child
                self.heap[i], self.heap[2 * i] = self.heap[2 * i], self.heap[i]
                i = 2 * i
            else:
                break  # Order property satisfied

        cur -= 1  # Move to previous non-leaf node
```

**Visual: Why O(n) instead of O(n log n)?**

```
Tree levels and work:
Level 3 (leaves): n/2 nodes, 0 percolations (skipped) ✓
Level 2: n/4 nodes, at most 1 percolation each
Level 1: n/8 nodes, at most 2 percolations each
Level 0 (root): 1 node, at most log n percolations

Total work:
n/4 * 1 + n/8 * 2 + n/16 * 3 + ... + 1 * log n

Mathematical sum: O(n) ✓

Intuition: Most nodes are leaves (no work), and nodes that
do work are closer to leaves (less percolation needed).
```

**Time Complexity Summary**

| Operation   | Time Complexity |
| ----------- | --------------- |
| Get Min/Max | O(1)            |
| Push        | O(log n)        |
| Pop         | O(log n)        |
| Heapify     | O(n)            |

**Interview narrative**
"To build a heap from an array, I can push elements one-by-one for O(n log n), or use heapify
for O(n). Heapify starts from the first non-leaf node (n/2) and percolates down backwards to
the root. We skip leaf nodes since they can't percolate. The math works out to O(n) because
most nodes are leaves (no work), and nodes that do work are near leaves (less percolation)."

**Full runnable code**
See `code/heap.py`.

---

### 24) Hash Usage

**Concept in plain English**
Hash maps and hash sets implement the Map and Set interfaces using hashing. Hash maps store
key-value pairs with O(1) average-time operations. Hash sets store unique keys only. They're
unordered (unlike TreeMap/TreeSet) but faster. When you see "unique", "count", or "frequency"
in a problem, think hash map/set!

**Visual: Set vs Map**

```
HashSet (keys only):
{"alice", "brad", "collin"}
- Unique values only
- Fast membership check

HashMap (key-value pairs):
{"alice": 123, "brad": 345, "collin": 678}
- Maps keys to values
- Fast lookup by key
```

**Motivation: When to Use Hash Maps**

**Visual: Operation Comparison**

| Operation         | TreeMap  | HashMap    | Array      |
| ----------------- | -------- | ---------- | ---------- |
| Insert            | O(log n) | O(1) avg ✓ | O(n)       |
| Remove            | O(log n) | O(1) avg ✓ | O(n)       |
| Search            | O(log n) | O(1) avg ✓ | O(log n)\* |
| Inorder Traversal | O(n)     | -          | -          |

\*Array: O(log n) if sorted, O(n) if unsorted

**Key trade-offs**

- **TreeMap**: Ordered (sorted keys), O(log n) operations
- **HashMap**: Unordered, O(1) average operations ✓
- **Array**: Ordered, O(n) insert/remove

**Visual: TreeMap vs HashMap**

```
TreeMap (ordered):
{"alice": 123, "brad": 345, "collin": 678}
Keys sorted alphabetically ✓
Iterate in order: O(n)

HashMap (unordered):
{"collin": 678, "alice": 123, "brad": 345}
Keys in random order ❌
Iterate: O(n) but unsorted
To sort: O(n log n) ❌
```

**Frequency Counting: The Killer Use Case**

Hash maps excel at counting frequencies. Perfect for problems with "count" or "frequency".

**Visual: Counting Name Frequencies**

```
Input array: ["alice", "brad", "collin", "brad", "dylan", "kim"]

Process:
1. "alice" → Not in map → Add {"alice": 1}
2. "brad" → Not in map → Add {"alice": 1, "brad": 1}
3. "collin" → Not in map → Add {"alice": 1, "brad": 1, "collin": 1}
4. "brad" → Already in map → Increment {"alice": 1, "brad": 2, "collin": 1}
5. "dylan" → Not in map → Add {"alice": 1, "brad": 2, "collin": 1, "dylan": 1}
6. "kim" → Not in map → Add {"alice": 1, "brad": 2, "collin": 1, "dylan": 1, "kim": 1}

Result:
Key      | Value
---------|------
alice    | 1
brad     | 2
collin   | 1
dylan    | 1
kim      | 1
```

**Code snippet: Frequency Counting**

```python
def count_frequency(arr):
    """
    Count frequency of each element using hash map.

    Algorithm:
    1. Iterate through array
    2. For each element:
       - If not in map: add with count 1
       - If in map: increment count
    3. Return frequency map

    Time: O(n) - single pass through array
    Space: O(n) - store unique elements
    """
    count_map = {}

    for item in arr:
        if item not in count_map:
            count_map[item] = 1  # First occurrence
        else:
            count_map[item] += 1  # Increment frequency

    return count_map

# Shorter version using get() with default
def count_frequency_short(arr):
    count_map = {}
    for item in arr:
        count_map[item] = count_map.get(item, 0) + 1
    return count_map
```

**Visual: Why HashMap is Faster for Counting**

```
TreeMap approach:
For each element:
  - Insert/search: O(log n)
  - Total: O(n log n) ❌

HashMap approach:
For each element:
  - Insert/search: O(1) average
  - Total: O(n) ✓

Example with n=1000:
- TreeMap: 1000 * log(1000) ≈ 10,000 operations
- HashMap: 1000 * 1 = 1,000 operations
10x faster! ✓
```

**When to Use Hash Maps/Sets**

**Keywords that suggest hashing:**

- "unique" → HashSet
- "count" → HashMap
- "frequency" → HashMap
- "appears X times" → HashMap
- "no duplicates" → HashSet

**Common patterns:**

1. **Frequency counting**: Count occurrences of elements
2. **Lookup optimization**: O(1) instead of O(n) search
3. **Duplicate detection**: Check if element exists
4. **Two Sum pattern**: Store complements for O(1) lookup

**Time and space**

- **Time**: O(1) average for insert, remove, search
  - Worst case: O(n) if all keys hash to same bucket (rare)
  - Interview assumption: O(1) constant time
- **Space**: O(n) where n is number of unique keys

**Interview narrative**
"Hash maps provide O(1) average-time operations for insert, remove, and search, making them
ideal for frequency counting and fast lookups. Unlike TreeMap, they're unordered, but the
speed advantage is worth it. When I see 'count', 'frequency', or 'unique' in a problem, I
immediately think hash map or hash set. For frequency counting, I iterate once, checking if
the key exists and incrementing its count - O(n) time total."

**Full runnable code**
See `code/hashing.py`.

---

**Hash Implementation**

**Concept in plain English**
Hash maps are implemented using arrays with a hash function that converts keys to array indices.
We handle collisions (multiple keys mapping to same index) using chaining or open addressing.
When the array gets half full, we resize and rehash all elements. Understanding this helps with
systems design and distributed systems, though you rarely implement from scratch in interviews.

**Visual: Hash Map Structure**

```
HashMap under the hood:
Array: [None, None, None, None, ...]
         ↓     ↓     ↓     ↓
       Index: 0     1     2     3

Key-value pairs:
"Alice" → hash("Alice") → index → store ("Alice", "NYC")
"Brad" → hash("Brad") → index → store ("Brad", "Chicago")
```

**Hash Function**

**Visual: How Hash Function Works**

```
Key: "Alice"

Step 1: Convert each character to ASCII
  'A' = 65
  'l' = 108
  'i' = 105
  'c' = 99
  'e' = 101

Step 2: Sum ASCII codes
  65 + 108 + 105 + 99 + 101 = 478

Step 3: Use modulo to get valid index
  Array size = 2
  478 % 2 = 0
  Store at index 0 ✓
```

**Key insight**
Hash function: converts key → integer → array index using modulo. Same key always produces
same index (deterministic). Different keys may produce same index (collision).

**Code snippet: Hash Function**

```python
def hash(self, key):
    """
    Convert key to array index.

    Algorithm:
    1. Sum ASCII codes of all characters
    2. Use modulo to get valid index

    Same key → same index (deterministic)
    Different keys → may collide (same index)
    """
    index = 0
    for c in key:
        index += ord(c)  # Get ASCII code
    return index % self.capacity  # Valid index
```

**Resizing and Rehashing**

**Visual: When to Resize**

```
HashMap state:
Capacity: 2
Size: 0 (empty)

Insert "Alice": size = 1
  Capacity: 2, Size: 1
  Load factor: 1/2 = 0.5 (half full)
  → Resize before next insert! ✓

After resize:
Capacity: 4 (doubled)
Size: 1
Rehash "Alice" to new position
```

**Key insight**
Resize when array is half full (load factor = 0.5) to minimize collisions. Double capacity,
then rehash all existing elements because their positions change with new capacity.

**Visual: Rehashing Process**

```
Before resize (capacity = 2):
"Alice" → hash("Alice") = 478 % 2 = 0
Array: [("Alice", "NYC"), None]

After resize (capacity = 4):
"Alice" → hash("Alice") = 478 % 4 = 2
Array: [None, None, ("Alice", "NYC"), None]

Position changed! Must rehash all elements.
```

**Collisions**

**Visual: Collision Example**

```
Array capacity: 8
"Alice" → hash = 478 % 8 = 6 → Index 6 ✓
"Collin" → hash = 33 % 8 = 1 → Index 1 ✓
"Brad" → hash = 27 % 8 = 3 → Index 3 ✓

Wait, let's recalculate:
"Collin": C(67) + o(111) + l(108) + l(108) + i(105) + n(110) = 609
609 % 8 = 1 ✓

But what if "Collin" also hashes to 6?
"Collin" → hash = 609 % 8 = 1 (different)
But if it was 614 % 8 = 6 → Collision with "Alice"!
```

**Two Collision Resolution Strategies**

**1. Chaining (Linked Lists)**

**Visual: Chaining**

```
Index 6: ["Alice" → "NYC"] → ["Collin" → "Seattle"]
         (Linked list of pairs)

Lookup "Collin":
1. Hash to index 6
2. Traverse linked list at index 6
3. Find "Collin" → return "Seattle"

Time: O(1) average, O(n) worst (all keys collide)
```

**2. Open Addressing (Linear Probing)**

**Visual: Open Addressing**

```
Array: [None, None, None, None, None, None, None, None]
        0     1     2     3     4     5     6     7

Insert "Alice" → index 6
Array: [None, None, None, None, None, None, ("Alice", "NYC"), None]

Insert "Collin" → index 6 (collision!)
  Try index 7 → Empty ✓
Array: [None, None, None, None, None, None, ("Alice", "NYC"), ("Collin", "Seattle")]

Lookup "Collin":
1. Hash to index 6
2. Check index 6 → "Alice" (not "Collin")
3. Check next index 7 → "Collin" ✓

Time: O(1) average, O(n) worst
```

**Key insight**
Chaining stores multiple pairs at same index using linked lists. Open addressing finds next
available slot. Chaining is simpler; open addressing is more efficient with few collisions
but limited by array size.

**Code Implementation: Open Addressing**

**Visual: Pair Class**

```python
class Pair:
    """Stores key-value pair."""
    def __init__(self, key, val):
        self.key = key
        self.val = val
```

**Code snippet: HashMap with Open Addressing**

```python
class HashMap:
    def __init__(self):
        self.size = 0  # Number of key-value pairs
        self.capacity = 2  # Array size
        self.map = [None, None]  # Array of Pairs

    def hash(self, key):
        """Convert key to array index."""
        index = 0
        for c in key:
            index += ord(c)
        return index % self.capacity

    def get(self, key):
        """
        Get value for key.

        Algorithm:
        1. Hash key to get starting index
        2. Check if key matches
        3. If not, check next index (open addressing)
        4. Wrap around using modulo
        """
        index = self.hash(key)

        while self.map[index] != None:
            if self.map[index].key == key:
                return self.map[index].val
            index += 1
            index = index % self.capacity  # Wrap around
        return None

    def put(self, key, val):
        """
        Insert or update key-value pair.

        Algorithm:
        1. Hash key to get starting index
        2. Three cases:
           a. Index is vacant → insert
           b. Index has same key → update
           c. Index has different key → try next (open addressing)
        3. Resize if half full
        """
        index = self.hash(key)

        while True:
            if self.map[index] == None:
                # Vacant slot → insert
                self.map[index] = Pair(key, val)
                self.size += 1
                if self.size >= self.capacity // 2:
                    self.rehash()  # Resize before next insert
                return
            elif self.map[index].key == key:
                # Same key → update value
                self.map[index].val = val
                return

            # Collision → try next index
            index += 1
            index = index % self.capacity

    def rehash(self):
        """
        Resize array and rehash all elements.

        Algorithm:
        1. Double capacity
        2. Create new array
        3. Rehash all existing pairs (positions change!)
        """
        self.capacity = 2 * self.capacity
        new_map = [None] * self.capacity

        old_map = self.map
        self.map = new_map
        self.size = 0

        # Rehash all existing pairs
        for pair in old_map:
            if pair:
                self.put(pair.key, pair.val)
```

**Visual: Why Prime Size?**

```
Why use prime capacity?

Example: Capacity = 8 (composite)
  Keys: 16, 24, 32, 40, ...
  All hash to: 16 % 8 = 0, 24 % 8 = 0, 32 % 8 = 0
  → Many collisions! ❌

Example: Capacity = 7 (prime)
  Keys: 16, 24, 32, 40, ...
  Hash to: 16 % 7 = 2, 24 % 7 = 3, 32 % 7 = 4
  → Better distribution! ✓

Prime numbers reduce collisions because they're only
divisible by 1 and themselves.
```

**Time Complexity**

| Operation | Average | Worst Case |
| --------- | ------- | ---------- |
| Insert    | O(1)    | O(n)       |
| Remove    | O(1)    | O(n)       |
| Search    | O(1)    | O(n)       |

**Key insight**
Average case is O(1) with good hash function and low collisions. Worst case O(n) happens
when all keys collide (bad hash function or malicious input).

**Interview narrative**
"Hash maps use arrays with a hash function converting keys to indices. We handle collisions
using chaining (linked lists) or open addressing (next available slot). When half full, we
resize and rehash all elements since positions change. Average operations are O(1), but worst
case is O(n) if all keys collide. Prime capacity helps reduce collisions."

**Full runnable code**
See `code/hashing.py` for complete implementation.

---

### 25) Introduction to Graphs

**Concept in plain English**
A graph is a data structure with nodes (vertices) connected by edges (pointers). Unlike trees,
graphs have no restrictions on connections - nodes can connect to any number of other nodes,
and edges can form cycles. We've seen graphs before: trees and linked lists are special cases
of directed graphs.

**Visual: What is a Graph?**

```
Graph example:
    🔴 A
   /   \
🟢 B   🔵 C
   \   /
    🟡 D

Vertices: A, B, C, D
Edges: A→B, A→C, B→D, C→D

No restrictions:
- Can have cycles ✓
- Variable number of edges per node ✓
- Can be disconnected ✓
```

**Graph Terminology**

- **Vertices (V)**: Nodes in the graph
- **Edges (E)**: Connections between vertices
- **Complete Graph**: Graph with V vertices and V² edges (every vertex connected to every other)

**Visual: Complete Graph**

```
Complete graph with 3 vertices:
    🔴 A
   / | \
  /  |  \
🟢 B---🔵 C

Vertices: 3 (A, B, C)
Edges: 6 (A→B, A→C, B→A, B→C, C→A, C→B)
Maximum edges: V² = 9 (if self-loops allowed)
```

**Key insight**
Maximum edges: E ≤ V² (assuming no duplicates). Each vertex can connect to every other vertex
(including itself in some cases).

**Directed vs Undirected**

**Visual: Directed vs Undirected**

```
Directed Graph (edges have direction):
    🔴 A → 🟢 B
    ↓       ↑
    🔵 C ← 🟡 D

A→B exists, but B→A may not
Edges are one-way

Undirected Graph (edges are bidirectional):
    🔴 A — 🟢 B
    |       |
    🔵 C — 🟡 D

A—B means A↔B (both directions)
Edges are two-way
```

**Key insight**

- **Directed**: Edges have direction (A→B ≠ B→A)
- **Undirected**: Edges are bidirectional (A—B = A↔B)
- Trees and linked lists are directed graphs (parent→child, prev→next)

**Graph Representations**

Graphs can be represented in three common ways:

**1. Matrix (2D Grid)**

**Visual: Matrix Representation**

```
Matrix:
grid = [[0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]]

Rows and columns represent positions.
0s are vertices, can move up/down/left/right.
Connected 0s form connected components (graph).

Space: O(n * m) where n=rows, m=columns
```

**Code snippet: Matrix**

```python
grid = [[0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]]

# Access element at row 1, column 0
value = grid[1][0]  # Returns 1

# Traverse: can move up, down, left, right
# Connected 0s form a graph
```

**2. Adjacency Matrix**

**Visual: Adjacency Matrix**

```
adjMatrix = [[0, 0, 0, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 1, 0, 0]]

Index represents vertex:
- adjMatrix[0] = vertex 0
- adjMatrix[1] = vertex 1
- etc.

adjMatrix[v1][v2] = 1 means edge exists from v1→v2
adjMatrix[v1][v2] = 0 means no edge from v1→v2

Example:
adjMatrix[1][2] = 0 → No edge 1→2
adjMatrix[2][3] = 1 → Edge exists 2→3
```

**Visual: Adjacency Matrix Graph**

```
Graph represented:
    0 → 1 (no, adjMatrix[0][1] = 0)
    1 → 0 (yes, adjMatrix[1][0] = 1)
    1 → 1 (yes, adjMatrix[1][1] = 1, self-loop)
    2 → 3 (yes, adjMatrix[2][3] = 1)
    3 → 1 (yes, adjMatrix[3][1] = 1)
```

**Key insight**
Adjacency matrix uses O(V²) space - wasteful if graph has few edges. Good for dense graphs
(many edges), bad for sparse graphs (few edges).

**Code snippet: Adjacency Matrix**

```python
adjMatrix = [[0, 0, 0, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 1, 0, 0]]

# Check if edge exists from vertex 1 to vertex 2
if adjMatrix[1][2] == 1:
    print("Edge exists from 1 to 2")
else:
    print("No edge from 1 to 2")

# Space: O(V²) - square matrix
```

**3. Adjacency List**

**Visual: Adjacency List**

```
Graph:
    🔴 A → 🟢 B
    ↓       ↓
    🔵 C   🟡 D

Adjacency List:
A: [B, C]
B: [D]
C: []
D: []

Each vertex stores list of neighbors.
Only stores edges that exist!
```

**Key insight**
Adjacency list is space-efficient: O(V + E). Only stores edges that exist, perfect for sparse
graphs. Most common representation in interviews.

**Code snippet: Adjacency List**

```python
class GraphNode:
    """Node in a graph with list of neighbors."""
    def __init__(self, val):
        self.val = val
        self.neighbors = []  # List of adjacent vertices

# Create graph
A = GraphNode("A")
B = GraphNode("B")
C = GraphNode("C")
D = GraphNode("D")

# Add edges
A.neighbors = [B, C]  # A → B, A → C
B.neighbors = [D]     # B → D
C.neighbors = []      # C has no neighbors
D.neighbors = []      # D has no neighbors

# Space: O(V + E) - vertices + edges
```

**Comparison: Adjacency Matrix vs Adjacency List**

| Aspect         | Adjacency Matrix | Adjacency List |
| -------------- | ---------------- | -------------- |
| Space          | O(V²)            | O(V + E)       |
| Check edge     | O(1)             | O(degree)      |
| List neighbors | O(V)             | O(degree)      |
| Best for       | Dense graphs     | Sparse graphs  |

**Time and space**

- **Matrix**: O(n \* m) space for n×m grid
- **Adjacency Matrix**: O(V²) space
- **Adjacency List**: O(V + E) space (most efficient)

**Interview narrative**
"Graphs have vertices connected by edges. They can be directed (one-way) or undirected
(two-way). Common representations: matrix for grid problems, adjacency matrix for dense graphs
(O(V²) space), and adjacency list for sparse graphs (O(V + E) space). Adjacency list is most
common - each vertex stores a list of neighbors, only storing edges that exist."

**Full runnable code**
See `code/graphs.py`.

---

**Matrix DFS (Depth-First Search)**

**Concept in plain English**
We can traverse matrices using DFS, similar to tree traversal. In a matrix, we can move in four
directions (up, down, left, right). We use backtracking to explore all paths, marking visited
cells and unmarking when backtracking. Perfect for problems like counting paths or finding
connected components.

**Visual: Matrix Traversal**

```
Matrix:
grid = [[0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]]

From any cell (r, c), we can move:
- Up: (r-1, c)
- Down: (r+1, c)
- Left: (r, c-1)
- Right: (r, c+1)

0 = valid path, 1 = obstacle
```

**Problem: Count Unique Paths**

Count unique paths from top-left (0,0) to bottom-right (3,3) that:

- Only move along 0s
- Don't visit same cell twice

**Visual: Example Paths**

```
Matrix:
[0, 0, 0, 0]
[1, 1, 0, 0]
[0, 0, 0, 1]
[0, 1, 0, 0]

Path 1:
🔴→→→↓
     ↓
     →→🔴

Path 2:
🔴→→→↓
     ↓
     →→🔴
  (different route)

Total: 2 unique paths
```

**Base Cases**

**1. Invalid Path (return 0)**

- **Out of bounds**: Row or column < 0 or >= matrix size
- **Already visited**: Cell in visited set
- **Obstacle**: Current cell is 1

**Visual: Invalid Cases**

```
Out of bounds:
  r < 0 or r >= ROWS ❌
  c < 0 or c >= COLS ❌

Already visited:
  (r, c) in visit set ❌

Obstacle:
  grid[r][c] == 1 ❌

All return 0 (no valid path)
```

**2. Valid Path Found (return 1)**

- Reached destination: r == ROWS-1 and c == COLS-1

**Visual: Destination Reached**

```
Matrix: 4 rows, 4 columns
Destination: (3, 3)

If r == 3 and c == 3:
  → Found valid path! ✓
  → Return 1
```

**Implementation: Backtracking DFS**

**Visual: DFS Process**

```
Start at (0, 0):
1. Mark (0,0) as visited
2. Try all 4 directions:
   - Down: (1,0) → obstacle (1) ❌
   - Right: (0,1) → valid ✓
     → Recursively explore (0,1)
3. When backtracking:
   - Unmark (0,0) from visited
   - Try other paths
```

**Key insight**
We use backtracking: mark cell as visited, explore recursively, then unmark when backtracking.
This allows exploring all paths from each cell.

**Code snippet: Matrix DFS**

```python
def dfs(grid, r, c, visit):
    """
    Count unique paths from (r,c) to bottom-right using DFS.

    Algorithm:
    1. Check base cases (invalid → return 0, destination → return 1)
    2. Mark current cell as visited
    3. Try all 4 directions recursively
    4. Sum up results from all directions
    5. Unmark current cell (backtrack)
    6. Return total count

    Time: O(4^(n*m)) - exponential, 4 choices at each cell
    Space: O(n*m) - recursion stack + visited set
    """
    ROWS, COLS = len(grid), len(grid[0])

    # Base case 1: Invalid path
    if (min(r, c) < 0 or           # Out of bounds (negative)
        r == ROWS or c == COLS or  # Out of bounds (too large)
        (r, c) in visit or         # Already visited
        grid[r][c] == 1):          # Obstacle
        return 0

    # Base case 2: Reached destination
    if r == ROWS - 1 and c == COLS - 1:
        return 1  # Found valid path!

    # Mark as visited
    visit.add((r, c))

    # Try all 4 directions
    count = 0
    count += dfs(grid, r + 1, c, visit)  # Down
    count += dfs(grid, r - 1, c, visit)  # Up
    count += dfs(grid, r, c + 1, visit)  # Right
    count += dfs(grid, r, c - 1, visit)  # Left

    # Backtrack: unmark current cell
    visit.remove((r, c))

    return count

# Usage
grid = [[0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]]

visit = set()
result = dfs(grid, 0, 0, visit)  # Start from top-left
```

**Visual: Why Use Hash Set?**

```
Visited tracking options:

Option 1: List
  visit = [(0,0), (0,1), (1,1), ...]
  Check: (r,c) in visit → O(n) ❌

Option 2: Hash Set
  visit = {(0,0), (0,1), (1,1), ...}
  Check: (r,c) in visit → O(1) ✓

Option 3: 2D Boolean Array
  visited[r][c] = True/False
  Check: visited[r][c] → O(1) ✓
  But needs O(n*m) space upfront
```

**Visual: Backtracking Process**

```
Path 1 exploration:
(0,0) → (0,1) → (0,2) → (0,3) → (1,3) → (2,3) → (3,3) ✓
Visit set: {(0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (3,3)}
Return 1, backtrack...

Path 2 exploration (after backtracking):
(0,0) → (0,1) → (0,2) → (0,3) → (1,3) → (2,2) → (2,3) → (3,3) ✓
Visit set: {(0,0), (0,1), (0,2), (0,3), (1,3), (2,2), (2,3), (3,3)}
Return 1

Total: 2 paths
```

**Time and space**

- **Time**: O(4^(n\*m)) - exponential
  - At each cell, 4 choices (up, down, left, right)
  - Decision tree with branching factor 4
  - Height = n\*m (worst case)
- **Space**: O(n\*m)
  - Recursion stack: O(n\*m) worst case
  - Visited set: O(n\*m) worst case

**Interview narrative**
"To traverse a matrix with DFS, I move in four directions from each cell. I use a hash set
to track visited cells for O(1) lookup. Base cases: out of bounds, already visited, or
obstacle → return 0. Reached destination → return 1. I mark cells as visited before
recursion, then unmark when backtracking to explore all paths. Time is exponential O(4^(n*m)),
space is O(n*m) for recursion and visited set."

**Full runnable code**
See `code/graphs.py` for matrix DFS implementation.

---

**Matrix BFS (Breadth-First Search)**

**Concept in plain English**
BFS on matrices finds shortest paths efficiently. Unlike DFS (exponential), BFS processes level
by level, guaranteeing the first path found is shortest. Use a queue to process all cells at
current distance before moving to next level. Perfect for shortest path problems.

**Visual: BFS vs DFS for Shortest Path**

```
Matrix:
[0, 0, 0, 0]
[1, 1, 0, 0]
[0, 0, 0, 1]
[0, 1, 0, 0]

DFS: Explores one path completely before backtracking
  → May find longer path first ❌
  → Must explore all paths to find shortest

BFS: Explores level by level
  → First path found is shortest ✓
  → More efficient for shortest path
```

**Problem: Shortest Path Length**

Find length of shortest path from (0,0) to (3,3) that:

- Only moves along 0s
- Can move up, down, left, right

**Visual: BFS Level-by-Level**

```
Level 0: (0,0) - length 0
Level 1: (0,1), (1,0) - length 1
Level 2: (0,2), (2,0) - length 2
Level 3: (0,3), (2,1), (2,2) - length 3
Level 4: (1,2), (2,3) - length 4
Level 5: (3,3) - length 5 ✓ (destination reached!)

Shortest path length: 5
```

**Initial Setup**

**Visual: BFS Setup**

```
1. Get matrix dimensions (ROWS, COLS)
2. Create visited set (hash set for O(1) lookup)
3. Create queue (deque for O(1) operations)
4. Add starting cell (0,0) to queue and visited
5. Initialize length = 0
```

**Code snippet: Initial Setup**

```python
from collections import deque

def bfs(grid):
    ROWS, COLS = len(grid), len(grid[0])
    visit = set()  # Track visited cells
    queue = deque()  # Queue for BFS

    # Start from top-left
    queue.append((0, 0))
    visit.add((0, 0))

    length = 0  # Track path length
```

**BFS Algorithm: Level-by-Level Processing**

**Visual: Processing Each Level**

```
Queue state at each level:

Level 0: [(0,0)]
  Process (0,0) → Add neighbors to queue
  Queue: [(0,1), (1,0)]
  length = 1

Level 1: [(0,1), (1,0)]
  Process (0,1) → Add (0,2)
  Process (1,0) → Blocked (obstacle)
  Queue: [(0,2), (2,0)]
  length = 2

Continue until destination reached...
```

**Key insight**
Process all cells at current level before moving to next. This guarantees shortest path
because we explore in order of distance from start.

**Code snippet: Complete BFS**

```python
def bfs(grid):
    """
    Find shortest path length from (0,0) to bottom-right using BFS.

    Algorithm:
    1. Initialize queue with start cell, mark as visited
    2. Process level by level:
       a. Process all cells at current level
       b. For each cell, check if destination
       c. Add valid neighbors to queue
       d. Mark neighbors as visited immediately
    3. Increment length after each level
    4. Return length when destination reached

    Time: O(n*m) - visit each cell at most once
    Space: O(n*m) - queue + visited set
    """
    ROWS, COLS = len(grid), len(grid[0])
    visit = set()
    queue = deque()
    queue.append((0, 0))
    visit.add((0, 0))

    length = 0

    while queue:
        # Process all cells at current level
        for _ in range(len(queue)):
            r, c = queue.popleft()

            # Check if reached destination
            if r == ROWS - 1 and c == COLS - 1:
                return length  # Shortest path found!

            # Try all 4 directions
            neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]]  # [right, left, down, up]
            for dr, dc in neighbors:
                new_r, new_c = r + dr, c + dc

                # Check if valid (same checks as DFS)
                if (min(new_r, new_c) < 0 or
                    new_r == ROWS or new_c == COLS or
                    (new_r, new_c) in visit or
                    grid[new_r][new_c] == 1):
                    continue  # Skip invalid cells

                # Add to queue and mark as visited
                queue.append((new_r, new_c))
                visit.add((new_r, new_c))

        length += 1  # Move to next level

    return -1  # No path exists
```

**Visual: Directions Array**

```
neighbors = [[0, 1],   # Right: (r, c+1)
             [0, -1],  # Left:  (r, c-1)
             [1, 0],   # Down:  (r+1, c)
             [-1, 0]]  # Up:    (r-1, c)

For current cell (r, c):
  Right: (r + 0, c + 1) = (r, c+1)
  Left:  (r + 0, c - 1) = (r, c-1)
  Down:  (r + 1, c + 0) = (r+1, c)
  Up:    (r - 1, c + 0) = (r-1, c)
```

**Key insight: Mark visited immediately**
Mark cells as visited when adding to queue, not when processing. This prevents duplicate
entries in queue and improves efficiency.

**Visual: Why Mark Immediately?**

```
Wrong approach (mark when processing):
  Queue: [(0,1), (1,0)]
  Process (0,1) → Add (0,2), (0,0)
  Queue: [(1,0), (0,2), (0,0)] ← Duplicate!
  (0,0) already processed but added again ❌

Correct approach (mark when adding):
  Queue: [(0,1), (1,0)]
  Process (0,1) → Add (0,2) [mark], skip (0,0) [already visited]
  Queue: [(1,0), (0,2)] ← No duplicates ✓
```

**BFS vs DFS for Shortest Path**

| Aspect       | DFS                     | BFS                  |
| ------------ | ----------------------- | -------------------- |
| Path finding | Explores all paths      | Finds shortest first |
| Time         | O(4^(n\*m)) exponential | O(n\*m) linear       |
| Space        | O(n\*m) recursion       | O(n\*m) queue        |
| Best for     | Count paths, explore    | Shortest path        |

**Time and space**

- **Time**: O(n\*m) - visit each cell at most once
  - Unlike DFS, we don't revisit cells
  - Process each cell exactly once
- **Space**: O(n\*m)
  - Queue: O(n\*m) worst case (all cells)
  - Visited set: O(n\*m) worst case

**Interview narrative**
"For shortest path in a matrix, I use BFS instead of DFS. BFS processes level by level,
guaranteeing the first path found is shortest. I use a queue to process all cells at current
distance, then move to next level. I mark cells as visited when adding to queue to prevent
duplicates. Time is O(n*m) since we visit each cell once, space is O(n*m) for queue and
visited set."

**Full runnable code**
See `code/graphs.py` for matrix BFS implementation.

---

### 26) Dynamic Programming

**Concept in plain English**
_Placeholder - Details to be added_

**Key concepts**

- Memoization
- Tabulation
- Overlapping subproblems
- Optimal substructure

**Common problems**

- Climbing Stairs
- House Robber
- Longest Common Subsequence

**Full runnable code**
See `code/dynamic_programming.py` (placeholder).

---

### 27) Bit Manipulation

**Concept in plain English**
_Placeholder - Details to be added_

**Key concepts**

- Bitwise operations
- Masks
- Optimization tricks

**Common problems**

- Number of 1 Bits
- Single Number
- Power of Two

**Full runnable code**
See `code/bit_manipulation.py` (placeholder).
