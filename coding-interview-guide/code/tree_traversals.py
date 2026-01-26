"""
Depth-First Search (DFS) - Tree Traversals

DFS goes as deep as possible before backtracking. Pick a direction (left), follow
pointers down until null, then backtrack to parent and go right. Repeat until
all nodes visited.

Three DFS traversal methods:
1. Inorder: Left → Root → Right (gives sorted order for BST!)
2. Preorder: Root → Left → Right
3. Postorder: Left → Right → Root

Best implemented with recursion (can use stack iteratively).

Time: O(n) - visit every node once, regardless of height
Space: O(h) where h is height (balanced: O(log n), skewed: O(n))
"""


class TreeNode:
    """Node in a binary tree."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def preorder(root):
    """
    Preorder traversal: Root → Left → Right
    
    Visit parent node first, then left subtree, finally right subtree.
    
    Use cases:
    - Copying a tree
    - Prefix notation for expressions
    - Top-down processing
    
    Time: O(n) - visit every node
    Space: O(h) - recursion stack depth
    """
    if not root:
        return
    
    print(root.val)      # Visit root first
    preorder(root.left)  # Then left subtree
    preorder(root.right) # Then right subtree


def inorder(root):
    """
    Inorder traversal: Left → Root → Right
    
    Recursively visit all nodes in left subtree, then visit parent node,
    finally visit all nodes in right subtree.
    
    Key insight: For BST, inorder gives sorted order!
    - All values left of node are smaller
    - We visit leftmost (smallest) first
    - Then parent, then right (larger)
    
    Use cases:
    - BST: Get nodes in sorted order
    - Build sorted array from BST
    - Infix notation for expressions
    
    Time: O(n) - visit every node
    Space: O(h) - recursion stack depth
    """
    if not root:
        return
    
    inorder(root.left)   # Visit left subtree first
    print(root.val)      # Visit root (or perform operation)
    inorder(root.right)  # Visit right subtree last


def postorder(root):
    """
    Postorder traversal: Left → Right → Root
    
    Visit left subtree, then right subtree, finally parent node last.
    
    Use cases:
    - Deleting a tree (delete children before parent)
    - Postfix notation for expressions
    - Bottom-up processing
    - Calculating expressions
    
    Time: O(n) - visit every node
    Space: O(h) - recursion stack depth
    """
    if not root:
        return
    
    postorder(root.left)  # Visit left subtree first
    postorder(root.right) # Then right subtree
    print(root.val)       # Visit root last


def bfs(root):
    """
    Breadth-First Search (Level-Order Traversal).
    
    BFS prioritizes breadth - visit all nodes on one level before moving to next.
    Implemented iteratively using a queue (deque).
    
    Algorithm:
    1. Enqueue root
    2. While queue not empty:
       a. Process all nodes at current level (loop through queue size)
       b. Enqueue children (left, then right) for next level
       c. Increment level
    
    Key insight: Queue (FIFO) ensures we process level by level.
    - Remove from head (popleft)
    - Add to tail (append)
    
    Time: O(n) - visit every node exactly once
    Space: O(n) - queue stores entire level, worst case last level ~n/2 nodes
    """
    from collections import deque
    
    queue = deque()
    
    if root:
        queue.append(root)  # Initially append root
    
    level = 0
    while queue:
        print(f"Level {level}:")
        level_size = len(queue)  # Number of nodes at current level
        
        # Process all nodes at current level
        for _ in range(level_size):
            curr = queue.popleft()  # Remove from head (FIFO)
            print(curr.val)
            
            # Add children to tail (for next level)
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        
        level += 1


def levelorder_levels(root):
    """
    BFS returning levels as separate lists.
    
    Useful when you need to process each level separately.
    Same as bfs() but collects results instead of printing.
    
    Time: O(n)
    Space: O(n) - queue + result storage
    """
    from collections import deque
    
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        # Process all nodes at current level
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result


def demo():
    """
    Demonstrate DFS traversal methods.
    """
    print("=== Depth-First Search (DFS) Traversals Demo ===\n")
    
    # Create a BST:
    #       4
    #      / \
    #     3   6
    #    / \ / \
    #   2  5 5  7
    
    root = TreeNode(4)
    root.left = TreeNode(3)
    root.right = TreeNode(6)
    root.left.left = TreeNode(2)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(5)
    root.right.right = TreeNode(7)
    
    print("BST structure:")
    print("       4")
    print("      / \\")
    print("     3   6")
    print("    / \\ / \\")
    print("   2  5 5  7\n")
    
    print("Inorder (Left→Root→Right):")
    print("  ", end="")
    inorder(root)
    print("  Result: [2, 3, 4, 5, 5, 6, 7] ← Sorted! ✓\n")
    
    print("Preorder (Root→Left→Right):")
    print("  ", end="")
    preorder(root)
    print("  Result: [4, 3, 2, 6, 5, 5, 7]\n")
    
    print("Postorder (Left→Right→Root):")
    print("  ", end="")
    postorder(root)
    print("  Result: [2, 3, 5, 5, 7, 6, 4]\n")
    
    print("BFS / Level-Order (Level by Level):")
    bfs(root)
    print("  Result: [4, 3, 6, 2, 5, 5, 7] (level by level)\n")
    
    print("BFS by levels:")
    levels = levelorder_levels(root)
    for i, level in enumerate(levels):
        print(f"  Level {i}: {level}")
    
    print("\n=== Key Insights ===")
    print("DFS (Depth-First):")
    print("  1. Goes deep before backtracking")
    print("  2. Inorder: Left→Root→Right (sorted for BST!)")
    print("  3. Preorder: Root→Left→Right (root first)")
    print("  4. Postorder: Left→Right→Root (root last)")
    print("  5. Best implemented with recursion")
    print("  6. Space: O(h) where h is height")
    print("\nBFS (Breadth-First):")
    print("  1. Visits level by level")
    print("  2. Uses queue (FIFO) - remove from head, add to tail")
    print("  3. Implemented iteratively")
    print("  4. Space: O(n) - queue stores entire level")
    print("\nBoth: O(n) time - visit every node once")


if __name__ == "__main__":
    demo()
