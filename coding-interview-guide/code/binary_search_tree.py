"""
Binary Search Trees (BST)

BSTs are a variation of binary trees with a sorted property:
- Every node in left subtree < root
- Every node in right subtree > root
- This property applies recursively to every node

Motivation:
- Search: O(log n) - like binary search on sorted array
- Insert: O(log n) - better than sorted array (O(n))
- Delete: O(log n) - better than sorted array (O(n))

Time: O(log n) balanced, O(n) worst (skewed tree)
Space: O(h) where h is height (recursion stack)
"""


class TreeNode:
    """Node in a binary search tree."""
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def search(root, target):
    """
    Search for target in BST using recursion.
    
    Algorithm (similar to binary search):
    1. Base case: If root is None, return False (target not found)
    2. If target > root.val, search right subtree (larger values)
    3. If target < root.val, search left subtree (smaller values)
    4. If target == root.val, return True (target found)
    
    The return value of recursive call becomes the return value of current call.
    
    Time: O(log n) balanced, O(n) worst (skewed)
    Space: O(h) recursion stack depth
    """
    # Base case 1: Reached null node, target doesn't exist
    if not root:
        return False
    
    # Compare target with current node
    if target > root.val:
        # Target is greater → search right subtree (eliminate left)
        return search(root.right, target)
    elif target < root.val:
        # Target is smaller → search left subtree (eliminate right)
        return search(root.left, target)
    else:
        # Base case 2: Found target!
        return True


def insert(root, val):
    """
    Insert a new node and return the root of the BST.
    
    Algorithm:
    1. If current node is null, return new node with value val (base case)
    2. If value > current node, recursively insert into right subtree
    3. If value < current node, recursively insert into left subtree
    4. Return current node after recursive call
    
    Key insight: We always insert at a leaf position. We traverse until we
    find a null position, then create a new node there.
    
    Time: O(log n) balanced, O(n) worst (skewed)
    Space: O(log n) balanced, O(n) worst (recursion stack)
    """
    # Base case: reached null position, create new leaf node
    if not root:
        return TreeNode(val)
    
    # Recursive case: traverse to find insertion position
    if val > root.val:
        # Value greater → insert into right subtree
        root.right = insert(root.right, val)
    elif val < root.val:
        # Value smaller → insert into left subtree
        root.left = insert(root.left, val)
    # If val == root.val, we could handle duplicates here if needed
    
    return root  # Return current node (maintains tree structure)


def min_value_node(root):
    """
    Return the minimum value node of the BST.
    
    The minimum is always the leftmost node in the subtree.
    This is used to find the in-order successor for deletion.
    
    Time: O(h) where h is height
    Space: O(1)
    """
    curr = root
    while curr and curr.left:
        curr = curr.left  # Keep going left until null
    return curr


def remove(root, val):
    """
    Remove a node and return the root of the BST.
    
    Two cases to consider:
    1. Node has 0 or 1 child: Replace with child (or None)
    2. Node has 2 children: Replace with in-order successor
    
    Algorithm:
    1. If root is None, return None (not found)
    2. If val > root.val, remove from right subtree
    3. If val < root.val, remove from left subtree
    4. If val == root.val (found node to delete):
       a. 0 or 1 child → return the other child (or None)
       b. 2 children → replace value with successor, delete successor
    
    Time: O(log n) balanced, O(n) worst
    Space: O(log n) balanced, O(n) worst (recursion stack)
    """
    if not root:
        return None  # Base case: node not found
    
    if val > root.val:
        # Target greater → remove from right subtree
        root.right = remove(root.right, val)
    elif val < root.val:
        # Target smaller → remove from left subtree
        root.left = remove(root.left, val)
    else:
        # Found node to delete
        if not root.left:
            # Case 1a: No left child → replace with right child (or None)
            return root.right
        elif not root.right:
            # Case 1b: No right child → replace with left child
            return root.left
        else:
            # Case 2: Has 2 children
            # Find in-order successor (leftmost in right subtree)
            min_node = min_value_node(root.right)
            
            # Replace current node's value with successor's value
            root.val = min_node.val
            
            # Delete the duplicate successor from right subtree
            root.right = remove(root.right, min_node.val)
    
    return root


def validate_bst(root, min_val=float('-inf'), max_val=float('inf')):
    """
    Validate if tree is a valid BST.
    
    BST property: For each node, all left subtree values < node < all right subtree values
    
    Algorithm:
    1. Check if current node value is within valid range
    2. Recursively validate left subtree (max = current node)
    3. Recursively validate right subtree (min = current node)
    
    Time: O(n) - visit each node once
    Space: O(h) recursion stack
    """
    if not root:
        return True
    
    # Check if current node violates BST property
    if root.val <= min_val or root.val >= max_val:
        return False
    
    # Validate left subtree (all values must be < root.val)
    # Validate right subtree (all values must be > root.val)
    return (validate_bst(root.left, min_val, root.val) and
            validate_bst(root.right, root.val, max_val))


def find_min(root):
    """
    Find minimum value in BST.
    
    Minimum is always the leftmost node.
    
    Time: O(h) where h is height
    Space: O(1) iterative, O(h) recursive
    """
    while root and root.left:
        root = root.left
    return root.val if root else None


def find_max(root):
    """
    Find maximum value in BST.
    
    Maximum is always the rightmost node.
    
    Time: O(h) where h is height
    Space: O(1) iterative, O(h) recursive
    """
    while root and root.right:
        root = root.right
    return root.val if root else None


def demo():
    """
    Demonstrate BST operations: search, insert, remove.
    """
    print("=== Binary Search Tree Demo ===\n")
    
    # Create BST: [4]
    root = TreeNode(4)
    print("Initial BST: [4]")
    print("       4\n")
    
    # Insert 6
    root = insert(root, 6)
    print("After inserting 6: [4, null, 6]")
    print("       4")
    print("        \\")
    print("         6\n")
    
    # Insert 5
    root = insert(root, 5)
    print("After inserting 5: [4, null, 6, 5, null]")
    print("       4")
    print("        \\")
    print("         6")
    print("        /")
    print("       5\n")
    
    # Search
    print(f"Search for 5: {search(root, 5)}")
    print(f"Search for 3: {search(root, 3)}\n")
    
    # Remove node with 0 children (5)
    root = remove(root, 5)
    print("After removing 5 (0 children):")
    print("       4")
    print("        \\")
    print("         6\n")
    
    # Insert more nodes for removal demo
    root = insert(root, 2)
    root = insert(root, 7)
    root = insert(root, 3)
    print("After inserting 2, 7, 3:")
    print("       4")
    print("      / \\")
    print("     2   6")
    print("      \\   \\")
    print("       3   7\n")
    
    # Remove node with 1 child (2)
    root = remove(root, 2)
    print("After removing 2 (1 child):")
    print("       4")
    print("        \\")
    print("         3")
    print("          \\")
    print("           6")
    print("            \\")
    print("             7\n")
    
    # Remove node with 2 children (4)
    root = insert(root, 2)  # Rebuild for demo
    root = insert(root, 1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(1)
    root.left.right = TreeNode(3)
    print("BST with 2 children case:")
    print("       4")
    print("      / \\")
    print("     2   6")
    print("    / \\   \\")
    print("   1   3   7\n")
    
    root = remove(root, 4)
    print("After removing 4 (2 children - replaced with successor):")
    print("       6")
    print("      / \\")
    print("     2   7")
    print("    / \\")
    print("   1   3\n")
    
    print("=== Key Insights ===")
    print("1. Insert: Always add as leaf → O(log n)")
    print("2. Remove 0-1 children: Replace with child → O(log n)")
    print("3. Remove 2 children: Replace with in-order successor → O(log n)")
    print("4. In-order successor = leftmost in right subtree")
    print("5. All operations O(log n) balanced, O(n) worst case")


if __name__ == "__main__":
    demo()
