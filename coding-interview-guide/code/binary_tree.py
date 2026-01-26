"""
Binary Trees

Similar to linked lists, binary trees use nodes and pointers. But instead of
connecting nodes in a straight line, binary trees connect nodes hierarchically
with left and right child pointers.

Key concepts:
- Root: First node (no parent)
- Leaf: Node with no children
- Non-leaf: Node with at least one child
- Height: Distance from root to lowest leaf (by nodes or edges)
- Depth: Distance from node to root (including node itself)
- Ancestor: Node connected to all nodes below it
- Descendant: Child or child of descendant

Properties:
- At most 2 children per node (left and/or right)
- No cycles (pointers only go downward)
- Connected (all nodes reachable from root)
- Guaranteed leaves exist

Time: O(n) for traversal (must visit each node)
Space: O(h) where h is height (recursion stack depth)
"""


class TreeNode:
    """
    Node in a binary tree.
    
    Similar to ListNode, but with left and right child pointers instead of
    next/prev pointers. Pointers go downward, creating a hierarchical structure.
    
    Each node contains:
    - val: The value stored in the node (any data type)
    - left: Pointer to left child (or None)
    - right: Pointer to right child (or None)
    """
    def __init__(self, val):
        self.val = val
        self.left = None   # Left child pointer
        self.right = None  # Right child pointer


def max_depth(root):
    """
    Calculate maximum depth (height) of binary tree.
    
    Depth: Distance from root (root depth = 0)
    Height: Longest path from root to leaf
    
    Algorithm:
    1. Base case: If node is None, return 0
    2. Recursively find depth of left subtree
    3. Recursively find depth of right subtree
    4. Return 1 + max(left_depth, right_depth)
    
    Time: O(n) - visit each node once
    Space: O(h) - recursion stack depth equals tree height
    """
    if not root:
        return 0
    
    # Recursively find depth of left and right subtrees
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    # Return max depth + 1 (current node adds 1 to depth)
    return 1 + max(left_depth, right_depth)


def same_tree(p, q):
    """
    Check if two binary trees are identical.
    
    Two trees are same if:
    1. Both are None (base case)
    2. Both have same value
    3. Left subtrees are same
    4. Right subtrees are same
    
    Time: O(n) - visit each node once
    Space: O(h) - recursion stack depth
    """
    # Base case: both None
    if not p and not q:
        return True
    
    # One is None, other isn't
    if not p or not q:
        return False
    
    # Check value and recursively check subtrees
    return (p.val == q.val and
            same_tree(p.left, q.left) and
            same_tree(p.right, q.right))


def invert_tree(root):
    """
    Invert (mirror) a binary tree.
    
    Swap left and right children for every node.
    
    Algorithm:
    1. Base case: If node is None, return None
    2. Swap left and right children
    3. Recursively invert left subtree
    4. Recursively invert right subtree
    5. Return root
    
    Time: O(n) - visit each node once
    Space: O(h) - recursion stack depth
    """
    if not root:
        return None
    
    # Swap left and right children
    root.left, root.right = root.right, root.left
    
    # Recursively invert subtrees
    invert_tree(root.left)
    invert_tree(root.right)
    
    return root


def count_nodes(root):
    """
    Count total number of nodes in tree.
    
    Time: O(n) - visit each node once
    Space: O(h) - recursion stack depth
    """
    if not root:
        return 0
    
    return 1 + count_nodes(root.left) + count_nodes(root.right)


def demo():
    """
    Demonstrate binary tree operations.
    """
    print("=== Binary Tree Demo ===\n")
    
    # Create a sample tree:
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    
    print("Tree structure:")
    print("       1")
    print("      / \\")
    print("     2   3")
    print("    / \\")
    print("   4   5\n")
    
    print(f"Max depth: {max_depth(root)}")
    print(f"Total nodes: {count_nodes(root)}")
    
    # Create another identical tree
    root2 = TreeNode(1)
    root2.left = TreeNode(2)
    root2.right = TreeNode(3)
    root2.left.left = TreeNode(4)
    root2.left.right = TreeNode(5)
    
    print(f"Trees are same: {same_tree(root, root2)}")
    
    # Invert the tree
    inverted = invert_tree(root)
    print("\nAfter inversion:")
    print("       1")
    print("      / \\")
    print("     3   2")
    print("        / \\")
    print("       5   4")
    
    print("\n=== Key Insights ===")
    print("1. Trees are recursive - each subtree is a tree")
    print("2. Most tree problems use recursion naturally")
    print("3. Base case: usually when node is None")
    print("4. Recursive case: process current node, recurse on children")
    print("5. Time complexity often O(n) - visit each node once")
    print("6. Space complexity O(h) - recursion stack depth")


if __name__ == "__main__":
    demo()
