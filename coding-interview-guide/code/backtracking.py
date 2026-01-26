"""
Backtracking - Tree Maze Problem

Backtracking is an algorithm that tries all possible solutions and backtracks
when hitting a dead-end. It's like being trapped in a maze: try all paths,
backtrack from dead-ends, find the correct path.

This module demonstrates backtracking on binary trees with the "Path Sum"
problem: find a path from root to leaf without any node having value 0.

Time: O(n) - visit every node in worst case
Space: O(h) - recursion stack depth + path list (h is tree height)
"""


class TreeNode:
    """Node in a binary tree."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def canReachLeaf(root):
    """
    Check if valid path exists (no zeros) from root to leaf.
    
    This is the basic backtracking version that only returns True/False.
    
    Algorithm:
    1. Base case: If root is None or root.val == 0 → False (constraint violation)
    2. Base case: If leaf node → True (valid path found!)
    3. Try left subtree → if True, return True
    4. Try right subtree → if True, return True
    5. Both failed → return False
    
    The key insight: If a solution exists, it's in either left or right subtree.
    We try left first, then right. If both fail, we backtrack (return False).
    
    Time: O(n) - visit every node in worst case
    Space: O(h) - recursion stack depth (h is tree height)
    """
    # Constraint check: empty tree or node with value 0 invalidates path
    if not root or root.val == 0:
        return False
    
    # Base case: leaf node reached - we found a valid path!
    if not root.left and not root.right:
        return True
    
    # Try left subtree first
    # If left subtree has a valid path, return True immediately
    if canReachLeaf(root.left):
        return True
    
    # Try right subtree
    # If right subtree has a valid path, return True
    if canReachLeaf(root.right):
        return True
    
    # Both subtrees failed - no valid path from this node
    return False


def leafPath(root, path):
    """
    Find valid path (no zeros) and build the path list.
    
    This version also builds the actual path by maintaining a list.
    When backtracking, we must remove nodes from the path (pop).
    
    Algorithm:
    1. Check constraint: If root is None or root.val == 0 → False
    2. Add current node to path (make choice)
    3. Base case: If leaf → True (valid path found!)
    4. Try left subtree → if True, return True
    5. Try right subtree → if True, return True
    6. Backtrack: Remove current node from path (undo choice)
    7. Return False (no valid path in this branch)
    
    Key insight: When backtracking, we must "undo" by removing the node
    from the path list before trying other branches.
    
    Time: O(n) - visit every node in worst case
    Space: O(h) - recursion stack + path list (h is tree height)
    """
    # Constraint check: empty tree or node with value 0 invalidates path
    if not root or root.val == 0:
        return False
    
    # Make choice: add current node to path
    path.append(root.val)
    
    # Base case: leaf node reached - we found a valid path!
    if not root.left and not root.right:
        return True
    
    # Try left subtree
    # If left subtree has a valid path, return True (path already contains nodes)
    if leafPath(root.left, path):
        return True
    
    # Try right subtree
    # If right subtree has a valid path, return True
    if leafPath(root.right, path):
        return True
    
    # Backtrack: undo choice by removing current node from path
    # This is crucial - we must clean up before trying other branches
    path.pop()
    return False


def demo():
    """
    Demonstrate backtracking on tree maze problem.
    """
    print("=== Backtracking - Tree Maze Problem ===\n")
    
    # Example 1: Valid path exists
    # Tree: [4, 0, 1, null, 7, 2, 0]
    #       4
    #      / \
    #     0   1
    #        / \
    #       7   2
    #            \
    #             0
    # Valid path: 4 → 1 → 2
    print("Example 1: Valid path exists")
    root1 = TreeNode(4)
    root1.left = TreeNode(0)
    root1.right = TreeNode(1)
    root1.right.left = TreeNode(7)
    root1.right.right = TreeNode(2)
    root1.right.right.right = TreeNode(0)
    
    print(f"Tree: [4, 0, 1, null, 7, 2, 0]")
    print(f"Can reach leaf? {canReachLeaf(root1)}")  # True
    
    path1 = []
    if leafPath(root1, path1):
        print(f"Valid path: {path1}")  # [4, 1, 2]
    print()
    
    # Example 2: No valid path exists
    # Tree: [4, 0, 1, null, 0, 2, 0]
    #       4
    #      / \
    #     0   1
    #        / \
    #       0   2
    #            \
    #             0
    # All paths contain 0
    print("Example 2: No valid path exists")
    root2 = TreeNode(4)
    root2.left = TreeNode(0)
    root2.right = TreeNode(1)
    root2.right.left = TreeNode(0)
    root2.right.right = TreeNode(2)
    root2.right.right.right = TreeNode(0)
    
    print(f"Tree: [4, 0, 1, null, 0, 2, 0]")
    print(f"Can reach leaf? {canReachLeaf(root2)}")  # False
    
    path2 = []
    if leafPath(root2, path2):
        print(f"Valid path: {path2}")
    else:
        print(f"No valid path found. Path after search: {path2}")  # []
    print()
    
    # Example 3: Building path step-by-step
    # Tree: [4, 0, 1, null, 7, 3, 2, null, null, null, 0]
    #       4
    #      / \
    #     0   1
    #        / \
    #       7   3
    #          / \
    #         2   0
    # Valid path: 4 → 1 → 2
    print("Example 3: Building path with backtracking")
    root3 = TreeNode(4)
    root3.left = TreeNode(0)
    root3.right = TreeNode(1)
    root3.right.left = TreeNode(7)
    root3.right.right = TreeNode(3)
    root3.right.right.left = TreeNode(2)
    root3.right.right.right = TreeNode(0)
    
    print(f"Tree: [4, 0, 1, null, 7, 3, 2, null, null, null, 0]")
    path3 = []
    if leafPath(root3, path3):
        print(f"Valid path found: {path3}")  # [4, 1, 2]
        print("Path building process:")
        print("  1. Add 4 → [4]")
        print("  2. Try left (0) → Invalid, backtrack")
        print("  3. Try right (1) → Valid, add 1 → [4, 1]")
        print("  4. Try 1's left (7) → Valid, add 7 → [4, 1, 7]")
        print("  5. 7 is leaf but no valid path → Remove 7 → [4, 1]")
        print("  6. Try 1's right (3) → Valid, add 3 → [4, 1, 3]")
        print("  7. Try 3's left (2) → Valid, add 2 → [4, 1, 3, 2]")
        print("  8. 2 is leaf → Found valid path!")
    print()
    
    print("=== Key Insights ===")
    print("1. Backtracking tries all possible solutions")
    print("2. When hitting dead-end, backtrack (undo choice)")
    print("3. Constraint checking: No zeros in path")
    print("4. Base case: Leaf node = valid path found")
    print("5. Must pop() when backtracking to undo choices")
    print("6. Time: O(n) - visit all nodes")
    print("7. Space: O(h) - recursion stack + path list")


if __name__ == "__main__":
    demo()
