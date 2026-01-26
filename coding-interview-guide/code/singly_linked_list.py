"""
Singly Linked List Implementation

Linked lists store elements in nodes that point to the next node.
Unlike arrays, elements don't need to be stored contiguously in memory.

Key advantage: Insertion/deletion at a known position is O(1).
Key disadvantage: Accessing by index is O(n) - must traverse.
"""


class ListNode:
    """
    A single node in the linked list.
    Contains the value and a pointer to the next node.
    """
    def __init__(self, val):
        self.val = val  # The data stored in this node
        self.next = None  # Pointer to the next node (None = end of list)


def traverse(head):
    """
    Traverse the entire list and collect all values. O(n) time.
    
    We start at head and follow the 'next' pointers until we hit None.
    """
    values = []
    cur = head  # Start at the beginning
    
    # Keep moving forward until we reach the end (None)
    while cur:
        values.append(cur.val)
        cur = cur.next  # Move to next node
    
    return values


def insert_after(node, new_node):
    """
    Insert new_node right after the given node. O(1) time.
    
    This is why linked lists are great - no shifting needed!
    We just update two pointers.
    """
    # Step 1: Point new node to what the current node was pointing to
    new_node.next = node.next
    
    # Step 2: Point current node to the new node
    node.next = new_node


def delete_after(node):
    """
    Delete the node that comes after the given node. O(1) time.
    
    We skip over the next node by pointing directly to the one after it.
    """
    if node.next:
        # Skip the next node by pointing to the one after it
        node.next = node.next.next


def demo():
    a = ListNode("red")
    b = ListNode("blue")
    c = ListNode("green")
    a.next = b
    b.next = c

    print("Initial:", traverse(a))
    insert_after(b, ListNode("yellow"))
    print("After insert:", traverse(a))
    delete_after(a)
    print("After delete:", traverse(a))


if __name__ == "__main__":
    demo()
