"""
Doubly Linked List Implementation

Each node has TWO pointers: next (forward) and prev (backward).
This allows traversal in both directions, but uses more memory.

Key advantage: Can traverse backwards, easier deletion from end.
Key disadvantage: More memory per node, more pointer updates.
"""


class DListNode:
    """
    A node in a doubly linked list.
    Has pointers to both the next AND previous node.
    """
    def __init__(self, val):
        self.val = val  # The data stored in this node
        self.prev = None  # Pointer to previous node (None = head)
        self.next = None  # Pointer to next node (None = tail)


class DoublyLinkedList:
    def __init__(self):
        """
        Track both head (first) and tail (last) nodes.
        """
        self.head = None  # First node in the list
        self.tail = None  # Last node in the list

    def append(self, node):
        """
        Add node to the end of the list. O(1) time.
        
        We need to update both next and prev pointers.
        """
        # If list is empty, this node becomes both head and tail
        if not self.head:
            self.head = self.tail = node
            return
        
        # Link the new node to the current tail
        node.prev = self.tail  # New node's prev points to old tail
        self.tail.next = node  # Old tail's next points to new node
        
        # Update tail to be the new node
        self.tail = node

    def pop_tail(self):
        """
        Remove and return the last node. O(1) time.
        
        This is easier than in singly linked lists because we have
        a prev pointer - we don't need to traverse to find the node before tail.
        """
        if not self.tail:
            return None  # List is empty
        
        node = self.tail  # Save reference to node we're removing
        
        # Move tail backwards
        self.tail = node.prev
        
        # Update pointers
        if self.tail:
            self.tail.next = None  # New tail has no next
        else:
            self.head = None  # List is now empty
        
        # Clean up the removed node's pointer
        node.prev = None
        
        return node

    def to_list(self):
        """
        Convert linked list to a regular list. O(n) time.
        """
        values = []
        cur = self.head
        while cur:
            values.append(cur.val)
            cur = cur.next
        return values


def demo():
    dll = DoublyLinkedList()
    for value in [1, 2, 3]:
        dll.append(DListNode(value))
    print("List:", dll.to_list())
    removed = dll.pop_tail()
    print("Removed:", removed.val if removed else None)
    print("List:", dll.to_list())


if __name__ == "__main__":
    demo()
