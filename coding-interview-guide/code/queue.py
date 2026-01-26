"""
Queue Implementation

A queue is FIFO (First In First Out) - like a line at a bank.
First person in is first person out. All operations are O(1).

Common uses:
- Task scheduling
- Breadth-first search (BFS)
- Print queues
"""


class QNode:
    """
    A node in the queue (using linked list structure).
    """
    def __init__(self, val):
        self.val = val  # The data stored in this node
        self.next = None  # Pointer to next node in queue


class Queue:
    def __init__(self):
        """
        We track both ends:
        - left: front of queue (where we remove from)
        - right: back of queue (where we add to)
        """
        self.left = None  # Front of queue (first to be removed)
        self.right = None  # Back of queue (last added)

    def enqueue(self, val):
        """
        Add element to the back of the queue. O(1) time.
        
        Think: joining the end of a line.
        """
        node = QNode(val)
        
        if self.right:
            # Queue has elements - add to the end
            self.right.next = node  # Link new node after current tail
            self.right = node  # Update tail pointer
        else:
            # Queue is empty - this becomes both front and back
            self.left = self.right = node

    def dequeue(self):
        """
        Remove and return element from the front of the queue. O(1) time.
        
        Think: first person in line gets served.
        """
        if not self.left:
            return None  # Queue is empty
        
        val = self.left.val  # Get value from front
        
        # Move front pointer forward
        self.left = self.left.next
        
        # If queue is now empty, update tail pointer too
        if not self.left:
            self.right = None
        
        return val


def demo():
    q = Queue()
    q.enqueue(10)
    q.enqueue(20)
    print("Dequeue:", q.dequeue())
    print("Dequeue:", q.dequeue())
    print("Dequeue:", q.dequeue())


if __name__ == "__main__":
    demo()
