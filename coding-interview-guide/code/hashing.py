"""
Hash Usage - HashSet and HashMap

Hash maps and hash sets implement Map and Set interfaces using hashing.
- HashMap: Key-value pairs with O(1) average-time operations
- HashSet: Unique keys only with O(1) average-time operations

Key advantages over TreeMap/TreeSet:
- O(1) average operations (vs O(log n))
- Faster for frequency counting and lookups

Key disadvantages:
- Unordered (keys not sorted)
- Cannot iterate in sorted order without O(n log n) sort

When to use:
- Keywords: "unique", "count", "frequency"
- Need fast lookups/insertions
- Don't need sorted order

Time: O(1) average for insert/remove/search
Space: O(n) for storing n unique keys
"""


def count_frequency(arr):
    """
    Count frequency of each element using hash map.
    
    This is the classic hash map use case - frequency counting.
    Perfect for problems asking "how many times does X appear?"
    
    Algorithm:
    1. Iterate through array once
    2. For each element:
       - If not in map: add with count 1
       - If in map: increment count
    3. Return frequency map
    
    Time: O(n) - single pass through array
           Each insert/lookup is O(1) average
    Space: O(n) - store unique elements (worst case: all unique)
    """
    count_map = {}
    
    for item in arr:
        if item not in count_map:
            count_map[item] = 1  # First occurrence
        else:
            count_map[item] += 1  # Increment frequency
    
    return count_map


def count_frequency_short(arr):
    """
    Shorter version using get() with default value.
    
    More Pythonic and concise.
    """
    count_map = {}
    for item in arr:
        count_map[item] = count_map.get(item, 0) + 1
    return count_map


def find_duplicates(arr):
    """
    Find duplicate elements using hash set.
    
    Uses HashSet to track seen elements.
    Perfect for "contains duplicate" problems.
    
    Time: O(n)
    Space: O(n)
    """
    seen = set()
    duplicates = []
    
    for item in arr:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)
    
    return duplicates


def two_sum(nums, target):
    """
    Two Sum using hash map for O(n) solution.
    
    Classic hash map pattern: store complements for O(1) lookup.
    
    Algorithm:
    1. Iterate through array
    2. For each number, calculate complement (target - num)
    3. If complement in map: found pair!
    4. Otherwise, store current number and index
    
    Time: O(n) - single pass
    Space: O(n) - store up to n elements
    """
    seen = {}  # {value: index}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]  # Found pair!
        seen[num] = i  # Store current number and index
    
    return []  # No solution


def demo():
    """
    Demonstrate hash map and hash set usage.
    """
    print("=== Hash Usage Demo ===\n")
    
    # Example 1: Frequency Counting
    print("Example 1: Frequency Counting")
    names = ["alice", "brad", "collin", "brad", "dylan", "kim"]
    print(f"Input: {names}")
    
    frequency = count_frequency(names)
    print("Frequency map:")
    for name, count in frequency.items():
        print(f"  {name}: {count}")
    print()
    
    # Example 2: Using shorter version
    print("Example 2: Shorter version (using get())")
    frequency2 = count_frequency_short(names)
    print(f"Result: {frequency2}")
    print()
    
    # Example 3: Find Duplicates
    print("Example 3: Find Duplicates")
    arr = [1, 2, 3, 2, 4, 3, 5]
    print(f"Input: {arr}")
    duplicates = find_duplicates(arr)
    print(f"Duplicates: {duplicates}")
    print()
    
    # Example 4: Two Sum
    print("Example 4: Two Sum")
    nums = [2, 7, 11, 15]
    target = 9
    print(f"Input: {nums}, Target: {target}")
    result = two_sum(nums, target)
    print(f"Indices: {result}")
    print(f"Values: {nums[result[0]]}, {nums[result[1]]}")
    print()
    
    # Example 5: HashSet for unique values
    print("Example 5: HashSet for Unique Values")
    arr_with_duplicates = [1, 2, 2, 3, 3, 3, 4, 5]
    unique_set = set(arr_with_duplicates)
    print(f"Input: {arr_with_duplicates}")
    print(f"Unique values: {unique_set}")
    print()
    
    print("=== When to Use Hash Maps/Sets ===")
    print("Keywords that suggest hashing:")
    print("  - 'unique' → HashSet")
    print("  - 'count' → HashMap")
    print("  - 'frequency' → HashMap")
    print("  - 'appears X times' → HashMap")
    print("  - 'no duplicates' → HashSet")
    print()
    print("Common patterns:")
    print("  1. Frequency counting")
    print("  2. Lookup optimization (O(1) vs O(n))")
    print("  3. Duplicate detection")
    print("  4. Two Sum pattern (store complements)")
    print()
    print("=== Time Complexity ===")
    print("Operation    | TreeMap      | HashMap")
    print("------------|--------------|----------")
    print("Insert      | O(log n)     | O(1) avg")
    print("Remove      | O(log n)     | O(1) avg")
    print("Search      | O(log n)     | O(1) avg")
    print("Sorted Iter | O(n)         | O(n log n)*")
    print()
    print("*HashMap: Need to sort first, then iterate")
    print()
    print("=== Key Trade-offs ===")
    print("TreeMap:")
    print("  ✓ Ordered (sorted keys)")
    print("  ✓ O(n) inorder traversal")
    print("  ✗ O(log n) operations")
    print()
    print("HashMap:")
    print("  ✓ O(1) average operations")
    print("  ✓ Faster for frequency counting")
    print("  ✗ Unordered")
    print("  ✗ O(n log n) to iterate in sorted order")


# ============================================================================
# Hash Map Implementation (Open Addressing)
# ============================================================================

class Pair:
    """
    Stores a key-value pair for the hash map.
    """
    def __init__(self, key, val):
        self.key = key
        self.val = val


class HashMap:
    """
    Hash Map implementation using open addressing (linear probing).
    
    Under the hood:
    - Uses array to store key-value pairs
    - Hash function converts key to array index
    - Handles collisions by finding next available slot
    - Resizes when half full (load factor = 0.5)
    - Rehashes all elements after resize
    
    Time: O(1) average, O(n) worst case
    Space: O(n) where n is capacity
    """
    def __init__(self):
        self.size = 0  # Number of key-value pairs
        self.capacity = 2  # Array size
        self.map = [None, None]  # Array of Pairs (or None)
    
    def hash(self, key):
        """
        Convert key to array index using hash function.
        
        Algorithm:
        1. Sum ASCII codes of all characters in key
        2. Use modulo to get valid index
        
        Same key → same index (deterministic)
        Different keys → may collide (same index)
        
        Time: O(k) where k is length of key
        """
        index = 0
        for c in key:
            index += ord(c)  # Get ASCII code
        return index % self.capacity  # Valid index in range [0, capacity)
    
    def get(self, key):
        """
        Get value for given key.
        
        Algorithm (open addressing):
        1. Hash key to get starting index
        2. Check if key matches at that index
        3. If not, check next index (linear probing)
        4. Wrap around using modulo if reach end
        5. Return None if key not found
        
        Time: O(1) average, O(n) worst case (all keys collide)
        """
        index = self.hash(key)
        
        # Linear probing: check starting index and next slots
        while self.map[index] != None:
            if self.map[index].key == key:
                return self.map[index].val  # Found!
            index += 1
            index = index % self.capacity  # Wrap around
        
        return None  # Key not found
    
    def put(self, key, val):
        """
        Insert or update key-value pair.
        
        Algorithm:
        1. Hash key to get starting index
        2. Three cases:
           a. Index is vacant → insert new pair
           b. Index has same key → update value
           c. Index has different key → try next index (collision)
        3. Resize if array becomes half full
        
        Time: O(1) average, O(n) worst case
        """
        index = self.hash(key)
        
        while True:
            if self.map[index] == None:
                # Vacant slot → insert new pair
                self.map[index] = Pair(key, val)
                self.size += 1
                # Resize if half full (load factor = 0.5)
                if self.size >= self.capacity // 2:
                    self.rehash()
                return
            elif self.map[index].key == key:
                # Same key → update value
                self.map[index].val = val
                return
            
            # Collision → try next index (open addressing)
            index += 1
            index = index % self.capacity
    
    def remove(self, key):
        """
        Remove key-value pair.
        
        Note: Removing with open addressing creates a "hole" that can
        break get() operations. A proper implementation would need to
        handle this by rehashing or using a tombstone marker.
        
        For simplicity, we just set to None (creates potential bug).
        
        Time: O(1) average, O(n) worst case
        """
        if not self.get(key):
            return  # Key doesn't exist
        
        index = self.hash(key)
        while True:
            if self.map[index] and self.map[index].key == key:
                self.map[index] = None  # Remove (creates hole!)
                self.size -= 1
                return
            index += 1
            index = index % self.capacity
    
    def rehash(self):
        """
        Resize array and rehash all existing elements.
        
        Algorithm:
        1. Double capacity
        2. Create new array
        3. Rehash all existing pairs (positions change with new capacity!)
        
        Why rehash?
        - Old: hash(key) % old_capacity
        - New: hash(key) % new_capacity
        - Positions change → must recompute!
        
        Time: O(n) where n is number of elements
        """
        self.capacity = 2 * self.capacity
        new_map = [None] * self.capacity
        
        old_map = self.map
        self.map = new_map
        self.size = 0  # Reset size (will be recalculated during rehash)
        
        # Rehash all existing pairs
        for pair in old_map:
            if pair:
                self.put(pair.key, pair.val)
    
    def print_map(self):
        """Print all key-value pairs."""
        for pair in self.map:
            if pair:
                print(f"{pair.key}: {pair.val}")


def demo_implementation():
    """
    Demonstrate hash map implementation.
    """
    print("=== Hash Map Implementation Demo ===\n")
    
    # Create hash map
    hash_map = HashMap()
    
    # Insert key-value pairs
    print("Inserting key-value pairs:")
    hash_map.put("Alice", "NYC")
    print(f"  Put ('Alice', 'NYC')")
    print(f"  Capacity: {hash_map.capacity}, Size: {hash_map.size}")
    
    hash_map.put("Brad", "Chicago")
    print(f"  Put ('Brad', 'Chicago')")
    print(f"  Capacity: {hash_map.capacity}, Size: {hash_map.size}")
    
    hash_map.put("Collin", "Seattle")
    print(f"  Put ('Collin', 'Seattle')")
    print(f"  Capacity: {hash_map.capacity}, Size: {hash_map.size}")
    print()
    
    # Get values
    print("Getting values:")
    print(f"  get('Alice'): {hash_map.get('Alice')}")
    print(f"  get('Brad'): {hash_map.get('Brad')}")
    print(f"  get('Collin'): {hash_map.get('Collin')}")
    print(f"  get('David'): {hash_map.get('David')}")  # Doesn't exist
    print()
    
    # Update value
    print("Updating value:")
    hash_map.put("Alice", "Boston")
    print(f"  Updated 'Alice' to 'Boston'")
    print(f"  get('Alice'): {hash_map.get('Alice')}")
    print()
    
    # Print all pairs
    print("All key-value pairs:")
    hash_map.print_map()
    print()
    
    print("=== Key Concepts ===")
    print("1. Hash function: key → integer → array index")
    print("2. Collisions: multiple keys map to same index")
    print("3. Open addressing: find next available slot")
    print("4. Resizing: double capacity when half full")
    print("5. Rehashing: recompute positions after resize")
    print()
    print("=== Time Complexity ===")
    print("Operation | Average | Worst Case")
    print("----------|---------|-----------")
    print("Insert    | O(1)    | O(n)")
    print("Remove    | O(1)    | O(n)")
    print("Search    | O(1)    | O(n)")
    print()
    print("Average: O(1) with good hash function and low collisions")
    print("Worst: O(n) when all keys collide")


if __name__ == "__main__":
    demo()
    print("\n" + "="*50 + "\n")
    demo_implementation()