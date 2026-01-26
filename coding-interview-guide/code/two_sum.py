"""
Two Sum Problem

Given an array and a target sum, find two numbers that add up to target.
Return their indices.

Key insight: Instead of checking all pairs (O(n²)), we can do it in one
pass using a hash map. For each number, we check if we've already seen
its "partner" (the number we need to reach the target).

Time: O(n) - single pass through array
Space: O(n) - hash map stores at most n elements
"""


def two_sum(nums, target):
    """
    Return indices of the two numbers that add up to target.
    Assumes exactly one solution exists.
    
    Strategy:
    1. As we scan, store each number and its index in a map
    2. For each new number, calculate what partner we need
    3. If partner is already in map, we found our pair!
    """
    seen = {}  # Maps number -> its index
    
    for i, value in enumerate(nums):
        # Calculate what number we need to pair with current value
        needed = target - value
        
        # Check if we've already seen the needed number
        if needed in seen:
            # Found it! Return both indices
            return [seen[needed], i]
        
        # Store current number and its index for future lookups
        seen[value] = i
    
    return []  # No solution found (shouldn't happen per problem statement)


if __name__ == "__main__":
    print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
