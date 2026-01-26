"""
Search a 2D Matrix

Given a sorted 2D matrix (each row sorted left-to-right, rows sorted top-to-bottom),
determine if a target value exists in the matrix.

Multiple approaches:
1. Brute force: O(m×n) - check every cell
2. Staircase search: O(m+n) - start top-right, eliminate row/column
3. Binary search (two-pass): O(log m + log n) - find row, then search row
4. Binary search (one-pass): O(log(m×n)) - treat matrix as flattened array
"""


def search_matrix_brute(matrix, target):
    """
    Brute force approach - check every cell.
    
    Time: O(m × n) - worst case check all cells
    Space: O(1)
    """
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            if matrix[r][c] == target:
                return True
    return False


def search_matrix_staircase(matrix, target):
    """
    Staircase search - start at top-right corner.
    
    Strategy:
    - Start at top-right (smallest in row, largest in column)
    - If value > target: move left (values decrease)
    - If value < target: move down (values increase)
    - Like walking down stairs!
    
    Time: O(m + n) - at most m+n steps
    Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    r, c = 0, n - 1  # Start at top-right corner
    
    while r < m and c >= 0:
        if matrix[r][c] > target:
            # Current value too large → move left (smaller values)
            c -= 1
        elif matrix[r][c] < target:
            # Current value too small → move down (larger values)
            r += 1
        else:
            # Found target!
            return True
    
    return False


def search_matrix_two_pass(matrix, target):
    """
    Two-pass binary search.
    
    Pass 1: Binary search over rows to find which row contains target
    Pass 2: Binary search within that row
    
    Time: O(log m + log n) = O(log(m×n))
    Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    ROWS, COLS = len(matrix), len(matrix[0])
    
    # Pass 1: Find the row that could contain target
    top, bot = 0, ROWS - 1
    while top <= bot:
        row = (top + bot) // 2
        
        if target > matrix[row][-1]:
            # Target is greater than last element → search lower rows
            top = row + 1
        elif target < matrix[row][0]:
            # Target is smaller than first element → search upper rows
            bot = row - 1
        else:
            # Target is between first and last element → found candidate row
            break
    
    # Check if we found a valid row
    if not (top <= bot):
        return False
    
    # Pass 2: Binary search within the identified row
    row = (top + bot) // 2
    l, r = 0, COLS - 1
    
    while l <= r:
        m = l + (r - l) // 2
        
        if target > matrix[row][m]:
            l = m + 1
        elif target < matrix[row][m]:
            r = m - 1
        else:
            return True
    
    return False


def search_matrix_one_pass(matrix, target):
    """
    One-pass binary search - treat matrix as flattened sorted array.
    
    Key insight: Convert 1D index to 2D coordinates:
    - row = index // COLS
    - col = index % COLS
    
    This lets us binary search without actually flattening the matrix.
    
    Time: O(log(m×n))
    Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    ROWS, COLS = len(matrix), len(matrix[0])
    l, r = 0, ROWS * COLS - 1
    
    while l <= r:
        m = l + (r - l) // 2
        
        # Convert 1D index to 2D coordinates
        row = m // COLS
        col = m % COLS
        
        if target > matrix[row][col]:
            l = m + 1
        elif target < matrix[row][col]:
            r = m - 1
        else:
            return True
    
    return False


def demo():
    """
    Demonstrate different approaches to searching 2D matrix.
    """
    print("=== Search 2D Matrix Demo ===\n")
    
    matrix = [
        [1, 4, 7, 11],
        [2, 5, 8, 12],
        [3, 6, 9, 16],
        [10, 13, 14, 17]
    ]
    
    target = 9
    
    print("Matrix:")
    for row in matrix:
        print(row)
    print(f"\nTarget: {target}\n")
    
    print("Approach 1: Brute Force")
    result1 = search_matrix_brute(matrix, target)
    print(f"Found: {result1} (O(m×n) time)\n")
    
    print("Approach 2: Staircase Search")
    result2 = search_matrix_staircase(matrix, target)
    print(f"Found: {result2} (O(m+n) time)\n")
    
    print("Approach 3: Two-Pass Binary Search")
    result3 = search_matrix_two_pass(matrix, target)
    print(f"Found: {result3} (O(log m + log n) time)\n")
    
    print("Approach 4: One-Pass Binary Search")
    result4 = search_matrix_one_pass(matrix, target)
    print(f"Found: {result4} (O(log(m×n)) time)\n")
    
    print("=== Key Insights ===")
    print("1. Staircase search: Start top-right, eliminate row/column each step")
    print("2. Two-pass binary: Find row first, then search within row")
    print("3. One-pass binary: Treat matrix as flattened array")
    print("4. Index conversion: row = m // COLS, col = m % COLS")
    print("\n=== Common Pitfalls ===")
    print("⚠️  Wrong conversion: Use COLS not ROWS for division")
    print("⚠️  Off-by-one: Recalculate row after first binary search")
    print("⚠️  Empty matrix: Always check matrix and matrix[0] exist")


if __name__ == "__main__":
    demo()
