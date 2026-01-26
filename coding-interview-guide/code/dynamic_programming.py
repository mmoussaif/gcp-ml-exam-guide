"""
1-Dimension Dynamic Programming

Dynamic Programming (DP) is optimized recursion. It breaks big problems into smaller
subproblems and stores results to avoid recomputation.

Key insight: Instead of recalculating the same subproblem multiple times (exponential),
we cache results and reuse them (linear time).

Two approaches:
1. Top-down (Memoization): Recursion + caching
2. Bottom-up (Tabulation): Build table from base cases

Space optimization: For 1-D DP, we often only need the last k values, allowing O(1) space.

Time: O(n) - each subproblem computed once (vs O(2^n) brute force)
Space: O(n) for memoization/tabulation, O(1) for space-optimized
"""


# ============================================================================
# Brute Force (Recursion without DP)
# ============================================================================

def brute_force_fibonacci(n):
    """
    Fibonacci using brute force recursion (no caching).
    
    This has exponential time complexity because we recalculate
    the same subproblems multiple times.
    
    Example: F(5) calculates F(2) three times!
    
    Time: O(2^n) - exponential
    Space: O(n) - recursion stack depth
    """
    if n <= 1:
        return n
    return brute_force_fibonacci(n - 1) + brute_force_fibonacci(n - 2)


# ============================================================================
# Top-Down Approach (Memoization)
# ============================================================================

def memoization_fibonacci(n, cache):
    """
    Fibonacci using top-down DP (memoization).
    
    Memoization = Recursion + Caching
    
    Algorithm:
    1. Base case: n <= 1 → return n
    2. Check cache: if n in cache → return cached value
    3. Compute: F(n) = F(n-1) + F(n-2)
    4. Store in cache before returning
    
    Key insight: Once we compute F(3), we store it and reuse it
    instead of recalculating. This eliminates repeated work.
    
    Time: O(n) - each subproblem computed once
    Space: O(n) - cache dictionary + recursion stack
    """
    if n <= 1:
        return n
    
    # Check cache (memoization) - avoid recomputation!
    if n in cache:
        return cache[n]
    
    # Compute and store in cache
    cache[n] = memoization_fibonacci(n - 1, cache) + memoization_fibonacci(n - 2, cache)
    return cache[n]


# ============================================================================
# Bottom-Up Approach (Tabulation)
# ============================================================================

def tabulation_fibonacci_full(n):
    """
    Fibonacci using bottom-up DP (tabulation) with full array.
    
    Tabulation = Build table from base cases upward
    
    Algorithm:
    1. Create array of size n+1
    2. Initialize base cases: dp[0] = 0, dp[1] = 1
    3. Fill array from 2 to n: dp[i] = dp[i-1] + dp[i-2]
    4. Return dp[n]
    
    Time: O(n) - iterate from 2 to n
    Space: O(n) - array of size n+1
    """
    if n < 2:
        return n
    
    dp = [0] * (n + 1)
    dp[0] = 0  # Base case: F(0) = 0
    dp[1] = 1  # Base case: F(1) = 1
    
    # Fill array from bottom up
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]


def dp_optimized_fibonacci(n):
    """
    Fibonacci using bottom-up DP with O(1) space optimization.
    
    Key insight: We only need the last two values to compute the next!
    Instead of storing all n values, we can use just two variables.
    
    Algorithm:
    1. Base case: n < 2 → return n
    2. Use two variables: prev2 (F(i-2)), prev1 (F(i-1))
    3. Iterate from 2 to n:
       a. Calculate curr = prev1 + prev2
       b. Update: prev2 = prev1, prev1 = curr
    4. Return prev1 (or curr)
    
    Time: O(n) - iterate from 2 to n
    Space: O(1) - only two variables!
    """
    if n < 2:
        return n
    
    # Only need last two values
    prev2 = 0  # F(i-2)
    prev1 = 1  # F(i-1)
    
    for i in range(2, n + 1):
        curr = prev1 + prev2
        # Slide window: move prev1 to prev2, curr to prev1
        prev2 = prev1
        prev1 = curr
    
    return prev1


def demo():
    """
    Demonstrate different DP approaches for Fibonacci.
    """
    print("=== 1-Dimension Dynamic Programming Demo ===\n")
    
    n = 10
    print(f"Calculate F({n}) using different approaches:\n")
    
    # Brute Force (slow for large n)
    print("1. Brute Force (Recursion without DP):")
    print(f"   F({n}) = {brute_force_fibonacci(n)}")
    print(f"   Time: O(2^n) - exponential ❌")
    print(f"   Space: O(n) - recursion stack")
    print()
    
    # Memoization (Top-down)
    print("2. Memoization (Top-down DP):")
    cache = {}
    result = memoization_fibonacci(n, cache)
    print(f"   F({n}) = {result}")
    print(f"   Cache: {cache}")
    print(f"   Time: O(n) - each subproblem computed once ✓")
    print(f"   Space: O(n) - cache + recursion stack")
    print()
    
    # Tabulation (Bottom-up)
    print("3. Tabulation (Bottom-up DP):")
    result = tabulation_fibonacci_full(n)
    print(f"   F({n}) = {result}")
    print(f"   Time: O(n) - iterate from 2 to n ✓")
    print(f"   Space: O(n) - array of size n+1")
    print()
    
    # Space-Optimized
    print("4. Space-Optimized Tabulation:")
    result = dp_optimized_fibonacci(n)
    print(f"   F({n}) = {result}")
    print(f"   Time: O(n) - iterate from 2 to n ✓")
    print(f"   Space: O(1) - only two variables! ✓")
    print()
    
    print("=== Key Concepts ===")
    print("1. DP = Optimized recursion (cache results)")
    print("2. Memoization: Top-down (recursion + cache)")
    print("3. Tabulation: Bottom-up (build table from base cases)")
    print("4. Space optimization: Only store needed values")
    print()
    
    print("=== Comparison ===")
    print("Approach        | Time  | Space | Notes")
    print("----------------|-------|-------|----------------------")
    print("Brute Force     | O(2^n)| O(n)  | Repeated calc ❌")
    print("Memoization     | O(n)  | O(n)  | Top-down ✓")
    print("Tabulation      | O(n)  | O(n)  | Bottom-up ✓")
    print("Space-Optimized | O(n)  | O(1)  | Best! ✓✓")
    print()
    
    print("=== Why DP Works ===")
    print("Brute Force:")
    print("  F(5) → F(4) + F(3)")
    print("  F(4) → F(3) + F(2)")
    print("  F(3) → F(2) + F(1)")
    print("  F(2) calculated 3 times! ❌")
    print()
    print("DP (Memoization):")
    print("  F(5) → F(4) + F(3)")
    print("  F(4) → F(3) + F(2) [store F(3), F(2)]")
    print("  F(3) → Check cache → Found! Return ✓")
    print("  F(2) calculated once! ✓")


# ============================================================================
# 2-Dimension Dynamic Programming
# ============================================================================

def brute_force_unique_paths(r, c, rows, cols):
    """
    Count unique paths using brute force recursion (no caching).
    
    Problem: Count paths from (r,c) to (rows-1, cols-1)
    Only allowed moves: down or right
    
    Algorithm:
    1. Base case: Out of bounds → return 0
    2. Base case: Reached destination → return 1
    3. Sum paths from moving down + moving right
    
    Time: O(2^(n+m)) - exponential
          Each cell has 2 choices, maximum depth = n+m
    Space: O(n+m) - recursion stack depth
    """
    if r == rows or c == cols:
        return 0  # Out of bounds
    
    if r == rows - 1 and c == cols - 1:
        return 1  # Reached destination!
    
    # Sum paths from both directions
    return (brute_force_unique_paths(r + 1, c, rows, cols) +  # Move down
            brute_force_unique_paths(r, c + 1, rows, cols))  # Move right


def memoization_unique_paths(r, c, rows, cols, cache):
    """
    Count unique paths using top-down DP (memoization) with 2D cache.
    
    Algorithm:
    1. Base case: Out of bounds → return 0
    2. Check cache: if computed → return cached value
    3. Base case: Reached destination → return 1
    4. Compute: paths = paths_down + paths_right
    5. Store in cache before returning
    
    Time: O(n*m) - each cell computed once
    Space: O(n*m) - 2D cache array + recursion stack
    """
    if r == rows or c == cols:
        return 0
    
    # Check cache (memoization) - avoid recomputation!
    if cache[r][c] > 0:
        return cache[r][c]
    
    if r == rows - 1 and c == cols - 1:
        return 1
    
    # Compute and store in cache
    cache[r][c] = (memoization_unique_paths(r + 1, c, rows, cols, cache) +
                   memoization_unique_paths(r, c + 1, rows, cols, cache))
    return cache[r][c]


def tabulation_unique_paths_full(rows, cols):
    """
    Count unique paths using bottom-up DP (tabulation) with full 2D array.
    
    Algorithm:
    1. Create 2D array dp[rows][cols]
    2. Initialize: dp[rows-1][cols-1] = 1 (destination)
    3. Fill from bottom-right to top-left:
       dp[r][c] = dp[r+1][c] + dp[r][c+1]
    4. Return dp[0][0]
    
    Time: O(n*m)
    Space: O(n*m) - full 2D array
    """
    dp = [[0] * cols for _ in range(rows)]
    dp[rows - 1][cols - 1] = 1  # Destination
    
    # Fill from bottom-right to top-left
    for r in range(rows - 1, -1, -1):
        for c in range(cols - 1, -1, -1):
            if r == rows - 1 and c == cols - 1:
                continue  # Already set
            down = dp[r + 1][c] if r + 1 < rows else 0
            right = dp[r][c + 1] if c + 1 < cols else 0
            dp[r][c] = down + right
    
    return dp[0][0]


def dp_optimized_unique_paths(rows, cols):
    """
    Count unique paths using bottom-up DP with O(m) space optimization.
    
    Key insight: To calculate row r, we only need row r+1.
    We can use just one previous row instead of full 2D array.
    
    Algorithm:
    1. Initialize prevRow with all 0s
    2. For each row from bottom to top:
       a. Initialize curRow
       b. Set rightmost column = 1
       c. Fill from right to left: curRow[c] = curRow[c+1] + prevRow[c]
       d. Update prevRow = curRow
    3. Return prevRow[0]
    
    Time: O(n*m)
    Space: O(m) - only two rows (prevRow and curRow)
    """
    prevRow = [0] * cols
    
    # Process from bottom row to top row
    for r in range(rows - 1, -1, -1):
        curRow = [0] * cols
        curRow[cols - 1] = 1  # Rightmost column always 1
        
        # Fill from right to left
        for c in range(cols - 2, -1, -1):
            curRow[c] = curRow[c + 1] + prevRow[c]
        
        prevRow = curRow  # Move to next row
    
    return prevRow[0]


def demo_2d_dp():
    """
    Demonstrate 2-Dimension DP for unique paths problem.
    """
    print("=== 2-Dimension Dynamic Programming Demo ===\n")
    
    rows, cols = 3, 3
    print(f"Problem: Count unique paths in {rows}x{cols} grid")
    print("Rules:")
    print("  - Start: (0, 0)")
    print("  - End: (rows-1, cols-1)")
    print("  - Only move: down or right")
    print()
    
    # Brute Force
    print("1. Brute Force (Recursion without DP):")
    result = brute_force_unique_paths(0, 0, rows, cols)
    print(f"   Unique paths: {result}")
    print(f"   Time: O(2^(n+m)) = O(2^6) - exponential ❌")
    print(f"   Space: O(n+m) - recursion stack")
    print()
    
    # Memoization
    print("2. Memoization (Top-down DP with 2D cache):")
    cache = [[0] * cols for _ in range(rows)]
    result = memoization_unique_paths(0, 0, rows, cols, cache)
    print(f"   Unique paths: {result}")
    print(f"   Cache:")
    for row in cache:
        print(f"     {row}")
    print(f"   Time: O(n*m) = O({rows}*{cols}) = O({rows*cols}) ✓")
    print(f"   Space: O(n*m) - 2D cache + recursion stack")
    print()
    
    # Tabulation
    print("3. Tabulation (Bottom-up DP with full 2D array):")
    result = tabulation_unique_paths_full(rows, cols)
    print(f"   Unique paths: {result}")
    print(f"   Time: O(n*m) = O({rows*cols}) ✓")
    print(f"   Space: O(n*m) - full 2D array")
    print()
    
    # Space-Optimized
    print("4. Space-Optimized Tabulation:")
    result = dp_optimized_unique_paths(rows, cols)
    print(f"   Unique paths: {result}")
    print(f"   Time: O(n*m) = O({rows*cols}) ✓")
    print(f"   Space: O(m) = O({cols}) - only one row! ✓")
    print()
    
    print("=== Key Concepts ===")
    print("1. 2-D DP: Subproblems depend on two variables (row, column)")
    print("2. Memoization: Cache results in 2D array")
    print("3. Tabulation: Build table bottom-up")
    print("4. Space optimization: Only need previous row → O(m) space")
    print()
    
    print("=== Comparison ===")
    print("Approach        | Time      | Space     | Notes")
    print("----------------|-----------|-----------|----------------------")
    print("Brute Force     | O(2^(n+m))| O(n+m)    | Repeated calc ❌")
    print("Memoization     | O(n*m)    | O(n*m)    | Top-down, 2D cache")
    print("Tabulation      | O(n*m)    | O(n*m)    | Bottom-up, full 2D")
    print("Space-Optimized| O(n*m)    | O(m)      | Bottom-up, one row ✓")
    print()
    
    print("=== Why Space Optimization Works ===")
    print("To calculate row r, we only need:")
    print("  - Values from row r+1 (prevRow)")
    print("  - Values from same row r (curRow, right to left)")
    print("So we can reuse space instead of storing all rows!")


if __name__ == "__main__":
    demo()
    print("\n" + "="*50 + "\n")
    demo_2d_dp()