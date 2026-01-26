"""
Introduction to Graphs

A graph is a data structure with nodes (vertices) connected by edges.
Unlike trees, graphs have no restrictions - nodes can connect to any number
of other nodes, and edges can form cycles.

Graph Terminology:
- Vertices (V): Nodes in the graph
- Edges (E): Connections between vertices
- Maximum edges: E ≤ V² (each vertex can connect to every other)

Types:
- Directed: Edges have direction (A→B ≠ B→A)
- Undirected: Edges are bidirectional (A—B = A↔B)

Representations:
1. Matrix: 2D grid for position-based graphs
2. Adjacency Matrix: O(V²) space, good for dense graphs
3. Adjacency List: O(V + E) space, good for sparse graphs (most common)

Time: O(V + E) for traversal
Space: O(V + E) for adjacency list, O(V²) for adjacency matrix
"""


# ============================================================================
# Representation 1: Matrix (2D Grid)
# ============================================================================

def matrix_example():
    """
    Matrix representation for grid-based graphs.
    
    Used for problems like:
    - Number of Islands
    - Word Search
    - Path finding in grids
    
    Space: O(n * m) where n=rows, m=columns
    """
    grid = [[0, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0]]
    
    # Access element at row 1, column 0
    value = grid[1][0]  # Returns 1
    
    # Traverse: can move up, down, left, right
    # Connected 0s form connected components (graph)
    return grid


# ============================================================================
# Representation 2: Adjacency Matrix
# ============================================================================

def adjacency_matrix_example():
    """
    Adjacency matrix representation.
    
    adjMatrix[v1][v2] = 1 means edge exists from v1→v2
    adjMatrix[v1][v2] = 0 means no edge from v1→v2
    
    Space: O(V²) - square matrix
    Best for: Dense graphs (many edges)
    """
    adjMatrix = [[0, 0, 0, 0],
                 [1, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 1, 0, 0]]
    
    # Check if edge exists from vertex 1 to vertex 2
    if adjMatrix[1][2] == 1:
        print("Edge exists from 1 to 2")
    else:
        print("No edge from 1 to 2")
    
    # Check if edge exists from vertex 2 to vertex 3
    if adjMatrix[2][3] == 1:
        print("Edge exists from 2 to 3")
    
    return adjMatrix


# ============================================================================
# Representation 3: Adjacency List (Most Common)
# ============================================================================

class GraphNode:
    """
    Node in a graph with list of neighbors.
    
    This is the most common representation in interviews.
    Each vertex stores a list of adjacent vertices.
    
    Space: O(V + E) - only stores edges that exist
    Best for: Sparse graphs (few edges)
    """
    def __init__(self, val):
        self.val = val
        self.neighbors = []  # List of adjacent vertices
    
    def add_neighbor(self, neighbor):
        """Add a neighbor to this node."""
        self.neighbors.append(neighbor)
    
    def __repr__(self):
        """String representation for debugging."""
        neighbor_vals = [n.val for n in self.neighbors]
        return f"GraphNode({self.val}, neighbors={neighbor_vals})"


def adjacency_list_example():
    """
    Adjacency list representation example.
    
    Graph:
        A → B
        ↓   ↓
        C   D
    """
    # Create vertices
    A = GraphNode("A")
    B = GraphNode("B")
    C = GraphNode("C")
    D = GraphNode("D")
    
    # Add edges (directed graph)
    A.neighbors = [B, C]  # A → B, A → C
    B.neighbors = [D]     # B → D
    C.neighbors = []      # C has no neighbors
    D.neighbors = []      # D has no neighbors
    
    return A  # Return root/starting node


def demo():
    """
    Demonstrate different graph representations.
    """
    print("=== Introduction to Graphs ===\n")
    
    print("1. Matrix Representation (2D Grid)")
    grid = matrix_example()
    print(f"   Grid: {grid}")
    print(f"   Space: O(n * m) = O({len(grid)} * {len(grid[0])})")
    print(f"   Used for: Grid-based problems (Number of Islands, etc.)")
    print()
    
    print("2. Adjacency Matrix Representation")
    adjMatrix = adjacency_matrix_example()
    print(f"   Matrix: {adjMatrix}")
    print(f"   Space: O(V²) = O({len(adjMatrix)}²) = O({len(adjMatrix)**2})")
    print(f"   Best for: Dense graphs (many edges)")
    print()
    
    print("3. Adjacency List Representation (Most Common)")
    root = adjacency_list_example()
    print(f"   Graph structure:")
    print(f"     {root}")
    print(f"     {root.neighbors[0]}")
    print(f"     {root.neighbors[1]}")
    print(f"     {root.neighbors[0].neighbors[0]}")
    print(f"   Space: O(V + E) = O(4 + 3) = O(7)")
    print(f"   Best for: Sparse graphs (few edges)")
    print()
    
    print("=== Graph Terminology ===")
    print("Vertices (V): Nodes in the graph")
    print("Edges (E): Connections between vertices")
    print("Maximum edges: E ≤ V²")
    print("Complete graph: V vertices, V² edges")
    print()
    
    print("=== Directed vs Undirected ===")
    print("Directed: Edges have direction (A→B ≠ B→A)")
    print("Undirected: Edges are bidirectional (A—B = A↔B)")
    print("Trees and linked lists are directed graphs")
    print()
    
    print("=== Representation Comparison ===")
    print("Representation    | Space      | Best For")
    print("------------------|------------|------------")
    print("Matrix            | O(n*m)     | Grid problems")
    print("Adjacency Matrix  | O(V²)      | Dense graphs")
    print("Adjacency List    | O(V+E)     | Sparse graphs ✓")
    print()
    print("Adjacency List is most common in interviews!")


# ============================================================================
# Matrix DFS (Depth-First Search)
# ============================================================================

def matrix_dfs_count_paths(grid, r, c, visit):
    """
    Count unique paths from (r,c) to bottom-right using DFS with backtracking.
    
    Problem: Count paths from top-left to bottom-right that:
    - Only move along 0s (not 1s)
    - Don't visit same cell twice
    
    Algorithm:
    1. Check base cases:
       - Out of bounds → return 0
       - Already visited → return 0
       - Obstacle (cell == 1) → return 0
       - Reached destination → return 1
    2. Mark current cell as visited
    3. Try all 4 directions (up, down, left, right)
    4. Sum results from all directions
    5. Unmark current cell (backtrack)
    6. Return total count
    
    Time: O(4^(n*m)) - exponential, 4 choices at each cell
    Space: O(n*m) - recursion stack + visited set
    """
    ROWS, COLS = len(grid), len(grid[0])
    
    # Base case 1: Invalid path (return 0)
    if (min(r, c) < 0 or           # Out of bounds (negative)
        r == ROWS or c == COLS or  # Out of bounds (too large)
        (r, c) in visit or         # Already visited
        grid[r][c] == 1):          # Obstacle
        return 0
    
    # Base case 2: Reached destination (return 1)
    if r == ROWS - 1 and c == COLS - 1:
        return 1  # Found valid path!
    
    # Mark as visited
    visit.add((r, c))
    
    # Try all 4 directions and sum results
    count = 0
    count += matrix_dfs_count_paths(grid, r + 1, c, visit)  # Down
    count += matrix_dfs_count_paths(grid, r - 1, c, visit)  # Up
    count += matrix_dfs_count_paths(grid, r, c + 1, visit)  # Right
    count += matrix_dfs_count_paths(grid, r, c - 1, visit)  # Left
    
    # Backtrack: unmark current cell
    # This allows exploring other paths that might use this cell
    visit.remove((r, c))
    
    return count


def demo_matrix_dfs():
    """
    Demonstrate matrix DFS for counting unique paths.
    """
    print("=== Matrix DFS Demo ===\n")
    
    # Example matrix
    grid = [[0, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0]]
    
    print("Matrix:")
    for row in grid:
        print(f"  {row}")
    print()
    print("Problem: Count unique paths from (0,0) to (3,3)")
    print("Rules:")
    print("  - Only move along 0s (not 1s)")
    print("  - Don't visit same cell twice")
    print("  - Can move: up, down, left, right")
    print()
    
    # Count paths
    visit = set()
    result = matrix_dfs_count_paths(grid, 0, 0, visit)
    
    print(f"Result: {result} unique paths")
    print()
    
    print("=== Key Concepts ===")
    print("1. DFS explores all paths recursively")
    print("2. Backtracking: mark → explore → unmark")
    print("3. Hash set for O(1) visited lookup")
    print("4. Base cases: invalid → 0, destination → 1")
    print("5. Try all 4 directions: up, down, left, right")
    print()
    
    print("=== Time Complexity ===")
    print("Time: O(4^(n*m)) - exponential")
    print("  - 4 choices at each cell")
    print("  - Decision tree with branching factor 4")
    print("  - Height = n*m (worst case)")
    print()
    print("Space: O(n*m)")
    print("  - Recursion stack: O(n*m)")
    print("  - Visited set: O(n*m)")
    print()
    
    print("=== Why Hash Set? ===")
    print("Option 1: List")
    print("  visit = [(0,0), (0,1), ...]")
    print("  Check: (r,c) in visit → O(n) ❌")
    print()
    print("Option 2: Hash Set ✓")
    print("  visit = {(0,0), (0,1), ...}")
    print("  Check: (r,c) in visit → O(1) ✓")
    print()
    print("Option 3: 2D Boolean Array")
    print("  visited[r][c] = True/False")
    print("  Check: visited[r][c] → O(1) ✓")
    print("  But needs O(n*m) space upfront")


# ============================================================================
# Matrix BFS (Breadth-First Search)
# ============================================================================

from collections import deque


def matrix_bfs_shortest_path(grid):
    """
    Find shortest path length from (0,0) to bottom-right using BFS.
    
    Problem: Find length of shortest path from top-left to bottom-right that:
    - Only moves along 0s (not 1s)
    - Can move up, down, left, right
    
    Algorithm:
    1. Initialize queue with start cell (0,0), mark as visited
    2. Process level by level:
       a. Process all cells at current level
       b. For each cell, check if destination reached
       c. Add valid neighbors to queue
       d. Mark neighbors as visited immediately (prevents duplicates)
    3. Increment length after processing each level
    4. Return length when destination reached
    
    Why BFS for shortest path?
    - BFS explores level by level (by distance)
    - First path found is guaranteed shortest
    - More efficient than DFS (O(n*m) vs O(4^(n*m)))
    
    Time: O(n*m) - visit each cell at most once
    Space: O(n*m) - queue + visited set
    """
    ROWS, COLS = len(grid), len(grid[0])
    visit = set()  # Track visited cells (hash set for O(1) lookup)
    queue = deque()  # Queue for BFS
    
    # Start from top-left
    queue.append((0, 0))
    visit.add((0, 0))
    
    length = 0  # Track path length (starts at 0)
    
    while queue:
        # Process all cells at current level
        # This ensures we explore by distance from start
        for _ in range(len(queue)):
            r, c = queue.popleft()
            
            # Check if reached destination
            if r == ROWS - 1 and c == COLS - 1:
                return length  # Shortest path found!
            
            # Try all 4 directions
            # Directions: [right, left, down, up]
            neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            for dr, dc in neighbors:
                new_r, new_c = r + dr, c + dc
                
                # Check if valid (same checks as DFS)
                if (min(new_r, new_c) < 0 or           # Out of bounds (negative)
                    new_r == ROWS or new_c == COLS or   # Out of bounds (too large)
                    (new_r, new_c) in visit or          # Already visited
                    grid[new_r][new_c] == 1):           # Obstacle
                    continue  # Skip invalid cells
                
                # Add to queue and mark as visited immediately
                # Marking immediately prevents duplicate entries in queue
                queue.append((new_r, new_c))
                visit.add((new_r, new_c))
        
        length += 1  # Move to next level (increment distance)
    
    return -1  # No path exists


def demo_matrix_bfs():
    """
    Demonstrate matrix BFS for finding shortest path.
    """
    print("=== Matrix BFS Demo ===\n")
    
    # Example matrix
    grid = [[0, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0]]
    
    print("Matrix:")
    for row in grid:
        print(f"  {row}")
    print()
    print("Problem: Find shortest path length from (0,0) to (3,3)")
    print("Rules:")
    print("  - Only move along 0s (not 1s)")
    print("  - Can move: up, down, left, right")
    print()
    
    # Find shortest path
    result = matrix_bfs_shortest_path(grid)
    
    print(f"Result: Shortest path length = {result}")
    print()
    
    print("=== BFS vs DFS for Shortest Path ===")
    print("Aspect        | DFS                    | BFS")
    print("--------------|------------------------|----------------")
    print("Path finding  | Explores all paths     | Finds shortest first")
    print("Time          | O(4^(n*m)) exponential | O(n*m) linear")
    print("Space         | O(n*m) recursion       | O(n*m) queue")
    print("Best for      | Count paths, explore   | Shortest path ✓")
    print()
    
    print("=== Key Concepts ===")
    print("1. BFS processes level by level (by distance)")
    print("2. First path found is guaranteed shortest")
    print("3. Mark cells as visited when adding to queue")
    print("4. Use directions array for 4 directions")
    print("5. Increment length after processing each level")
    print()
    
    print("=== Why Mark Visited Immediately? ===")
    print("Wrong: Mark when processing")
    print("  → Same cell can be added multiple times ❌")
    print("  → Queue: [(0,1), (1,0), (0,1)] ← Duplicate!")
    print()
    print("Correct: Mark when adding to queue ✓")
    print("  → Each cell added at most once")
    print("  → Queue: [(0,1), (1,0)] ← No duplicates")
    print()
    
    print("=== Time Complexity ===")
    print("Time: O(n*m)")
    print("  - Visit each cell at most once")
    print("  - Process each cell exactly once")
    print("  - Much better than DFS: O(4^(n*m))")
    print()
    print("Space: O(n*m)")
    print("  - Queue: O(n*m) worst case")
    print("  - Visited set: O(n*m) worst case")


if __name__ == "__main__":
    demo()
    print("\n" + "="*50 + "\n")
    demo_matrix_dfs()
    print("\n" + "="*50 + "\n")
    demo_matrix_bfs()