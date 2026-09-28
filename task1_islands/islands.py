"""Counting islands in a binary matrix.

An island is a group of 1-cells connected horizontally or vertically
(4-connectivity). Diagonal neighbours are NOT connected - this follows
from test case 2, where three diagonally touching cells give 3 islands.
"""

from collections import deque

# Offsets to the 4 neighbours: up, down, left, right.
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def count_islands(grid: list[list[int]]) -> int:
    """Return the number of islands in ``grid``.

    Uses iterative BFS (flood fill).

    Time:  O(M*N) - every cell is visited a constant number of times.
    Memory: O(M*N) - for the ``visited`` matrix and the queue.
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    islands = 0

    for r in range(rows):
        for c in range(cols):
            # Skip ocean and cells that already belong to a counted island.
            if grid[r][c] != 1 or visited[r][c]:
                continue

            # Found a new island - flood it so we never count it again.
            islands += 1
            visited[r][c] = True
            queue = deque([(r, c)])

            while queue:
                cr, cc = queue.popleft()
                for dr, dc in DIRECTIONS:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and grid[nr][nc] == 1
                        and not visited[nr][nc]
                    ):
                        visited[nr][nc] = True
                        queue.append((nr, nc))

    return islands


def parse_input(text: str) -> list[list[int]]:
    """Parse the task input format: first line ``M N``, then M rows of N numbers."""
    numbers = [int(x) for x in text.split()]
    m, n = numbers[0], numbers[1]
    cells = numbers[2:]
    if len(cells) != m * n:
        raise ValueError(f"Expected {m * n} cells for a {m}x{n} matrix, got {len(cells)}")
    return [cells[i * n:(i + 1) * n] for i in range(m)]


if __name__ == "__main__":
    import sys

    # Usage: python islands.py input.txt   or   python islands.py < input.txt
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    grid = parse_input(text)
    print(count_islands(grid))
