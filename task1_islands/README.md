# Task 1 — Counting Islands

Count the number of islands in an `M x N` binary matrix, where `1` is land and `0` is ocean.

## Connectivity

Cells are connected **horizontally and vertically only** (4-connectivity). Diagonal cells
are *not* connected. This follows from test case 2:

```
0 0 0 1
0 0 1 0      -> 3 islands (with diagonals it would be 1)
0 1 0 0
```

## Algorithm

Flood fill with iterative BFS:

1. Scan the matrix cell by cell.
2. When an unvisited land cell is found, increment the counter and start a BFS from it.
3. BFS marks every land cell reachable through the 4 neighbours as visited, so the whole
   island is counted exactly once.

**Complexity:** `O(M*N)` time, `O(M*N)` memory. Every cell has to be read at least once,
so this is asymptotically optimal.

An iterative queue (`collections.deque`) is used instead of recursive DFS: a recursive
solution hits Python's recursion limit (~1000) on large islands, e.g. a 1000x1000 matrix
full of land. The input matrix is not modified.

## Usage

Requires Python 3.9+. No third-party dependencies for the solution itself.

Input format (as in the task):

```
M N
<M rows of N numbers>
```

Run with an input file:

```bash
python islands.py examples/example1.in.txt
# 2
```

or via stdin (Linux / macOS):

```bash
python islands.py < examples/example1.in.txt
```

> On Windows PowerShell 5.1, pass the file as an argument instead of piping — piping adds
> a BOM character to the input.

## Tests

```bash
pip install pytest
python -m pytest -v
```

The test suite covers:

- the three test cases from the task (read from `examples/*.in.txt` / `*.out.txt`);
- shapes that break if any of the 4 directions is missing (line, column, U-shape, L-shape);
- tricky shapes: ring with a lake, checkerboard (diagonals), mixed sides and corners;
- edge cases: empty matrix, all ocean, all land, 1000x1000 land (no recursion error),
  input matrix is not modified;
- input parsing, including a matrix with the wrong number of cells.

To add a new example, put `name.in.txt` and `name.out.txt` into `examples/` — it is picked
up automatically.

## Files

```
islands.py        solution: count_islands(), parse_input(), CLI
test_islands.py   pytest tests
examples/         input / expected output pairs
```
