"""Tests for islands.py. Run from the task1_islands folder: python -m pytest -v"""

from pathlib import Path

import pytest

from islands import count_islands, parse_input

EXAMPLES_DIR = Path(__file__).parent / "examples"


# --- Test cases from the task description (examples/*.in.txt) -------------

@pytest.mark.parametrize(
    "input_file",
    sorted(EXAMPLES_DIR.glob("*.in.txt")),
    ids=lambda p: p.name,
)
def test_task_examples(input_file):
    expected_file = input_file.with_name(input_file.name.replace(".in.txt", ".out.txt"))
    grid = parse_input(input_file.read_text(encoding="utf-8"))
    expected = int(expected_file.read_text(encoding="utf-8"))
    assert count_islands(grid) == expected


# --- Each shape breaks if one of the 4 directions is missing --------------

@pytest.mark.parametrize(
    "grid",
    [
        pytest.param([[1, 1]], id="needs-right"),
        pytest.param([[1],
                      [1]], id="needs-down"),
        pytest.param([[1, 0, 1],
                      [1, 1, 1]], id="needs-up (U-shape)"),
        pytest.param([[0, 1],
                      [1, 1]], id="needs-left (L-shape)"),
    ],
)
def test_all_four_directions_are_used(grid):
    assert count_islands(grid) == 1


# --- Tricky shapes ---------------------------------------------------------

def test_ring_with_lake_is_one_island():
    grid = [[1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]]
    assert count_islands(grid) == 1


def test_diagonals_do_not_connect():
    grid = [[1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]]
    assert count_islands(grid) == 5


def test_mixed_sides_and_corners():
    grid = [[1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]]
    assert count_islands(grid) == 3


# --- Edge cases --------------------------------------------------------------

def test_empty_grid():
    assert count_islands([]) == 0
    assert count_islands([[]]) == 0


def test_all_ocean():
    assert count_islands([[0] * 5 for _ in range(4)]) == 0


def test_all_land():
    assert count_islands([[1] * 5 for _ in range(4)]) == 1


def test_large_grid_no_recursion_error():
    # 1000x1000 of land = one island with 1,000,000 cells.
    # A recursive DFS would crash here; iterative BFS must not.
    grid = [[1] * 1000 for _ in range(1000)]
    assert count_islands(grid) == 1


def test_input_grid_is_not_modified():
    grid = [[1, 0],
            [0, 1]]
    count_islands(grid)
    assert grid == [[1, 0], [0, 1]]


# --- Input parsing -----------------------------------------------------------

def test_parse_input():
    text = "3 3\n0 1 0\n0 0 0\n0 1 1\n"
    assert parse_input(text) == [[0, 1, 0], [0, 0, 0], [0, 1, 1]]


def test_parse_input_wrong_size():
    with pytest.raises(ValueError):
        parse_input("2 2\n1 0 1\n")
