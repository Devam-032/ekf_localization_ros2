#!/usr/bin/env python3

import numpy as np
from scipy.optimize import linear_sum_assignment
import math

def normalize_angle(angle):
    """Normalize angle to the range [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def compute_pairwise_costs(mat1, mat2, lam=0.5):
    """
    Build a cost matrix for all pairs (i, j) where i indexes columns of mat1
    and j indexes columns of mat2.
    Both mat1 and mat2 are 2 x m and 2 x n respectively (each column = [dist, angle]).
    
    cost(i,j) = lam * |d1 - d2| + (1 - lam) * |angle1 - angle2|
    """
    m = mat1.shape[1]
    n = mat2.shape[1]
    cost_mat = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            dist_diff = abs(mat1[0, i] - mat2[0, j])
            angle_diff = abs(normalize_angle(mat1[1, i]) - normalize_angle(mat2[1, j]))
            cost_mat[i, j] = lam * dist_diff + (1 - lam) * angle_diff
    return cost_mat

def match_matrices(mat1, mat2, lam=0.5, match_threshold=0.8, large_cost=999999.0):
    """
    Match columns of mat1 (2 x m) with columns of mat2 (2 x n) using the Hungarian algorithm.
    A pair is accepted only if its cost is below match_threshold.
    
    The results are stored in instance-like variables (here returned as a tuple):
      - matched_mat2: 2 x m array, where for each column i in mat1:
            if a match with cost <= threshold was found, matched_mat2[:, i] contains
            the corresponding column from mat2; otherwise, the column is all zeros.
      - unmatched_mat1: 2 x X array of columns from mat1 that remain unmatched.
      - unmatched_mat2: 2 x Y array of columns from mat2 that remain unmatched.
      - match_vector: length m array; match_vector[i] = j if mat1 col i is matched with mat2 col j (and cost <= threshold), else -1.
      - match_vector_curr: length n array; for each col j in mat2, if matched then the corresponding
                                index from mat1 is stored; otherwise -10.
      - match_vector_curr_2: an array containing the indices from mat2 (only valid matches)
                                sorted by their corresponding matched index from mat1.
    
    If m != n, dummy rows/columns (with a high cost) are added so that the Hungarian algorithm produces a complete assignment.
    """
    m = mat1.shape[1]
    n = mat2.shape[1]
    
    # Compute raw cost matrix
    cost_matrix_raw = compute_pairwise_costs(mat1, mat2, lam=lam)
    
    # Create square cost matrix with dummy entries set to large_cost.
    size = max(m, n)
    cost_matrix = np.full((size, size), large_cost, dtype=float)
    cost_matrix[:m, :n] = cost_matrix_raw
    
    # Solve assignment using Hungarian algorithm.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Initialize match vectors and matched matrix.
    match_vector = np.full(m, -1, dtype=int)       # For mat1: index in mat2 if matched; -1 if unmatched.
    matched_mat2 = np.zeros_like(mat1)               # Reordered columns of mat2, aligned with mat1.
    match_vector_curr = np.full(n, -10, dtype=int)     # For mat2: index in mat1 if matched; -10 if unmatched.
    
    # Sets for tracking used indices.
    used_rows_mat1 = set()
    used_cols_mat2 = set()
    
    # Lists for unmatched indices.
    unmatched_mat1_indices = []
    unmatched_mat2_indices = []
    
    # Process assignment.
    for (i, j) in zip(row_ind, col_ind):
        if i < m and j < n:
            if cost_matrix[i, j] <= match_threshold:
                match_vector[i] = j
                matched_mat2[:, i] = mat2[:, j]
                used_rows_mat1.add(i)
                used_cols_mat2.add(j)
                match_vector_curr[j] = i
            else:
                unmatched_mat1_indices.append(i)
                unmatched_mat2_indices.append(j)
        elif i < m and j >= n:
            unmatched_mat1_indices.append(i)
        elif i >= m and j < n:
            unmatched_mat2_indices.append(j)
        # else: dummy to dummy, no effect.
    
    # Any indices not assigned in the valid ranges are marked as unmatched.
    for i in range(m):
        if i not in used_rows_mat1 and i not in unmatched_mat1_indices:
            unmatched_mat1_indices.append(i)
    for j in range(n):
        if j not in used_cols_mat2 and j not in unmatched_mat2_indices:
            unmatched_mat2_indices.append(j)
    
    unmatched_mat1 = mat1[:, unmatched_mat1_indices] if unmatched_mat1_indices else np.zeros((2, 0))
    unmatched_mat2 = mat2[:, unmatched_mat2_indices] if unmatched_mat2_indices else np.zeros((2, 0))
    
    # Build match_vector_curr_2: sorted valid z_curr indices by their associated mat1 index.
    valid_matches = [(match_vector_curr[j], j) for j in range(n) if match_vector_curr[j] != -10]
    valid_matches_sorted = sorted(valid_matches, key=lambda x: x[0])
    match_vector_curr_2 = np.array([j for (i, j) in valid_matches_sorted])
    
    # Store as instance-like variables.
    results = {
        "matched_mat2": matched_mat2,
        "unmatched_mat1": unmatched_mat1,
        "unmatched_mat2": unmatched_mat2,
        "match_vector": match_vector,
        "match_vector_curr": match_vector_curr,
        "match_vector_curr_2": match_vector_curr_2
    }
    
    return (matched_mat2, unmatched_mat1, unmatched_mat2, match_vector,
            match_vector_curr, match_vector_curr_2)

# =========================== EXAMPLE USAGE / TESTING ===========================

if __name__ == "__main__":
    # ---------- Test Case 1 ----------
    # mat1:
    #   [2,   3.5, 8,   3]
    #   [1.2, 1.5, 0,  -1.5]
    mat1_case1 = np.array([
        [2.0, 3.5, 8.0, 3.0],
        [1.2, 1.5, 0.0, -1.5]
    ])
    # mat2:
    #   [1.9, 2.87, 3.8, 8.1]
    #   [1.3, -1.49, 1.46, -0.02]
    mat2_case1 = np.array([
        [1.9,  2.87, 3.8,   8.1],
        [1.3, -1.49, 1.46, -0.02]
    ])
    res1 = match_matrices(mat1_case1, mat2_case1, lam=0.5, match_threshold=0.8)
    matched1, unmatch1_1, unmatch1_2, mv1, mvec_curr1, mvec_curr_2_1 = res1
    print("=== Test Case 1 ===")
    print("Matched mat2:\n", np.array2string(matched1, separator=", "))
    print("Unmatched from mat1:\n", np.array2string(unmatch1_1, separator=", "))
    print("Unmatched from mat2:\n", np.array2string(unmatch1_2, separator=", "))
    print("Match vector (mat1 -> mat2):", mv1)
    print("Match vector for mat2 (unmatched as -10):", mvec_curr1)
    print("Sorted match vector for mat2 (valid matches sorted by mat1 index):", mvec_curr_2_1)

    # ---------- Test Case 2 ----------
    # mat1:
    #   [7, 9, 4]
    #   [3, 2, -3]
    mat1_case2 = np.array([
        [7.0, 9.0, 4.0],
        [3.0, 2.0, -3.0]
    ])
    # mat2:
    #   [4,    9,    7.2,  2.87]
    #   [-3.02,1.95, 3.1,  -1.49]
    mat2_case2 = np.array([
        [4.0,   9.0,   2.87,  7.2],
        [-3.02, 1.95,  -1.49, 3.1]
    ])
    res2 = match_matrices(mat1_case2, mat2_case2, lam=0.5, match_threshold=0.8)
    matched2, unmatch2_1, unmatch2_2, mv2, mvec_curr2, mvec_curr_2_2 = res2
    print("\n=== Test Case 2 ===")
    print("Matched mat2:\n", np.array2string(matched2, separator=", "))
    print("Unmatched from mat1:\n", np.array2string(unmatch2_1, separator=", "))
    print("Unmatched from mat2:\n", np.array2string(unmatch2_2, separator=", "))
    print("Match vector (mat1 -> mat2):", mv2)
    print("Match vector for mat2 (unmatched as -10):", mvec_curr2)
    print("Sorted match vector for mat2 (valid matches sorted by mat1 index):", mvec_curr_2_2)

    # ---------- Test Case 3 ----------
    # mat1:
    #   [6.8, 0.5, 5.2, 9.1]
    #   [2,   0,   2.98, 3.1]
    mat1_case3 = np.array([
        [6.8, 0.5, 5.2, 9.1],
        [2.0, 0.0, 2.98, 3.1]
    ])
    # mat2:
    #   [5.9, 18.5, 17.0, 10.49]
    #   [3.0, 1.7, 2.0, 0.05]
    mat2_case3 = np.array([
        [5.9, 18.5, 17.0, 10.49],
        [3.0, 1.7, 2.0, 0.05]
    ])
    res3 = match_matrices(mat1_case3, mat2_case3, lam=0.5, match_threshold=0.8)
    matched3, unmatch3_1, unmatch3_2, mv3, mvec_curr3, mvec_curr_2_3 = res3
    print("\n=== Test Case 3 ===")
    print("Matched mat2:\n", np.array2string(matched3, separator=", "))
    print("Unmatched from mat1:\n", np.array2string(unmatch3_1, separator=", "))
    print("Unmatched from mat2:\n", np.array2string(unmatch3_2, separator=", "))
    print("Match vector (mat1 -> mat2):", mv3)
    print("Match vector for mat2 (unmatched as -10):", mvec_curr3)
    print("Sorted match vector for mat2 (valid matches sorted by mat1 index):", mvec_curr_2_3)
