import numpy as np

def example_append_mat(match_vector, matched_curr):
    """
    This function simulates the process of building z_sub_mat from matched_curr,
    using a flag to check if it's the first valid assignment.
    
    Parameters:
      - match_vector: a list/array where each element corresponds to a match for a column
                      in the "previous" matrix (e.g., if match_vector[i] == -10 then that
                      column is considered unmatched).
      - matched_curr: a 2 x N matrix where each column is the candidate measurement 
                      from the current matrix.
                      
    Returns:
      - z_sub_mat: a 2 x K matrix containing the columns from matched_curr corresponding
                   to valid matches, appended in order.
    """
    # Initialize z_sub_mat as a (2,1) array with uninitialized values.
    z_sub_mat = np.empty((2, 1))
    first_assignment = True

    for i in range(len(match_vector)):
        # Check if the match is valid (i.e. not equal to -10)
        if match_vector[i] == -10:
            # Unmatched: do nothing (or, optionally, log/collect this index separately)
            continue
        else:
            if first_assignment:
                # For the first valid match, directly assign to the first column.
                z_sub_mat[:, 0] = matched_curr[:, i]
                first_assignment = False
            else:
                # For subsequent valid matches, append the column to z_sub_mat.
                z_sub_mat = np.concatenate((z_sub_mat, matched_curr[:, i:i+1]), axis=1)
    return z_sub_mat

# ------------------- Example Usage -------------------
# Suppose we have 4 potential measurements with the following match vector:
# - match_vector[i] indicates a valid match if not -10.
# For example, let's assume:
#   match_vector = [0, -10, 2, 1]
# This means:
#   - The 0th measurement is valid.
#   - The 1st measurement is unmatched (indicated by -10).
#   - The 2nd measurement is valid.
#   - The 3rd measurement is valid.
#
# Let matched_curr be a 2x4 matrix where each column is a candidate:
matched_curr = np.array([
    [1, 2, 3, 4],   # Distances or first feature
    [10, 20, 30, 40] # Angles (or second feature)
])
# Expected behavior:
#   Only the columns corresponding to indices 0, 2, and 3 are taken (in that order).
#   So, the expected z_sub_mat is:
#       [[1, 3, 4],
#        [10, 30, 40]]

match_vector = [0, -10, 2, 1]

z_sub_mat_result = example_append_mat(match_vector, matched_curr)
print("Resulting z_sub_mat:")
print(z_sub_mat_result)