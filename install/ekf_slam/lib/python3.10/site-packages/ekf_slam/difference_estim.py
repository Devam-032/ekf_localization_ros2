import numpy as np
import math

# Original coordinate arrays.
aray1 = np.array([
    [1.3014, -1.74091],
    [1.98771, 1.13637],
    [-1.13443, -0.902541],
    [-1.37644, 2.26725],
    [3.5403, -1.8486],
    [1.05571, 3.07389],
    [3.58534, 2.92355],
    [4.6648, 1.68058],
    [4.81043, -2.61818],
    [3.30091, -4.2055],
    [0.414463, -4.33209],
    [-1.32098, -3.92076],
    [-2.85487, -2.88071],
    [-3.23012, 0.1117],
    [-3.15908, 3.68903],
    [-0.413215, 3.60968],
    [-1.80846, 0.902126]
])

array2 = np.array([
    [-2.8999998569488525, -2.830000400543213],
    [-3.1999998092651367, 3.7100000381469727],
    [-1.340000033378601, -3.9100005626678467],
    [4.780000686645508, -2.5900001525878906],
    [3.580000400543213, 2.929999828338623],
    [-1.3999998569488525, 2.2699999809265137],
    [0.4000001549720764, -4.330000400543213],
    [2.020000696182251, 1.1899998188018799],
    [4.840000152587891, -3.7300002574920654],
    [3.4600000381469727, -1.8700004816055298],
    [-1.1599997282028198, -0.910000205039978],
    [3.2200005054473877, -4.150000095367432],
    [1.0000001192092896, 3.109999895095825],
    [-3.259999990463257, 0.10999985039234161],
    [4.720000743865967, 1.6699997186660767],
    [-1.7599997520446777, 0.8299999237060547],
    [-0.3799997568130493, 3.5899999141693115]
])

# --- Append an extra coordinate pair to demonstrate the threshold ---
# These extra pairs are chosen so that their distance is greater than the threshold.
aray1 = np.vstack((aray1, np.array([100, 100])))
array2 = np.vstack((array2, np.array([105, 105])))

# Define the threshold for accepting a match.
threshold = 1.0

# Create the distance matrix between each pair in aray1 and array2.
num_aray1 = aray1.shape[0]
num_array2 = array2.shape[0]
dist = np.zeros((num_aray1, num_array2))

for i in range(num_aray1):
    for j in range(num_array2):
        # Calculate Euclidean distance between points
        x_diff = array2[j, 0] - aray1[i, 0]
        y_diff = array2[j, 1] - aray1[i, 1]
        dist[i, j] = math.sqrt(x_diff * x_diff + y_diff * y_diff)

# To store the minimum distance for each point in aray1, and corresponding index pairs.
min_distances = np.zeros(num_aray1)
corresponding_indices = []  # List to hold [aray1 index, array2 index]

for i in range(num_aray1):
    min_dist = float('inf')
    index_j = -1
    for j in range(num_array2):
        if dist[i, j] < min_dist:
            min_dist = dist[i, j]
            index_j = j
    # Apply the threshold: if the smallest distance exceeds the threshold, disregard the match.
    if min_dist > threshold:
        index_j = -1
    min_distances[i] = min_dist
    corresponding_indices.append([i, index_j])

# Convert corresponding_indices to a NumPy array for further processing.
corresponding_indices = np.array(corresponding_indices)

# --- Post-Processing Step for Duplicate Matches ---
# For cases where multiple aray1 points match to the same array2 index,
# we keep only the one with the smallest distance and set the rest to -1.
groups = {}  # key: array2 index, value: list of tuples (i from aray1, distance)
for i, pair in enumerate(corresponding_indices):
    j = pair[1]
    # Skip unmatched entries
    if j == -1:
        continue
    if j not in groups:
        groups[j] = [(i, min_distances[i])]
    else:
        groups[j].append((i, min_distances[i]))

for j, matches in groups.items():
    if len(matches) > 1:
        # Find the match with the minimum distance.
        best_i, best_dist = min(matches, key=lambda x: x[1])
        # Set all other matches for this array2 index to -1.
        for i, d in matches:
            if i != best_i:
                corresponding_indices[i, 1] = -1

# --- Determine Unmatched Indices in array2 ---
# Create a set of all indices from array2 and remove those that are matched (i.e. not -1).
matched_indices = set()
for pair in corresponding_indices:
    if pair[1] != -1:
        matched_indices.add(pair[1])

all_indices = set(range(num_array2))
unmatched_array2_indices = list(all_indices - matched_indices)
unmatched_array2_indices.sort()

# --- Output Results ---
print("Distance matrix between aray1 and array2:")
print(dist)
print("\nMinimum distances for each row in aray1:")
print(min_distances)
print("\nCorresponding index pairs (aray1 index, array2 index):")
print(corresponding_indices)
print("\nUnmatched indices in array2:")
print(unmatched_array2_indices)
