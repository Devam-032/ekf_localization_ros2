import numpy as np

def resample_wheel(particles, weights):
    N = len(particles)
    new_particles = []
    
    # Step 1: Compute the cumulative sum of weights
    cumulative_weights = np.cumsum(weights)
    
    # Step 2: Select a random starting point
    start = np.random.uniform(0, 1/N)
    
    # Step 3: Define step size
    step = 1 / N
    
    # Step 4: Select particles using the resampling wheel
    index = 0
    for i in range(N):
        u = start + i * step
        while u > cumulative_weights[index]:
            index += 1
        new_particles.append(particles[index])
    
    return new_particles

# Example usage
particles = ['A', 'B', 'C', 'D']
weights = [0.1, 0.3, 0.4, 0.2]  # Normalized weights

resampled_particles = resample_wheel(particles, weights)
print(resampled_particles)
