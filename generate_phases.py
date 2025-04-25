import math
import random
import numpy as np

def random_partition(total, parts):
    # Calculate the lower and upper bound for each part.
    lower_bound = math.ceil(total / parts / 2)  # half of average rounded up
    upper_bound = math.floor(2 * total / parts)   # double of average rounded down

    result = []
    remaining_total = total
    remaining_parts = parts

    for i in range(parts):
        # Ensure the total sum is correct by adjusting bounds for the current part.
        current_lower = max(lower_bound, remaining_total - (remaining_parts - 1) * upper_bound)
        current_upper = min(upper_bound, remaining_total - (remaining_parts - 1) * lower_bound)
        value = random.randint(current_lower, current_upper)
        result.append(value)
        remaining_total -= value
        remaining_parts -= 1
    
    random.shuffle(result)
    return result

def generate_phases(total_iterations=100, num_phases=25, seed=42):
    """
    Generate phase boundaries for iterations.
    
    Args:
        total_iterations: Total number of iterations (default: 100)
        num_phases: Number of phases to generate (default: 25)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        List of phase boundaries (24 points that divide 100 iterations into 25 phases)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate random partition sizes
    phase_sizes = random_partition(total_iterations, num_phases)
    
    # Calculate phase boundaries
    boundaries = []
    current_sum = 0
    for size in phase_sizes[:-1]:  # Exclude last size as we don't need its boundary
        current_sum += size
        boundaries.append(current_sum)
    
    return boundaries

if __name__ == "__main__":
    # Generate phase boundaries
    boundaries = generate_phases()
    
    # Print results with phase numbers
    print("Phase boundaries (24 points that divide 100 iterations into 25 phases):")
    print("Phase\tBoundary\tIterations")
    print("-" * 40)
    
    # Print first phase (0 to first boundary)
    print(f"1\t{boundaries[0]}\t0-{boundaries[0]}")
    
    # Print middle phases
    for i in range(1, len(boundaries)):
        print(f"{i+1}\t{boundaries[i]}\t{boundaries[i-1]+1}-{boundaries[i]}")
    
    # Print last phase (last boundary to 99)
    print(f"25\t99\t{boundaries[-1]+1}-99")
    
    # Verify the boundaries
    print("\nVerification:")
    print(f"Number of boundaries: {len(boundaries)}")
    print(f"First boundary: {boundaries[0]}")
    print(f"Last boundary: {boundaries[-1]}")
    print(f"All boundaries are within range [0, 99]: {all(0 <= x <= 99 for x in boundaries)}")
    
    # Calculate phase sizes
    phase_sizes = [boundaries[0]] + [boundaries[i] - boundaries[i-1] for i in range(1, len(boundaries))] + [100 - boundaries[-1]]
    print("\nPhase sizes:")
    print(phase_sizes)
    print(f"Sum of phase sizes: {sum(phase_sizes)}")


'''

Phase boundaries (24 points that divide 100 iterations into 25 phases):
Phase   Boundary        Iterations
----------------------------------------
1       3       0-3
2       5       4-5
3       8       6-8
4       12      9-12
5       17      13-17
6       19      18-19
7       21      20-21
8       23      22-23
9       25      24-25
10      28      26-28
11      35      29-35
12      37      36-37
13      40      38-40
14      46      41-46
15      53      47-53
16      57      54-57
17      63      58-63
18      70      64-70
19      72      71-72
20      75      73-75
21      82      76-82
22      89      83-89
23      91      90-91
24      97      92-97
25      99      98-99
'''