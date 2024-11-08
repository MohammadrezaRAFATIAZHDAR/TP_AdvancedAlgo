"""
Bin Packing Problem Solver Using PySCIPOpt

This program solves the bin-packing problem using the PySCIPOpt library and the SCIP solver.

The program solves the problem in two ways:
1. **Without Symmetry-Breaking**: Solves the problem without additional constraints to reduce symmetries.
2. **With Symmetry-Breaking**: Applies lexicographical constraints to avoid redundant solutions.

**Usage**:
Run the program from the command line:

python3 bin_packing_symmetry.py <instance_file.bpa>   

"""

import argparse
import time
from pyscipopt import Model, quicksum

def parse_instance(filename):
    """Parses a bin-packing problem instance from the file."""
    with open(filename, 'r') as file:
        # Read instance name
        instance_name = file.readline().strip()
        
        # Read the second line: capacity, number of objects, (third number unused)
        capacity, number_of_objects, _ = map(int, file.readline().strip().split())
        
        # Read the sizes of each object
        sizes = [int(file.readline().strip()) for _ in range(number_of_objects)]
        
    return instance_name, capacity, number_of_objects, sizes

def solve_bin_packing(filename, with_symmetry=False):
    """Solves the bin-packing problem instance defined in the given file using PySCIPOpt."""
    instance_name, capacity, number_of_objects, sizes = parse_instance(filename)
    
    # Create the model
    model = Model(f"BinPacking_{instance_name}")
    
    # Decision variables
    x = {}
    for i in range(number_of_objects):
        for j in range(number_of_objects):
            x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")
    
    y = {}
    for j in range(number_of_objects):
        y[j] = model.addVar(vtype="B", name=f"y_{j}")
    
    # Objective function: Minimize the number of boxes used
    model.setObjective(quicksum(y[j] for j in range(number_of_objects)), "minimize")
    
    # Constraints
    # Each object must be placed in exactly one box
    for i in range(number_of_objects):
        model.addCons(quicksum(x[i, j] for j in range(number_of_objects)) == 1, f"Object_{i}_placement")
    
    # Capacity constraint for each box
    for j in range(number_of_objects):
        model.addCons(quicksum(sizes[i] * x[i, j] for i in range(number_of_objects)) <= capacity * y[j], f"Box_{j}_capacity")
    
    # Linking constraint: if an object is placed in box j, then y[j] must be 1
    for i in range(number_of_objects):
        for j in range(number_of_objects):
            model.addCons(x[i, j] <= y[j], f"Linking_{i}_{j}")
    
    # Apply symmetry-breaking constraints if specified
    if with_symmetry:
        # Symmetry-breaking constraint: Ensure that boxes are filled in lexicographical order
        for j in range(number_of_objects - 1):
            model.addCons(y[j] >= y[j + 1], f"Lexicographic_{j}")

    # Track solving time and branch-and-bound nodes
    start_time = time.time()
    
    # Optimize the model
    model.optimize()
    
    solving_time = time.time() - start_time
    nodes = model.getNNodes()

    # Output the solution and metrics
    if model.getStatus() == "optimal":
        print(f"Optimal solution found for instance '{instance_name}' with {int(model.getObjVal())} boxes used.")
        print(f"Solving time: {solving_time:.4f} seconds")
        print(f"Branch-and-Bound nodes: {nodes}")
    else:
        print("No optimal solution found.")
        print(f"Solving time: {solving_time:.4f} seconds")
        print(f"Branch-and-Bound nodes: {nodes}")
    
    return solving_time, nodes

def main():
    # Parse the command-line argument for the file name
    parser = argparse.ArgumentParser(description="Solve bin-packing problems with and without symmetry breaking")
    parser.add_argument("filename", type=str, help="The bin-packing instance file (e.g., u20_00.bpa)")
    args = parser.parse_args()
    
    # Run the bin-packing problem without symmetry-breaking constraints
    print("Running without symmetry-breaking constraints...\n")
    time_without_symmetry, nodes_without_symmetry = solve_bin_packing(args.filename, with_symmetry=False)

    # Run the bin-packing problem with symmetry-breaking constraints
    print("\nRunning with symmetry-breaking constraints...\n")
    time_with_symmetry, nodes_with_symmetry = solve_bin_packing(args.filename, with_symmetry=True)

    # Compare the results
    print("\nComparison of Results:")
    print(f"Solving time without symmetry-breaking constraints: {time_without_symmetry:.4f} seconds")
    print(f"Branch-and-Bound nodes without symmetry-breaking constraints: {nodes_without_symmetry}")
    
    print(f"\nSolving time with symmetry-breaking constraints: {time_with_symmetry:.4f} seconds")
    print(f"Branch-and-Bound nodes with symmetry-breaking constraints: {nodes_with_symmetry}")

if __name__ == "__main__":
    main()


"""
Expected Results on Some Bin-Packing Instances :

*Note: The following results were obtained on an Ubuntu operating system running inside a virtual machine. 
The performance may vary depending on the underlying hardware resources allocated to the virtual machine.*



|                   | nb Boxes  |     Time without Symmetry    |     Time with Symmetry      |
| ----------------- | --------- | ---------------------------- | ----------------------------|
| u20_00            | 9         | 0.21                         | 0.14                        |
| u40_00            | 17        | 86                           | 7                           |
| u60_00            |           |                              |                             |
| u120_00           |           |                              |                             |
| inst60_non-unif_9 | 14        | 3                            | 2                           |


"""