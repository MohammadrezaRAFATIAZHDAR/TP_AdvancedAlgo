import sys
import time
from pyscipopt import Model, quicksum

def parse_instance(filename):
    with open(filename, 'r') as file:
        instance_name = file.readline().strip()
        capacity, number_of_objects, _ = map(int, file.readline().strip().split())
        sizes = [int(file.readline().strip()) for _ in range(number_of_objects)]
    return instance_name, capacity, number_of_objects, sizes

def solve_bin_packing(filename, with_symmetry=True):
    instance_name, capacity, number_of_objects, sizes = parse_instance(filename)
    
    # Initialize model
    model = Model(f"BinPacking_{instance_name}_{'Symmetry' if with_symmetry else 'NoSymmetry'}")
    
    # Define binary variables x[i, j] and y[j]
    x = {}
    for i in range(number_of_objects):
        for j in range(number_of_objects):
            x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")
    
    y = {}
    for j in range(number_of_objects):
        y[j] = model.addVar(vtype="B", name=f"y_{j}")
    
    # Set objective: minimize the number of boxes used
    model.setObjective(quicksum(y[j] for j in range(number_of_objects)), "minimize")
    
    # Constraint: each object is placed in exactly one box
    for i in range(number_of_objects):
        model.addCons(quicksum(x[i, j] for j in range(number_of_objects)) == 1, f"Object_{i}_placement")
    
    # Constraint: the total size in each box does not exceed its capacity if the box is used
    for j in range(number_of_objects):
        model.addCons(quicksum(sizes[i] * x[i, j] for i in range(number_of_objects)) <= capacity * y[j], f"Box_{j}_capacity")
    
    # Linking constraints: if an object is in a box, the box must be used
    for i in range(number_of_objects):
        for j in range(number_of_objects):
            model.addCons(x[i, j] <= y[j], f"Linking_{i}_{j}")
    
    if with_symmetry:
        # Symmetry-breaking constraint 1: Lexicographic ordering of boxes
        for j in range(number_of_objects - 1):
            model.addCons(y[j] >= y[j + 1], f"Lexicographic_{j}")

        # Symmetry-breaking constraint 2: Load order constraint
        for j in range(number_of_objects - 1):
            model.addCons(
                quicksum(sizes[i] * x[i, j] for i in range(number_of_objects)) >= 
                quicksum(sizes[i] * x[i, j + 1] for i in range(number_of_objects)), 
                f"LoadOrder_{j}"
            )
    
    # Start the timer
    start_time = time.time()
    # Optimize the model
    model.optimize()
    end_time = time.time()
    
    # Calculate and print the time taken
    time_taken = end_time - start_time
    print(f"{'With' if with_symmetry else 'Without'} symmetry: Optimal solution found in {time_taken:.2f} seconds.")
    
    # Check if an optimal solution is found
    if model.getStatus() == "optimal":
        print(f"Optimal solution for instance '{instance_name}' with {int(model.getObjVal())} boxes used.")
        
        # Print the items in each box
        for j in range(number_of_objects):
            if model.getVal(y[j]) > 0.5:
                items_in_box = [i + 1 for i in range(number_of_objects) if model.getVal(x[i, j]) > 0.5]
                print(f"Box {j + 1} contains items: {items_in_box}")
    else:
        print("No optimal solution found.")

# Main entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 bin_packing_symmetry.py <filename>")
    else:
        filename = sys.argv[1]
        
        print("Solving without symmetry-breaking constraints:")
        solve_bin_packing(filename, with_symmetry=False)
        
        print("\nSolving with symmetry-breaking constraints:")
        solve_bin_packing(filename, with_symmetry=True)
