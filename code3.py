# Define two sets E and N
E = {0, 2, 4, 6, 8}
N = {1, 2, 3, 4, 5}

# Perform and display set operations
union_result = E.union(N)
intersection_result = E.intersection(N)
difference_result = E.difference(N)
symmetric_difference_result = E.symmetric_difference(N)

# Print the results
print("Union of E and N is", union_result)
print("Intersection of E and N is", intersection_result)
print("Difference of E and N is", difference_result)
print("Symmetric difference of E and N is", symmetric_difference_result)
