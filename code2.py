# Function to print positive numbers from a list
def print_positive_numbers(numbers):
    positive_numbers = [num for num in numbers if num > 0]
    print("Output:", positive_numbers)

# Example inputs and outputs
list1 = [12, -7, 5, 64, -14]
list2 = [12, 14, -95, 3]

# Displaying positive numbers for both lists
print("Input: list1 =", list1)
print_positive_numbers(list1)

print("Input: list2 =", list2)
print_positive_numbers(list2)
