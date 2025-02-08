# Function to print Fibonacci sequence up to n terms
def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        print(a, end=" ")
        a, b = b, a + b

# Input: number of terms
num_terms = int(input("Enter the number of terms: "))
fibonacci(num_terms)
