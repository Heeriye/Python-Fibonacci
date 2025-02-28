# Importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Ensuring inline plotting for Jupyter Notebook (only for Jupyter)
%matplotlib inline  

# Checking if the file exists before reading
file_path = 'mnist.csv'

if os.path.exists(file_path):
    # Using pandas to read the database stored in the same folder
    data = pd.read_csv(file_path)
    print("Data loaded successfully!\n")
    
    # Viewing column heads
    print("First 5 rows of the dataset:")
    print(data.head())

    # Extracting data from the dataset (Row 3, All Pixel Values)
    a = data.iloc[3, 1:].values  # Extracting only pixel values (excluding the label)
    print("\nExtracted raw pixel data from row 3:", a[:10], "...")  # Displaying first 10 values

    # Reshaping the extracted data into a 28x28 image format
    a = a.reshape(28, 28).astype('uint8')

    # Displaying the extracted digit as an image
    plt.imshow(a, cmap='gray')
    plt.title("Extracted Handwritten Digit (Row 3)")
    plt.axis("off")
    plt.show()

    # **Preparing the data for model training**
    # Separating labels and pixel values
    df_x = data.iloc[:, 1:]  # Pixel values (features)
    df_y = data.iloc[:, 0]   # Labels (digits)

    # Splitting the dataset into training (80%) and testing (20%) sets
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

    print("\nData preparation completed. Training set size:", x_train.shape, "Testing set size:", x_test.shape)

    # **Train a Random Forest Classifier**
    print("\nTraining Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(x_train, y_train)

    print("Model training complete!")

    # **Make predictions on the test data**
    pred = rf.predict(x_test)

    # Display first 10 predictions
    print("\nPredictions on test data:", pred[:10])

    # **Calculating Accuracy Manually**
    s = y_test.values  # Actual labels
    count = sum(1 for i in range(len(pred)) if pred[i] == s[i])  # Counting correct predictions
    total = len(pred)
    manual_accuracy = count / total

    print(f"\nManual Accuracy Calculation: {count}/{total} = {manual_accuracy:.4f}")

    # **Using sklearn's accuracy_score**
    sklearn_accuracy = accuracy_score(y_test, pred)
    print(f"Sklearn Accuracy Score: {sklearn_accuracy:.4f}")

else:
    print(f"Error: File '{file_path}' not found.")
