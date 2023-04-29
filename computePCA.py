import numpy as np
from sklearn.decomposition import PCA
import joblib

# file path to the output file from extractLandmarks.py
inputFile = "landmarksLFW.txt"
savedPCAPath = "pca.pkl"

# Create an empty dictionary to store the arrays for each name
nameArrays = {}

print("Reading input file...")
# Open the text file and read its contents
with open(inputFile, 'r') as f:
    for line in f:
        # Split the line into name and values
        name, markNum, x, y,z = line.strip().split(",")

        # Convert the values to floats
        val1 = float(x)
        val2 = float(y)
        val3 = float(z)

        # If the name is not in the dictionary, create a new empty array
        if name not in nameArrays:
            nameArrays[name] = []

        # Append the values to the array for the current name
        nameArrays[name].append(val1)
        nameArrays[name].append(val2)
        nameArrays[name].append(val3)

# Close the file
f.close()

print("Combining arrays...")
# Combine the arrays for each name into a final 2D array output
outputArray = []
for name in nameArrays:
    outputArray.append(nameArrays[name])

print("Converting to numpy array...")
# Convert the output array to a numpy array
outputArray = np.array(outputArray)

print("Fitting PCA...")
# Initialize the PCA object
pca = PCA(n_components=100)

# Fit the PCA object to the data
pca.fit(outputArray)

print("Saving PCA...")
joblib.dump(pca, savedPCAPath)
print("Saved PCA to " + savedPCAPath + ".")


