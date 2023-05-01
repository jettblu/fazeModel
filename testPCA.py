# load saved pca
import joblib
import matplotlib.pyplot as plt
import numpy as np

# load pca from file
savedPCAPath = "pca.pkl"
pca = joblib.load(savedPCAPath)

# load lfw data and transform 
# Sample data
X = np.random.rand(1, 1404)  # 10 samples with 1404 features each

# Apply PCA transformation to sample data
X_transformed = pca.transform(X)

print(X_transformed)
print(X_transformed.shape)


