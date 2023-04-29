# load saved pca
import joblib
import matplotlib.pyplot as plt
import numpy as np

savedPCAPath = "pca.pkl"
pca = joblib.load(savedPCAPath)

# create chart of explained variance ratio
# label accordingly
# create figure that can be saved
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
# add main title with vspace to make room for the subtitle
plt.title('Explained Variance Ratio vs. Principal Component')
# save the plot
plt.savefig('pcaRatio.png')
plt.show()


# create chart of cumulative explained variance ratio
# label accordingly
plt.plot(np.cumsum(pca.explained_variance_ratio_))
# add horizontal line at 95% explained variance
plt.axhline(y=0.95, color='g', linestyle='-')
# add label to horizontal line
plt.text(0, 0.96, '95% Explained Variance', color = 'green', fontsize=10)
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance Ratio')
# add main title
plt.title('Cumulative Explained Variance Ratio vs. Principal Component')
# save the plot
plt.savefig('pcaCumulative.png')
plt.show()


# number of components to keep based on the cumulative explained variance ratio
# select the number of components that explain 95% of the variance
numComponents = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print("Number of components to keep: " + str(numComponents))
