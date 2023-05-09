from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier


class SAE_clustering():
 
    def __init__(self):



all_zs = np.reshape(all_zs,(-1,lat_dim))
all_labels = np.reshape(all_labels,(10000,1))

zs_names = [ 'z_'+str(i) for i in range(all_zs.shape[1])]
lat_df = pd.DataFrame(all_zs,columns=zs_names)
lat_df['label'] = all_labels

tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(lat_df[zs_names])

lat_df['tsne-2d-one'] = tsne_results[:,0]
lat_df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns_plot = plt.scatter(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    palette=sns.color_palette("hls", 10),
    data=lat_df,
    legend="full",
    alpha=0.3
)
sns_plot.figure.savefig('tsne.png')

# Split the data into training and test sets
lat_df["label"] = lat_df["label"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(lat_df[["tsne-2d-one","tsne-2d-two"]], lat_df["label"], test_size=0.2, random_state=42)

# Convert the dataframes to NumPy arrays
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values



# Define the model
model = AdaBoostClassifier()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model's performance
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")