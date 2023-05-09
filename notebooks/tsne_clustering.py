import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from functools import reduce

class TSNEclustering():
    def __init__(self, all_zs, all_labels, sample_size = None):
        self.all_zs = np.vstack(all_zs)
        self.all_labels = all_labels
   
        self.zs_names = [ 'z_'+str(i) for i in range(all_zs.shape[1])]
        self.lat_df = pd.DataFrame(all_zs,columns=self.zs_names)
        self.lat_df['label'] = all_labels

        if sample_size != None :
            df_sort = [self.lat_df.loc[self.lat_df["label"]==i].reset_index() for i in range(4)]
            df_sort = [df.truncate(0,sample_size) for df in df_sort]
            self.lat_df = df_sort[0]
            for i in range(1,4):
                self.lat_df = pd.concat([self.lat_df,df_sort[i]])
    
    def perform_tsne(self):
        print("starting_tsne")
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(self.lat_df[self.zs_names])

        self.lat_df['tsne-2d-one'] = tsne_results[:,0]
        self.lat_df['tsne-2d-two'] = tsne_results[:,1]

    def plot_result(self):
        sns_plot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            data=self.lat_df,
            legend="full",
            alpha=0.3,
            hue="label",
            palette=sns.color_palette("hls", 4),
        )
        sns_plot.figure.savefig('tsne.png')

    def classify(self):
        # Split the data into training and test sets
        self.lat_df["label"] = self.lat_df["label"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(self.lat_df[["tsne-2d-one","tsne-2d-two"]], self.lat_df["label"], test_size=0.2, random_state=42)

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
        return accuracy