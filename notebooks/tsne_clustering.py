import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from functools import reduce
import umap.umap_ as umap

class TSNEclustering():
    def __init__(self, all_zs = None, all_labels = None, lat_df = 0, lat_dim = 32, sample_size = None):
        self.zs_names = [ 'z_'+str(i) for i in range(lat_dim)]
        if type(lat_df) == int : 
            self.all_zs = np.vstack(all_zs)
            self.all_labels = all_labels
            self.lat_df = pd.DataFrame(all_zs,columns=self.zs_names)
            self.lat_df['labels'] = all_labels
        else : 
            self.lat_df = lat_df
        
        self.dict_labels = {0.0 : "electron", 1.0 : "gamma", 2.0:"muon", 3.0: "pion"}
        self.str_labels = ["electron", "gamma", "muon", "pion"]
        self.lat_df["cut"] = (self.lat_df["nhits"]>25) & (self.lat_df["dwall"] > 50) & (self.lat_df["towall"] > 100)
        self.base_df = self.lat_df
        if sample_size != None :
            self.df_sort = [self.lat_df.loc[self.lat_df["labels"]==i].reset_index() for i in range(4)]
            df_sort_trunc = [df.truncate(0,sample_size) for df in self.df_sort]
            self.lat_df = df_sort_trunc[0]
            for i in range(1,4):
                self.lat_df = pd.concat([self.lat_df,df_sort_trunc[i]])

            df_sort_trunc_bis = [df.truncate(sample_size,2*sample_size+1) for df in self.df_sort]
            self.test_df = df_sort_trunc_bis[0]
            for i in range(1,4):
                self.test_df = pd.concat([self.test_df,df_sort_trunc_bis[i]])
             
    
    def perform_umap(self, n_neighbors=15, min_dist=0.1, metric='euclidean', supervised = True):
        print("Starting Dimensionality reduction")
        #tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
        self.umapT = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
        if supervised :
            umap_results = self.umapT.fit_transform(self.lat_df[self.zs_names], y = self.lat_df["labels"])
        else:
            umap_results = self.umapT.fit_transform(self.lat_df[self.zs_names])
        self.lat_df['umap-2d-one'] = umap_results[:,0]
        self.lat_df['umap-2d-two'] = umap_results[:,1]

    def plot_result_train(self, title = "UMAP results", use_cut = False, cutted_events = False):
        return self.plot_result(self.lat_df, title, use_cut, cutted_events)
    
    def plot_result_test(self,title = "UMAP results", use_cut = False, cutted_events = False):
        return self.plot_result(self.test_df, title, use_cut, cutted_events, test = True)
    
    def plot_result(self, df = None, title = "UMAP results", use_cut = False, cutted_events = False, test= False):
        if test: 
            umap_results = self.umapT.transform(df[self.zs_names])
            df['umap-2d-one'] = umap_results[:,0]
            df['umap-2d-two'] = umap_results[:,1]
        if use_cut:
            plot_df = df.loc[df["cut"]]
        elif cutted_events:
            plot_df = df.loc[df["cut"].replace({True : False, False : True})]
        else : 
            plot_df = df
        ax = sns.scatterplot('umap-2d-one', 'umap-2d-two', data=plot_df,
                hue='labels',
                linewidth=0,
                alpha=0.6,
                palette = "tab10",
                s=7)
        plt.legend(labels = self.str_labels)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   
    def get_df(self):
        return self.lat_df

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