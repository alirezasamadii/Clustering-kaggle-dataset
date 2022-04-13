
import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
####################################################################################################
######### data set includes som NaNs, or true false values, we need to convert them to meaningful ##
######### numeric data by this function                                                           ##
####################################################################################################
def filler(df):
    numeric_columns = ['app_id','review_id','timestamp_created','timestamp_updated','recommended',
                       'votes_helpful','votes_funny','weighted_vote_score','comment_count',
                       'steam_purchase','received_for_free','written_during_early_access',
                       'author.num_games_owned','author.steamid','author.num_reviews',
                       'author.playtime_forever','author.playtime_last_two_weeks','author.playtime_at_review']
                
    df =  df.loc[:, numeric_columns]
    df.dropna(inplace=True)
    df=df*1 # maps true and false to zero and one
    convert = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    converted=df[convert].astype(np.float64)
    return converted
###################################################################################################
#### to do clustering using K means, we need to know before hand howmany clusters we have.  #######
### however, from dataset we cant figure it out. so we use elbow method to know number of k #######
###################################################################################################
def find_best_cluster_number(df):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    d=2 # if by running we didnt find any albow we might probably have more cluster so we might need
            #to incrase d
    K = range(1, 10) 
    for k in K:
        print(str(k)+" / "+str(max(K))) # to check if program is running
        kmeanModel = KMeans(n_clusters=k).fit(df)
        kmeanModel.fit(df)
        distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / df.shape[0])
        inertias.append(kmeanModel.inertia_)
        mapping1[k] = sum(np.min(cdist(df, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / df.shape[0]
        mapping2[k] = kmeanModel.inertia_
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()
    
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()
################################################################################################
#################################### BODY ######################################################
################################################################################################
print("starting")
df = pd.read_csv("steam_reviews0.csv")
print("csv read done")
mydf=df.sample(n = 10000)    
mydf = filler(mydf)
print("filtering done")
find_best_cluster_number(mydf)
print("check optimal number of k on figure")
kmeans = KMeans(n_clusters=2).fit(mydf) # based on figure elbow is on 3 so k=3
labels = pd.DataFrame(kmeans.labels_)
#########################################################################################################
###### We need to visulize the data but since the features are too much ( too many dimentions)   ########
###### , we need to use PCA for dimentionality reduction and visulize data. I tried pca with ommiting ###
###### probabily notuseful columns, like row index and so on , but i figured out the out put is the #####
###### same , thanks to PCA.                                                                        #####
#########################################################################################################
pca = PCA(2)
pca.fit(mydf)
reduced = pca.transform(mydf)
reduced_df = pd.DataFrame(data = reduced, columns = ['PC1', 'PC2'])
labeled__df = pd.concat((reduced_df,labels),axis=1)
labeled__df = labeled__df.rename({0:'labels'},axis=1)
labeled__df['color'] = labeled__df.apply (lambda row:  'r' if row['labels'] == 1 else ('g' if row['labels'] == 0 else 'b') , axis=1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(labeled__df['PC1'], labeled__df['PC2'], c = labeled__df['color'])
plt.show()