
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import  IncrementalPCA
from sklearn.preprocessing import MinMaxScaler
import re
import nltk.corpus
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

####################################################################################################
######### data set includes som NaNs, we need to convert them to meaningful fill them             ##
####################################################################################################
def filler(df):
    convert = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    converted=df[convert].astype(np.float64)
    return converted
#################### returns only string columns ##################################################
def only_Strings(df):
    string_coulmn_names=['language','review']
                         #'app_name' is dropped
    df = df.loc[:, string_coulmn_names]
    return df

def removeWords(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]
def stm(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]
###################################################################################################
#### to do clustering using K means, we need to know before hand howmany clusters we have.  #######
### however, from dataset we cant figure it out. so we use elbow method to know number of k #######
###################################################################################################
def find_best_cluster_number(df):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    d=10 # if by running we didnt find any albow we might probably have more cluster so we might need
            #to incrase d
    K = range(1, d) 
    for k in K:
        print(str(k)+"/"+str(max(K))) # to check if program is running
    
        kmeanModel = KMeans(n_clusters=k).fit(df)
        kmeanModel.fit(df)

        distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / df.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(df, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / df.shape[0]
        mapping2[k] = kmeanModel.inertia_

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion ussing feature eng.')
    plt.show()
    
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia using feature eng.')
    plt.show()
###############################################################################################
i = 0 
def row_modifier(row):
    global i 
    print(str(" {:.5f}".format(i/(len(str_df)-1) *100)) + "%"+"  of rows processed, please wait",end="\r")
    global include_title
    i += 1
    language = row['language']
    reuslt = None
    result = (str(row['review'])).replace(u'\ufffd', '8')   # Replaces the ASCII '�' symbol with '8'
    result = result.replace(',', '')  
    result = result.rstrip('\n')   
    result = result.casefold()               
    # spam, number and punctuation remover
    result = re.sub(r"\s+(.)\1+\b", "",result) 
    result = ''.join([i for i in result if not i.isdigit()])
    result = re.sub(r'[^\w\s]','',result)
    listOfTokens = word_tokenize(result)
    try: 
        stopwords = nltk.corpus.stopwords.words(language)
        stopwords.append("game")
        param_stemmer = SnowballStemmer(language)
        listOfTokens = stm(listOfTokens, param_stemmer)
        listOfTokens = removeWords(listOfTokens, stopwords)
    finally:
        listOfTokens = list(filter(lambda item: len(item) < 25,listOfTokens))   # other spam filter
        listOfTokens = removeWords(listOfTokens, ["game","gam","igr","juego","you","jogao"]) # game in languages in stream reviews
        
        result   = " ".join(listOfTokens)
        result = unidecode(result)
        result = re.sub(r'[^\x00-\x7F]+','', result)    # remove non ascii

        return result
################################################################################################
#################################### BODY ######################################################
################################################################################################
print("(0/10 ) Strating")

df = pd.read_csv("steam_reviews001.csv")
print("(1/10 ) csv reading Done")

numeric_columns = ['app_id','review_id','timestamp_created','timestamp_updated',
                   'votes_helpful','votes_funny','weighted_vote_score','comment_count',
                   'author.num_games_owned','author.steamid','author.num_reviews',
                   'author.playtime_forever','author.playtime_last_two_weeks','author.playtime_at_review']
                   #these columns removed: 'steam_purchase','received_for_free','written_during_early_access','recommended',
df1 =  df.loc[:, numeric_columns]
#df1=df1*1              # maps true and false to zero and one

#scaler = MinMaxScaler()
#df1[list(df1.columns)] = scaler.fit_transform(df1[list(df1.columns)]) # Many machine learning algorithms work better when features are on a relatively similar scale and close to normally distributed
print("(2/10 ) Numerical columns filtered")
str_df=only_Strings(df)
print("(3/10 ) string columns filtered")
str_df['tokens'] = str_df.apply(lambda row: row_modifier(row),axis=1)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(str_df['tokens'])
tf_idf = pd.DataFrame(data = X.toarray(), columns=vectorizer.get_feature_names())
print("(4/10 ) string columns tokenized and vectorized")
df = pd.concat([df1, tf_idf], axis=1)
good_dataframe = filler(df)

#minmaxscaler = MinMaxScaler()
#good_dataframe = minmaxscaler.fit_transform(good_dataframe)
good_dataframe=(good_dataframe - good_dataframe.mean()) / good_dataframe.std() # mean normalization
saved_df=good_dataframe.copy()

pca__ = IncrementalPCA(n_components = 1024, batch_size=4096)
reduced = pca__.fit_transform(good_dataframe)
columns__ = ['pca_comp_%i' % i for i in range(reduced.shape[1])]

good_dataframe = pd.DataFrame(data = reduced, columns = columns__)
print("(5/10 ) coulmns concatenated ")
print("(6/10 ) finding best #clusteres please wait ")
find_best_cluster_number(good_dataframe)
print("check optimal number of k on figure")
print("(7/10) doing Kmeans++ ")
kmeans = KMeans(n_clusters=4).fit(good_dataframe) # based on figure elbow is on 3 so k=3
labels = pd.DataFrame(kmeans.labels_)
print("(8/10 ) Doing PCA. please wait")

print("(9/10 ) normalization done")
#########################################################################################################
###### We need to visulize the data but since the features are too much ( too many dimentions)   ########
###### , we need to use PCA for dimentionality reduction and visulize data. I tried pca with ommiting ###
###### probabily notuseful columns, like row index and so on , but i figured out the out put is the #####
###### same , thanks to PCA.                                                                        #####
#########################################################################################################
pca = PCA(2)
pca.fit(saved_df)
reduced = pca.transform(saved_df)
reduced_df = pd.DataFrame(data = reduced, columns = ['PC1', 'PC2'])
labeled__df = pd.concat((reduced_df,labels),axis=1)
labeled__df = labeled__df.rename({0:'labels'},axis=1)
labeled__df['color'] = labeled__df.apply (lambda row:  'r' if row['labels'] == 1 else ('g' if row['labels'] == 0 else 'b') , axis=1)
print("(10/10 ) PCA done")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(labeled__df['PC1'], labeled__df['PC2'], c = labeled__df['color'])
plt.title('clusters using feature eng.')
plt.show()
