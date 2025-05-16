
import pandas as pd
import numpy as np
import pickle
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


##paths:

model_file='path_to_sentence-camembert-base folder' #https://huggingface.co/dangvantuan/sentence-camembert-base

data_path='path_to_csv_dataframe.csv'
col_name_txt='txt' #colomn name in data_path containing the text to analysis

pathStopWords='./stopwords-fr.txt' #list of stop words https://github.com/stopwords-iso

path_save='./bertopic_output.pkl'








##parameters:

n_neighbors=3
n_components=5
min_dist=0 # to force tight clusters
min_cluster_size=5
min_samples=10







##script:


def prop_upper(txt):
    try:
        is_upper = [word.isupper() for word in txt]
        prop_upper = sum(is_upper)/len(is_upper)
    except:
        prop_upper = 0
    return prop_upper

def lower_all_caps(df):

	df['prop_upper'] = df.txt_raw.apply(lambda x: prop_upper(x))

	index_upper = df[df.prop_upper > .5].index
	#print('is upper >.5: {:.2f}'.format(len(index_upper)), flush=True)

	df['txt'] = df.apply(lambda row: row.txt_raw.lower() if row.prop_upper >.5 else row.txt_raw,
	                     axis=1)

	df = df.reset_index(drop=True)

	return df

def topic_representation(topic_model, docs, topics, probs, df):

    # Generate nicer looking labels and set them in our model
    topic_labels = topic_model.generate_topic_labels(nr_words=2,
                                                     topic_prefix=False,
                                                     #word_length=15,
                                                     separator=", ")
    topic_model.set_topic_labels(topic_labels)

    # Approximate distribution
    (topic_distr, topic_token_distr) = topic_model.approximate_distribution(docs,
                                                                            calculate_tokens=True)


    topic_info = topic_model.get_topic_info()
    topic_info['Freq (%)'] = 100* topic_info.Count / (len(docs))

    topic_dic = topic_info[['Topic', 'CustomName']].set_index('Topic').to_dict()['CustomName']
    print(topic_dic)

    topic_info['NextWords'] = pd.DataFrame(topic_model.generate_topic_labels(nr_words=5,
                                                 topic_prefix=False,
                                                 #word_length=15,
                                                 separator=", ")).apply(lambda x: ', '.join(x[0].split(', ')[2:5]), axis=1)

    topic_info = topic_info[['Topic','CustomName', 'NextWords', 'Count','Freq (%)']]
    topic_info = topic_info.rename(columns={'CustomName':'MainWords'})

    # Create df with all info
    data_topics = pd.DataFrame([topics, probs, docs, topic_distr, topic_token_distr],
                               index=['topics','probs','txt','topic_distr', 'topic_token_distr']).T

    data_topics['topic_name'] = data_topics.topics.replace(topic_dic)







    return data_topics, topic_info




df = pd.read_csv(data_path, index_col = False)

df = df[df[col_name_txt].str.len() > 0]

print('number of sample',len(df))

# handle all caps texts
df = lower_all_caps(df)




docs = list(df[col_name_txt])




with open(pathStopWords, "r") as file:
    fr_stopwords = file.read().split()



sentence_model = SentenceTransformer(model_file)

vectorizer_model= CountVectorizer(ngram_range=(1, 1), stop_words=fr_stopwords)
umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric='cosine', random_state=2)

hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean', cluster_selection_method='eom')



topic_model = BERTopic(
    embedding_model=sentence_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model
    )


embs = sentence_model.encode(docs)




topics, probs = topic_model.fit_transform(docs, embs)


data_topics, topic_info = topic_representation(topic_model,docs,topics,probs,df)


umap_embeddings = UMAP(n_neighbors=n_neighbors,
                       n_components=2,
                       min_dist=0.0,
                       metric='cosine',
                       random_state=2).fit_transform(embs)



df_proj = pd.DataFrame(umap_embeddings, columns=["x", "y"])
data_topics = pd.concat([data_topics, df_proj], axis=1)



#data_topics.to_pickle(path_save)

data_topics.to_csv(path_or_buf=path_save)

print('embedding finished')
