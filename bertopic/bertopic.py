import pandas as pd
import numpy as np
import pickle
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# for reproducibility
np.random.seed(11)

path = "/pasteur/appa/homes/gbizelbi/comcor/Gaston/"
model_file = path + 'camembert/sentence-camembert-base/'
output_data_path = path + 'cr/output/'


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

    cols = ['IDIPSOS',
             'txt_situation',
             'txt_len',
             'sexe',
             'age_cat_1',
             'imc',
             'agglo',
             'csp_partic',
             'evenement_suspect_contexte',
             'evenement_suspect_type']

    data_topics[cols] = df.reset_index(drop=True)[cols].reset_index(drop=True)[cols]
    
    return data_topics, topic_info


 def add_best_sentences(topic_info, data_topics, max_len = 20):
    topic_info['sentence_1'] = pd.Series(dtype='str')
    topic_info['sentence_2'] = pd.Series(dtype='str')
    topic_info['sentence_3'] = pd.Series(dtype='str')
    for i in range(0, len(topic_info)):
        best_candidates = data_topics[(data_topics.topics == i-1) & (data_topics.txt_len < max_len)]\
                          .sort_values(by='probs', ascending=False)
        try:
            topic_info['sentence_1'].iloc[i] = best_candidates.txt.iloc[0]
        except:
            best_candidates = data_topics[(data_topics.topics == i-1)].sort_values(by='probs', ascending=False)
            topic_info['sentence_1'].iloc[i] = best_candidates.txt.iloc[0]

        try:
            topic_info['sentence_2'].iloc[i] = best_candidates.txt.iloc[1]
        except:
            best_candidates = data_topics[(data_topics.topics == i-1)].sort_values(by='probs', ascending=False)
            topic_info['sentence_2'].iloc[i] = best_candidates.txt.iloc[1]
        
        try:
            topic_info['sentence_3'].iloc[i] = best_candidates.txt.iloc[2]
        except:
            best_candidates = data_topics[(data_topics.topics == i-1)].sort_values(by='probs', ascending=False)
            topic_info['sentence_3'].iloc[i] = best_candidates.txt.iloc[2]

    return topic_info


 def list_topics(data_topics):
    l_topics = []
    for i in data_topics.topics.value_counts().index:
        l_topics+=[data_topics[data_topics.topics == i].topic_name.iloc[0]]
    return l_topics


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
	print('is upper >.5: {:.2f}'.format(len(index_upper)), flush=True)

	df['txt'] = df.apply(lambda row: row.txt_raw.lower() if row.prop_upper >.5 else row.txt_raw,
	                     axis=1)  

	df = df.reset_index(drop=True)

	return df


# import dataset
df = pd.read_csv(path+'data/data.csv', index_col = False)

df = df[df.txt_len > 0][['IDIPSOS',
                         'txt_situation',
                         'txt_len',
                         'txt_raw',
                         'sexe',
                         'age_cat_1',
                         'imc',
                         'agglo',
                         'csp_partic',
                         'evenement_suspect_contexte',
                         'evenement_suspect_type']]


# handle all caps texts
df = lower_all_caps(df)

# filter data
df_si = df[(df.txt_situation == 'SI')]





sentence_model = SentenceTransformer(model_file)
fr_stopwords = ["y","y'","m", "l", "d", "t", "qu", "s","c","m'",'hein', 'celle-là', 'ceux-ci', 'dring', 'sa', 'ollé', 'en', 'a', "d'", 'plutôt', 'auxquels', 'celles-ci', 'dès', 'tel', 'lui-meme', 'quelle', 'les', 'dont', 'aie', 'quand', 'pour', 'où', 'lès', 'suivant', 'ho', 'memes', 'hem', 'surtout', 'mien', 'tellement', 'qui', 'le', 'quels', 'tant', 'une', 'tien', 'ohé', 'i', 'mêmes', 'ceux', "l'", 'quelque', 'si', 'unes', 'lequel', 'tous', 'chacune', 'son', 'que', 'quel', 'au', 'ai', 'celui-là', 'chaque', 'ouste', 'es', 'hep', 'elles-mêmes', 'lors', 'cette', 'cependant', 'toc', 'tsouin', 'chacun', 'seule', 'siennes', 'hum', 'la', 'certains', "t'", 'trop', 'dans', 'desquels', 'lui', 'hors', 'celles-là', 'lui-même', 'pouah', 'toi-même', 'boum', 'vive', 'rend', 'mes', 'vos', 'nous', "qu'", 'des', 'tiens', 'hé', 'lorsque', 'zut', 'vlan', 'mienne', 'na', 'ma', 'selon', "s'", 'vous-mêmes', 'eh', 'ah', 'ses', 'meme', 'lesquels', 'miens', 'vôtres', 'paf', 'pif', 'quant-à-soi', 'tes', "c'", 'sien', 'ça', 'lesquelles', 'tout', 'telles', 'même', 'ces', 'maint', 'notre', 'quanta', 'elle-même', 'aupres', 'bas', 'votre', 'plusieurs', 'moi', 'par', 'hurrah', 'bah', 'laquelle', 'auxquelles', 'vé', 'peux', 'pure', 'tiennes', "aujourd'hui", 'hormis', 'couic', 'vous', 'ore', 'envers', 'moindres', 'aucune', 'gens', 'ouias', 'cela', 'quelles', 'aux', 'pff', 'etc', 'toutefois', 'leurs', 'ton', 'clic', 'las', 'pfut', "t'", 'toutes', 'cet', 'ta', 'da', 'toute', 'aucun', 'o', 'sapristi', 'quoi', 'desquelles', 'té', 'vôtre', 'euh', 'pres', 'as', 'fi', 'ci', 'allo', 'oh', "s'", 'quiconque', 'floc', 'avec', 'se', 'bat', 'tic', 'jusqu', "qu'", 'unique', 'certes', 'celles', 'dire', 'tienne', 'ha', 'nôtre', 'jusque', 'tac', 'ceux-là', 'sienne', 'uns', 'ouf', 'moi-même', 'et', 'vers', 'miennes', 'autrefois', 'houp', 'été', 'à', "d'", 'nouveau', 'être', 'peu', 'dite', "s'", 'dit', 'tels', 'ou', 'toi', 'entre', 'avoir', 'hop', 'delà', 'nos', 'tres', 'telle', 'voilà', 'dessous', 'soit', 'autres', 'psitt', 'hélas', 'anterieur', 'hou', 'près', 'auquel', 'juste', 'chut', 'un', 'stop', 'eux', 'ès', 'vifs', 'ce', 'quoique', 'du', 'moi-meme', 'mon', 'brrr', 'sous', 'parmi', 'deja','déja','celle', 'siens', 'suffisant', 'â', "l'", 'apres', 'sans', 'soi-même', 'là', 'pur', 'via', 'differentes', 'specifique', 'holà', 'tsoin', 'pan', 'car', 'donc', 'dits', 'merci', 'particulièrement', 'nous-mêmes', 'personne', 'allô', 'soi', 'voici', 'sur', 'vif', 'celle-ci', 'malgré', 'puis', 'sauf', 'autre', 'hui', 'ceci', 'leur', 'celui-ci', 'necessairement', 'sacrebleu', 'hue', 'eux-mêmes', 'outre', 'alors', 'desormais', 'plouf', 'longtemps', 'malgre', 'après', 'de', 'oust', 'neanmoins', 'certain', 'crac', 'depuis', 'olé', 'hi', 'te', 'puisque', "m'", 'me', 'ô', 'celui', 'aussi', 'rares', 'chiche', 'rien', 'pfft', "c'", 'vu', 'clac', 'duquel', 'aavons', 'avez', 'ont', 'eu', 'avais', 'avait', 'avions', 'aviez', 'avaient', 'eus', 'eut', 'eûmes', 'eûtes', 'eurent', 'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', 'aurais', 'aurait', 'aurions', 'auriez', 'auraient', 'aies', 'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent', 'ayant', 'suis', 'est', 'sommes', 'êtes', 'sont', 'étais', 'était', 'étions', 'étiez', 'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 'serais', 'serait', 'serions', 'seriez', 'seraient', 'sois', 'soyons', 'soyez', 'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', 'étant']
fr_stopwords+=['je', 'ne','pas', 'on', 'comme', 'plus', 'il', 'mais', 'chez']
fr_stopwords+=['elle', 'personnes']
vectorizer_model= CountVectorizer(ngram_range=(1, 1), stop_words=fr_stopwords)
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=130, min_samples=20, metric='euclidean', cluster_selection_method='eom')

topic_model = BERTopic(
    embedding_model=sentence_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model
    )

docs = list(df_si.txt)
embs = sentence_model.encode(data, show_progress_bar=True)


topics, probs = topic_model.fit_transform(docs, embs)


data_topics, topic_info = topic_representation(topic_model,
                                               docs,
                                               topics,
                                               probs,
                                               df_si)

topic_info = add_best_sentences(topic_info, data_topics, max_len=20)


umap_embeddings = UMAP(n_neighbors=15,
                       n_components=2,
                       min_dist=0.0,
                       metric='cosine',
                       random_state=42).fit_transform(embs)


df_proj = pd.DataFrame(umap_embeddings, columns=["x", "y"])
df_proj


data_topics = pd.concat([data_topics, df_proj], axis=1)
data_topics


x_c_median = list(data_topics.groupby(['topics']).median().x)
y_c_median = list(data_topics.groupby(['topics']).median().y)


fig, ax = plt.subplots(figsize=(8, 8))

facet = sns.scatterplot(data=data_topics,
                   x='y',
                   y='x',
                   hue='topic_name',
                   hue_order=list_topics(data_topics),
                   palette='tab20',
                   alpha=0.5,
                   #scatter_kws={'alpha':0.3},
                   #fit_reg=False,
                   legend=True,
                   #legend_out=True
                   s=20
                   )

facet.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

for i,topic in enumerate(list_topics(data_topics)):   
    #add label
    plt.annotate(topic, 
                 (y_c_median[i],x_c_median[i]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=9,
                 #weight='bold',
                 #color=tab20(i),
                 color='black') 

plt.ylim([4,18])
plt.xlim([0,10])

plt.savefig('bertopic_representation.pdf', dpi=1200, bbox_inches='tight')