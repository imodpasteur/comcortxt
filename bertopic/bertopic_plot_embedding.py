
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_umap import *

##paths:

embedding_path='./bertopic_output.csv'







data_topics = pd.read_csv(embedding_path)




miny=np.min(data_topics.y)
maxy=np.max(data_topics.y)
minx=np.min(data_topics.x)
maxx=np.max(data_topics.x)


ylim=[miny,maxy]
xlim=[minx,maxx]


distance_threshold_grouping=0.2

filename_save='plot.pdf'
figsize=(5,5)
hue_name_plotted='topic_name'

makePlot(data_topics,hue_name_plotted=hue_name_plotted,xlim=xlim,ylim=ylim,distance_threshold_grouping=0.2,figsize=figsize,filename_save=filename_save)





















print('plot finished')
