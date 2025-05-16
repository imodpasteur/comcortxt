#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.colors as mcolors


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.lines as mlines


def makePlot(data_topics_plotted,hue_name_plotted='topic_name',distance_threshold_grouping=0.1,xlim=None,ylim=None,figsize=(5,5),filename_save='tmp.pdf',palette=None,alpha=0.6,fontsize=8):

    if palette is None:
        palette=getTab30()
    
    
    # topics list extraction for legend and color ()
    sorted_topics = data_topics_plotted[hue_name_plotted].value_counts().sort_values(ascending=False).index
    uniq_list = list(sorted_topics)
    uniq_topicname_list=[]
    uniq_topicname_superimposed_list=[]
    topic_counts=[]
    topic_names_with_number=dict()
    list_topic_ordered=[]
    list_topic_with_number=[]
    total=0
    for u in uniq_list:
        topic_name_tmp=data_topics_plotted[data_topics_plotted[hue_name_plotted] == u][hue_name_plotted].unique()[0]
        list_topic_ordered.append(topic_name_tmp)
        uniq_topicname_list.append(topic_name_tmp)
        topic_count_tmp=len(data_topics_plotted[data_topics_plotted[hue_name_plotted]==u])
        topic_counts.append(topic_count_tmp)
        list_topic_with_number.append(topic_name_tmp+' (n='+str(topic_count_tmp)+')')
        topic_names_with_number[u]=topic_name_tmp+' (n='+str(topic_count_tmp)+')'
        uniq_topicname_superimposed_list.append(superimposeText(topic_name_tmp))
        total+=topic_count_tmp
    
    # cluster list extraction that will appear in umap (topic names obtained  with bertopic)
    cluster_name=data_topics_plotted['topic_name'].unique()
    cluster_name_superimposed=[]
    x_c_median=[]
    y_c_median=[]
    for clus in cluster_name:
        cluster_name_superimposed.append(superimposeText(clus))
        x_c_median.append(data_topics_plotted[data_topics_plotted['topic_name'] == clus].x.median())
        y_c_median.append(data_topics_plotted[data_topics_plotted['topic_name'] == clus].y.median())
        
    
    # we create hierarchical clustering to merge closed dots (to reduce number of dots)
    clustered_data = []
    for topic, group in data_topics_plotted.groupby(hue_name_plotted):
        
        data_xy = group[['x', 'y']].values

        Z = linkage(data_xy, method='centroid') 

        group['cluster'] = fcluster(Z, distance_threshold_grouping, criterion='distance')

        cluster_df = group.groupby('cluster').agg(
            x=('x', 'mean'),
            y=('y', 'mean'),
            size=('x', 'count')
        ).reset_index()

        cluster_df["topic_num"] = topic_names_with_number[topic]  # Preserve the topic label
        cluster_df["topic"] = topic
        clustered_data.append(cluster_df)

    final_cluster_df = pd.concat(clustered_data)

    
    #sort topics by decreasing occ numbers (for legend only)
    sorted_indices = sorted(range(len(topic_counts)), key=lambda i: topic_counts[i], reverse=True)
    sorted_topic_names_with_number = [list_topic_with_number[i] for i in sorted_indices]
    
    #put 'other' & "autre" topic name at the end if exist (for legend only)
    other_topics = [topic for topic in sorted_topic_names_with_number if 'autre' in topic.lower() or 'other' in topic.lower()]
    non_other_topics = [topic for topic in sorted_topic_names_with_number if 'autre' not in topic.lower() and 'other' not in topic.lower()]
    sorted_topic_names_with_number = non_other_topics + other_topics
    
    
    
    #we shuffle before plotting because hierarchical clustering orders by topic
    final_cluster_df_shuffled = final_cluster_df.sample(frac=1).reset_index(drop=True)
    
    print("number of points:",len(final_cluster_df_shuffled),'/',total)
    
    # Plot with topic-based colors
    plt.figure(figsize=figsize)
    scatter=sns.scatterplot(
        data=final_cluster_df_shuffled,
        x='x', y='y',
        size='size', sizes=(2, 200), alpha=alpha,
        hue='topic_num',hue_order=sorted_topic_names_with_number ,palette=palette,  # Assign colors per topic
        legend='auto' 
    )
    
    
    
    
    for i,clus in enumerate(cluster_name_superimposed):   
        #add label
        plt.plot()
        plt.annotate(clus, 
                     (x_c_median[i],y_c_median[i]),
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=fontsize,
                     #weight='bold',
                     #color=tab20(i),
                     color='black') 
    
    

   
    
    
    ax = plt.gca() 
    handles, labels = ax.get_legend_handles_labels()
    
    for handle, label in zip(handles, labels):
        if not label.isdigit():
            print(label,alpha)
            handle.set_alpha(alpha)  # Apply transparency
        
    filtered_handles = []
    filtered_labels = []
    for h, l in zip(handles, labels):
        if l.lower() not in ['size', 'topic_num']:
            filtered_handles.append(h)
            filtered_labels.append(l)

    plt.legend(filtered_handles, filtered_labels, loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1, title="", fontsize=fontsize)  
    
    plt.ylabel('UMAP-2', fontsize=12, fontweight='bold')


    plt.xlabel('UMAP-1', fontsize=12, fontweight='bold') 


    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(filename_save, dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
def superimposeText(oldText):
    new_text=''
    tmp=oldText.replace(' ','')
    tmp=tmp.split(',')
    nb_char_max=0
    for i,tt in enumerate(tmp):
        if len(tt)>nb_char_max:
            nb_char_max=len(tt)
    for i,tt in enumerate(tmp):
        spaces_added=(nb_char_max-len(tt))//2
        new_str=(' '*spaces_added)+tt+(' '*spaces_added)
        new_text+=new_str
        if i<len(tmp)-1:
            new_text+='\n'
    return(new_text)

    
# Function to generate a random fully saturated color
def get_palette(number,color_number):
    pal=[]
    R=number//color_number
    print(R)
    for k in range(number):
        # Random hue between 0 and 1
        hue = (k//R)/color_number
        
        r=(k%color_number)/R
        r=(k%R)/R
        # Set saturation to 1 (maximum saturation)
        saturation = 1-r
        # Set lightness/value to 0.5 (a good balance between bright and dark)
        lightness = .99
        # Convert HSV to RGB
        pal.append(mcolors.hsv_to_rgb((hue, saturation, lightness)))
    return pal

# Get the tab10 colormap
from matplotlib.colors import ListedColormap, rgb_to_hsv, hsv_to_rgb

def getTab30():
    tab10 = plt.cm.tab10.colors

    # Extend tab10 to create tab30 by lightening the colors
    tab30 = []

    # Create new colors by blending or lightening the tab10 colors
    for color in tab10:
        tab30.append(color) 
    for color in tab10:
        hsv = rgb_to_hsv(np.array([color]))
        hsv[0][1] *= 0.6
        hsv[0][2] += (1 - hsv[0][2]) * (1 - 0.6) 
        hsv[0][1] = np.clip(hsv[0][1], 0, 1)
        hsv[0][2] = np.clip(hsv[0][2], 0, 1)
        lighter_color=hsv_to_rgb(hsv[0])
        tab30.append(lighter_color)
    for color in tab10:
        hsv = rgb_to_hsv(np.array([color]))
        hsv[0][1] *= 0.36
        hsv[0][2] += (1 - hsv[0][2]) * (1 - 0.36) 
        hsv[0][1] = np.clip(hsv[0][1], 0, 1)
        hsv[0][2] = np.clip(hsv[0][2], 0, 1)
        lighter_color2=hsv_to_rgb(hsv[0])
        tab30.append(lighter_color2)



        #lighter_color = np.array(color) * 1.3 
        #tab30.append(lighter_color)
    return tab30



# In[ ]:




