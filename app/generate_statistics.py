import streamlit as st
import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import RendererAgg
##import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns


def seggrate_id_data(result_data, new_pig_count):
  variables = globals()
  ##st.text(result_data)
  n_tracker = [i+1 for i in range(new_pig_count)]
  ##st.text(n_tracker)
  result_data=pd.DataFrame(result_data, columns=['Frame','Tracker_id','Class','Coordinates'])
  n_tracker_namelist=[]
  for id in n_tracker:
    variables['data{}'.format(id)]= pd.DataFrame(result_data[result_data['Tracker_id']==id])
    n_tracker_namelist.append('data'+str(id))
  return n_tracker_namelist






def plot_behavior_id(result_data, new_pig_count):


  result_data=pd.DataFrame(result_data, columns=['Frame','Tracker_id','Class','Coordinates'])

  for n in range(len(new_pig_count)):
    temp = result_data[result_data['Tracker_id']==n]
    temp=result_data
    class_array = flatten_list(temp['Class'])

    behaviors_labels =['Standing','Sitting','lying_lateral ','lying_sternal','Drinking','Eating']


    Standing_count = class_array.count('Standing')
    Sitting_count = class_array.count('Sitting')
    lying_lateral = class_array.count('lying_lateral')
    lying_sternal = class_array.count('lying_sternal')
    Drinking = class_array.count('Drinking')
    Eating = class_array.count('Eating')

    temp_count = [Standing_count,Sitting_count,lying_lateral,lying_sternal,Drinking,Eating]
    st.text(temp_count)
    chart_data = pd.DataFrame()
    chart_data['behaviors_labels']=behaviors_labels
    chart_data['behavior_count']=temp_count
    st.text(chart_data)

  
  st.altair_chart(alt.Chart(chart_data).mark_bar().encode(
  	x=alt.X('behaviors_labels', bin=True), y='behavior_count',column='Tracker_id'))
    
    
    
def plot_behavior_id(result_data, new_pig_count):


  result_data=pd.DataFrame(result_data, columns=['Frame','Tracker_id','Class','Coordinates'])
  st.text(result_data)
  result_data['Class'] = flatten_list(result_data['Class'])
  st.text(result_data)
  
  behaviors_labels =['Standing','Sitting','lying_lateral ','lying_sternal','Drinking','Eating']

  chart_data = pd.DataFrame()

  
  st.altair_chart(alt.Chart(result_data).mark_bar().encode(
  	x=alt.X('Class'), y='count(Class)',column='Tracker_id', color=alt.Color('Tracker_id',scale=alt.Scale(scheme='dark2'))))
