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
  st.text(result_data)
  n_tracker = [i+1 for i in range(new_pig_count)]
  st.text(n_tracker)
  result_data=pd.DataFrame(result_data, columns=['Frame','Tracker_id','Class','Coordinates'])
  n_tracker_namelist=[]
  for id in n_tracker:
    variables['data{}'.format(id)]= pd.DataFrame(result_data[result_data['Tracker_id']==id])
    n_tracker_namelist.append('data'+str(id))
  return n_tracker_namelist


def plot_behavior_id(n_tracker_namelist):
  

  behaviors =['Standing','Sitting','lying_lateral ','lying_sternal','Drinking','Eating']

  for name in n_tracker_namelist:
    id = name[4:]
    temp= pd.DataFrame(eval(name))
    Standing_count = len(temp[temp['Class']=='standing'])
    Sitting_count = len(temp[temp['Class']=='sitting'])
    lying_lateral = len(temp[temp['Class']=='lying_lateral'])
    lying_sternal = len(temp[temp['Class']=='lying_sternal'])
    Drinking = len(temp[temp['Class']=='drinking'])
    Eating = len(temp[temp['Class']=='eating'])
    
    behavior_count = [Standing_count,Sitting_count,lying_lateral,lying_sternal,Drinking,Eating]
    
    data = [behaviors,behavior_count]
    st.altair_chart(data)
