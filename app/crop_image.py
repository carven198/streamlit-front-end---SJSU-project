import os
import cv2
import random
import numpy as np
#import tensorflow as tf
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt

def load_txt(file_txt):
  import pandas as pd
  with open(file_txt) as f:
    result = []
    for line in f:
      if line.startswith("FPS"):
        pass
      else:
        result.append(line)

    for i in range(len(result)):
      result[i]=result[i].replace(' ','')
      if result[i].endswith('\n'):
        result[i] = result[i].replace('\n','')
      result_data=[]

    for i in range(len(result)):

      if result[i].startswith('Frame'):
          _, frame_number = result[i].split(':')
      elif result[i].startswith('Tracker'):
        _, trackerid, classdata, coord = result[i].split(':')
        trackerid=int(trackerid.replace(',Class',''))
        classdata=classdata.replace(',BBoxCoords(xmin,ymin,xmax,ymax)','')
        coord=coord.replace(' ','').replace('(','').replace(')','')
        xmin, ymin, xmax, ymax = coord.split(',')
        result_data.append([int(frame_number), trackerid, str(classdata), int(xmin), int(ymin), int(xmax), int(ymax)])
    
    result_data = pd.DataFrame(result_data,columns=['Frame','Tracker_id','Class','xmin','ymin','xmax','ymax'])
    return result_data

def crop_image_coordinates(coordinates_data,im_name):
 
  ##frame_read = cv2.imread(im_name)
  ##for box in list(coordinates_data):
    ##start_coord = coordinates_data[:2]
    w, h = coordinates_data[2:]
    xmin = coordinates_data[0]
    ymin = coordinates_data[1]
    if xmin < 0:
    	xmin=0
        
    if ymin < 0:
    	ymin =0

    
   ##end_coord = start_coord[0] + w, start_coord[1] + h
    xmax = xmin+w
    ymax = ymin+h
    


    crop = im_name[ymin:ymax,xmin:xmax]
    resized_image = cv2.resize(crop, (415,415), interpolation = cv2.INTER_CUBIC)
    return resized_image
 

