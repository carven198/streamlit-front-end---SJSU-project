import streamlit as st
import numpy
import sys
import os
import pandas as pd
import tempfile
drive_path = '/content/drive/MyDrive/Colab Notebooks/streamlit/streamlit - SJSU Master/'
sys.path.append(os.getcwd())
import livestock_detector as ld
import classify_posture as cp
import crop_image as cimage
##import generate_statistics as gs
from altair import Chart, X, Y, Axis, SortField, OpacityValue
import cv2 
import time
import SessionState as SessionState
#import utils.SessionState as SessionState
from random import randint
from streamlit import caching
import streamlit.report_thread as ReportThread 
from streamlit.server.server import Server
import copy
##from components.custom_slider import custom_slider
import altair as alt
import time



# define the weights to be used along with its config file
drive_path = '/content/drive/MyDrive/Colab Notebooks/streamlit/streamlit - SJSU Master/'
config_detection =  drive_path+'darknet/cfg/custom-yolov4-detector.cfg'
wt_file_detection = drive_path+'darknet/backup/custom-yolov4-detector_last.weights'

config_classification = drive_path+'darknet/cfg/custom-yolov4-posture_classify.cfg'
wt_file_classificaton = drive_path+'darknet/backup/custom-yolov4-classification_best.weights'

# define recommend values for model confidence and nms suppression 
def_values ={'conf': 70, 'nms': 50} 
keys = ['conf', 'nms']


@st.cache(
    hash_funcs={
        st.delta_generator.DeltaGenerator: lambda x: None,
        "_regex.Pattern": lambda x: None,
    },
    allow_output_mutation=True,
)


def load_obj_detector(config, wt_file):
    """
    wrapper func to load and cache object detector 
    """
    obj_detector = ld.ObjectDetector(wt_file, config, confidence = def_values['conf']/100,
     nms_threshold=def_values['nms']/100)

    return obj_detector
    
def load_posture_classify(config, wt_file):
    """
    wrapper func to load and cache object detector 
    """
    posture_classify = cp.PostureClassifier(wt_file, config, confidence = def_values['conf']/100,
     nms_threshold=def_values['nms']/100)

    return posture_classify

def parameter_sliders(key, enabled = True, value = None):
    conf = st.sidebar.slider("Model Confidence", min_value=0.0, max_value=1.0, step=0.1)
    nms = st.sidebar.slider('Overlapping Threshold', min_value=0.0, max_value=1.0, step=0.1)
        
    return(conf, nms)
    



def trigger_rerun():
    """
    mechanism in place to force resets and update widget states
    """
    session_infos = Server.get_current()._session_info_by_id.values() 
    for session_info in session_infos:
        this_session = session_info.session
    this_session.request_rerun()




def main():

    st.title("""
    Livestock Behavior Monitoring App
    """)

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Introduction','Behavior Detection', 'Performance Metrics')
    )
    
    if selected_box == 'Introduction':
        intro() 
    if selected_box == 'Behavior Detection':
        behavior_detection()
    if selected_box == 'Performance Metrics':
        stats()
        

def flatten_list(class_list):
    class_array=[]
    for item in class_list:
        ##st.text(item)
        ##st.text(type(item))
        if item == []:
            item = 'No data'
            class_array.append(item)
        else:
            item = str(item[0])
            ##st.text(item)
            class_array.append(item)
            
    ##st.text('end')
    ##st.text(class_array)
    return class_array


def plot_behavior_id(result_data, new_pig_count):




  result_data=pd.DataFrame(result_data, columns=['Frame','Tracker_id','Class','Coordinates'])

  result_data['Class'] = flatten_list(result_data['Class'])

  
  behaviors_labels =['Standing','Sitting','lying_lateral ','lying_sternal','Drinking','Eating']

  chart_data = pd.DataFrame()

  
  bars = alt.Chart(result_data).mark_bar().encode(x=alt.X('Class'), y='count(Class)',column='Tracker_id', color=alt.Color('Tracker_id',scale=alt.Scale(scheme='dark2'))).properties(width=100,height=250)
  

    
  return bars





    
  

def hide_streamlit_widgets():
    """
    hides widgets that are displayed by streamlit when running
    """
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
def ProcessFrames(vf, tracker, obj_detector,stop,posture_classify): 
    """
        main loop for processing video file:
        Params
        vf = VideoCapture Object
        tracker = Tracker Object that was instantiated 
        obj_detector = Object detector (model and some properties) 
    """

    try:
        num_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vf.get(cv2.CAP_PROP_FPS)) 
        print('Total number of frames to be processed:', num_frames,
        '\nFrame rate (frames per second):', fps)
    except:
        print('We cannot determine number of frames and FPS!')


    frame_counter = 0
    _stop = stop.button("stop")
    new_pig_count_txt = st.empty()
    fps_meas_txt = st.empty()
    countframe_txt = st.empty()
    bar = st.progress(frame_counter)
    stframe = st.empty()
    start = time.time()
    statistics = []
    countframe = 0
    new_pig_count=0
    
    
    bar_plot = st.empty()
 
    
    while vf.isOpened():
        # if frame is read correctly ret is True
        ret, frame = vf.read()
        if _stop:
            break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        countframe = countframe + 1
        labels, current_boxes, confidences= obj_detector.ForwardPassOutput(frame)

        
		## assign id to pigs
        new_pig_count, pig_ids, current_boxes = tracker.TrackPigs(current_boxes)
        new_pig_count_txt.markdown(f'**Total pig count:** {new_pig_count}')
        countframe_txt.markdown(f'**Frame number processed:** {countframe}')
        
        ## yolo v4 2 - classify posture
        posture_labels=[]
        for boxes,id in zip(current_boxes,pig_ids):
        	#st.text(boxes)
        	new_crop = cimage.crop_image_coordinates(boxes, frame)
        	postures, _, confidences_pos= posture_classify.ForwardPassOutput(new_crop)
        	posture_labels.append(str(postures))
        	statistics.append([frame_counter,id, postures, boxes])
        	crop = cv2.cvtColor(new_crop, cv2.COLOR_BGR2RGB)
        	##stframe.image(crop, width = 415) 

        ##st.text(posture_labels)
        ##st.text(statistics)

        ##frame = ld.drawBoxes(frame, labels, current_boxes, confidences)
        frame = cp.drawBoxes(frame, posture_labels, current_boxes, pig_ids)     
        

        end = time.time()

        frame_counter += 1
        fps_measurement = frame_counter/(end - start)
        fps_meas_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
        bar.progress(frame_counter/num_frames)

        frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frm, width = 720)
        
        n_tracker = [i+1 for i in range(new_pig_count)]
        #original_list = ['<select>']+ n_tracker
        #result = st.selectbox('Select the graph you want to display', original_list)
        #if result == '<select>':
        #    result == 1
        

    
        bars= plot_behavior_id(statistics, new_pig_count)
        ##time.sleep(0.01)
        
        bar_plot.altair_chart(bars)          
    


    
def intro():
    
    st.subheader('This app takes videos from livestock living environment.\n'+'\nThe video is analysed and classified using the states of the art Machine Learning algorithm YOLO v4.\n'+'\nDataset: User can upload videos or choose the streaming video recorded from the pig pen\n.'+ '\n'+
    '\nThere are two functionalities: Behavior detection and Behavior statistics.\n'+
    'You can choose the options from the dropdown on the left.\n' )
    front_image = drive_path+'images/download.jpg'
    st.image(front_image)
    
    
    
def behavior_detection():
    
    obj_detector = load_obj_detector(config_detection, wt_file_detection)
    posture_classify = load_posture_classify(config_classification, wt_file_classificaton)

    tracker = ld.PigsInFrameTracker(num_previous_frames = 10, frame_shape = (720, 1080))
    
    state = SessionState.get(upload_key = None, enabled = True, 
    start = False, conf = 70, nms = 50, run = False)
    hide_streamlit_widgets()    
    
    
    """
      
    Choose or upload a video file to track and count Pigs. Don't forget to change parameters to tune the model!

    """   
    
    with st.sidebar:
        """
        ## :floppy_disk: Parameters  
        """
        state.conf, state.nms = parameter_sliders(
            keys, state.enabled, value = [state.conf, state.nms])
        
        st.text("")
        st.text("")
        st.text("")

        """

        """

    #set model confidence and nms threshold 
    if (state.conf is not None):
        obj_detector.confidence = state.conf/ 100
    if (state.nms is not None):
        obj_detector.nms_threshold = state.nms/ 100 
 
    opinion = ['Pig Pen 1','Upload video']
    file_input = st.radio('Select opinion:', opinion)

    global tfile, start_button, f
    if file_input == 'Pig Pen 1':
        f = 'images/test_pig.mp4'
        tfile  = f
        ##st.video(tfile)
    else:
        upload = st.empty()
        f = st.file_uploader('Upload Video file (mpeg/mp4 format)', key = state.upload_key)
        tfile  = ''
        if f is not None:
          tfile  = tempfile.NamedTemporaryFile(delete = True)
          tfile  = tfile.write(f.read())
          upload.empty()
        
        
    if tfile is not None: 
        ##tfile  = tempfile.NamedTemporaryFile(delete = True)
            

        start_button = st.empty()
        stop_button = st.empty()

        vf = cv2.VideoCapture(tfile)


        if not state.run:
            start = start_button.button("start")
            state.start = start

        if state.start:
            start_button.empty()
            #state.upload_key = str(randint(1000, int(1e6)))
            state.enabled = False
            if state.run:
                ##tfile.close()
                ##f.close()
                state.upload_key = str(randint(1000, int(1e6)))
                state.enabled = True
                state.run = False
                ProcessFrames(vf, tracker, obj_detector, stop_button,posture_classify)
            else:
                state.run = True
                trigger_rerun()
                
def stats():
    st.subheader('This performance metric for the models.\n')
    metrics_image = drive_path+'darknet/chart.png'
    st.image(metrics_image)


if __name__=='__main__':
  main()



