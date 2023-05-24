import streamlit as st
import cv2
import torch
from utils.hubconf import  loadModel
import numpy as np
import tempfile
import time
from collections import Counter
from model_utils import get_save_stat,showInfo,getDFPredict , get_system_stat,get_predict,loadFood
from numpy import random
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import os 
import gdown

st.sidebar.title('Settings')
path_model_file='./last.pt'
if not os.path.isfile(path_model_file):
    url = 'https://drive.google.com/file/d/1Vp0oBDJl1OAF3W70rHzPw1NAfB8M2l8Z/view?usp=drive_link'
    output_path = 'last.pt'
    gdown.download(url, output_path, quiet=False,fuzzy=True)
food_names,calories,df_nf=loadFood()

# Choose the model
model_type = st.sidebar.selectbox(
    'Choose YOLO Model', ( 'YOLOv7','Info')
)

# st.title(f'YOLOv7 Predictions')

TITLE1 = st.empty()
TITLE2 = st.empty()
save1 = st.empty()
save2 = st.empty()
save3 = st.empty()
sample_img = cv2.imread('./assets/logo.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')


cap = None
vid =None
if model_type == 'Info':
    FRAME_WINDOW.empty()
    showInfo()

# YOLOv7 Model
if model_type == 'YOLOv7':
    TITLE1.title("Food Detection 🍔📷")
    TITLE2.header("Identify what's in your food photos!")
    # GPU
    gpu_option = st.sidebar.radio(
        'PU Options:', ('CPU', 'GPU'))

    if not torch.cuda.is_available():
        st.sidebar.warning(
            'CUDA Not Available, So choose CPU', icon="⚠️")
    else:
        st.sidebar.success(
            'GPU is Available on this Device, Choose GPU for the best performance',
            icon="✅"
        )
    # Model
    if gpu_option == 'CPU':
        # model = custom(path_or_model=path_model_file)
        model = loadModel(path_or_model=path_model_file)
        # model = None
    if gpu_option == 'GPU':
        # model = custom(path_or_model=path_model_file, gpu=True)
        model = loadModel(path_or_model=path_model_file, gpu=True)


    # Load Class names
    class_labels = model.names

    # Inference Mode
    options = st.sidebar.radio(
        'Options:', ('Image', 'Webcam'), index=0)

    # Confidence
    confidence = st.sidebar.slider(
        'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

    # Draw thickness
    draw_thick = st.sidebar.slider(
        'Draw Thickness:', min_value=1,
        max_value=20, value=2
    )


    # random.seed(2)
    color_pick_list = [[random.randint(0, 255) for _ in range(3)] for _ in class_labels]
    # color_pick_list = []    
    # for i in range(len(class_labels)):
    #     classname = class_labels[i]
    #     color = color_picker_fn(classname, i)
    #     color_pick_list.append(color)

    # Image
    if options == 'Image':
        upload_img_file = st.sidebar.file_uploader(
            'Upload Image', type=['jpg', 'jpeg', 'png'])
        if upload_img_file is not None:
            st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
            save2 = st.empty()
            save3 = st.empty()
            # Read image
            file_bytes = np.asarray(
                bytearray(upload_img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            FRAME_WINDOW.image(img, channels='BGR')
            # Get predict
            img, current_no_class,listIdPred = get_predict(
                img, model, confidence, color_pick_list, class_labels, draw_thick)
            FRAME_WINDOW.image(img, channels='BGR')

            # Current number of classes
            # class_fq = dict(
            #     Counter(i for sub in current_no_class for i in set(sub)))
            # class_fq = dict(Counter(current_no_class))
            # idCounter = Counter(listIdPred)
            # print(idCounter)
            # print(class_fq)
            # class_fq = json.dumps(class_fq, indent=4)
            # class_fq = json.loads(class_fq)
            # df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
            # Get df nutrition
            # df = pd.read_json(r'./food.json')
            # df = df.explode(['nf', 'value'])
            # df.value = df.value.astype('float').round(2)
            # listname = [1]
            # df=df[df["id"].apply(lambda x : x in listname)]
            # food_names,calories,df_nf=loadFood()
            
            # listIdPred =[0,0,0,0,1]
            df_result = getDFPredict(listIdPred,food_names,calories)
            df_nf=df_nf[df_nf["id"].apply(lambda x : x in listIdPred)]
            get_save_stat(False,save2,save3,False,df_result,df_nf)
    # Video
    # if options == 'Video':
    #     upload_video_file = st.sidebar.file_uploader(
    #         'Upload Video', type=['mp4', 'avi', 'mkv'])
    #     if upload_video_file is not None:
    #         pred = st.checkbox(f'Predict Using YOLOv7')

    #         tfile = tempfile.NamedTemporaryFile(delete=False)
    #         tfile.write(upload_video_file.read())
    #         cap = cv2.VideoCapture(tfile.name)

    # Web-cam
    if options == 'Webcam':
        cam_options = st.sidebar.selectbox('Webcam Channel',
                                            ('Select Channel', '0', '1', '2', '3'))

        if not cam_options == 'Select Channel':
            pred = st.checkbox(f'Predict Using YOLOv7')
            cap = cv2.VideoCapture(int(cam_options))


p_time = 0
if (cap != None) and pred:
    
    save_btn = st.button('save')
    # save1 = st.empty()
    # save2 = st.empty()
    # save3 = st.empty()
    stframe1 = st.empty()
    stframe2 = st.empty()


    if(save_btn):
        st.session_state['save'] =True
    # Save img
    if 'save' in st.session_state and 'img_save' in st.session_state:
        # df = pd.read_json(r'./food.json')
        # df = df.explode(['nf', 'value'])
        # get_save_stat(save1,save2,save3,
        #                 st.session_state['img_save'],
        #                 st.session_state['df_save'],
        #                 df
        #             )
        df_nf=df_nf[df_nf["id"].apply(lambda x : x in st.session_state['listIdPred'])]
        get_save_stat(save1,save2,save3,
                      st.session_state['img_save'],
                      st.session_state['df_result'],
                      df_nf)
    while True:
        success, img = cap.read()
       
        if not success:
            st.error(
                f"{options} NOT working\nCheck {options} properly!!",
                icon="🚨"
            )
            break
        # img1 = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
        # img = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)
        current_no_class =[]
        # Get predict
        img, current_no_class,listIdPred = get_predict(
                img, model, confidence, color_pick_list, class_labels, draw_thick)
        # img = GetPredict(img)
        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        # cv2.putText(img,f"FPS : {int(fps)}", (20,30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
        FRAME_WINDOW.image(img, channels='BGR', use_column_width=True)
        st.session_state['img_save']= img

        # Current number of classes
        # class_fq = dict(
        #     Counter(i for sub in current_no_class for i in set(sub)))
        # class_fq = json.dumps(class_fq, indent=4)
        # class_fq = json.loads(class_fq)
        # df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
        
        # listIdPred =[0,0,0,0,1]
        df_result = getDFPredict(listIdPred,food_names,calories)
        st.session_state['df_result']= df_result
        st.session_state['listIdPred']= listIdPred

        # Updating Inference results
        get_system_stat(stframe1, stframe2, fps, df_result)
    cap.release()
    
else:
    if 'save' in st.session_state:
        del st.session_state['save']
    if  'img_save' in st.session_state:
        del st.session_state['img_save']
    if  'df_save' in st.session_state:
        del st.session_state['df_result']
    if  'listIdPred' in st.session_state:
        del st.session_state['listIdPred']

