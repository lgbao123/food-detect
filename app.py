import streamlit as st
import cv2
import torch
from utils.hubconf import  loadModel
import numpy as np



from model_utils import get_save_stat,showInfo,getDFPredict , get_current_detect,get_predict,loadFood
from numpy import random


import os 
import gdown
from streamlit_webrtc import webrtc_streamer
import av
from twilio.rest import Client
import threading

lock = threading.Lock()
img_container = {"img": None}

st.sidebar.title('Settings')
# download model
path_model_file='./last.pt'
if not os.path.isfile(path_model_file):
    url = 'https://drive.google.com/file/d/1Vp0oBDJl1OAF3W70rHzPw1NAfB8M2l8Z/view?usp=drive_link'
    output_path = 'last.pt'
    gdown.download(url, output_path, quiet=False,fuzzy=True)
#load fact food
food_names,calories,df_nf=loadFood()

# Choose the model
model_type = st.sidebar.selectbox(
    'Choose YOLO Model', ( 'YOLOv7','Info')
)

# st.title(f'YOLOv7 Predictions')
ctx = None
TITLE1 = st.empty()
TITLE2 = st.empty()
save1 = st.empty()
save2 = st.empty()
save3 = st.empty()
sample_img = cv2.imread('./assets/logo.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')


# cap = None
# vid =None
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
        'Detection Confidence', min_value=0.0, max_value=1.0, value=0.35)
    
    # Draw thickness
    draw_thick = st.sidebar.slider(
        'Draw Thickness:', min_value=1,
        max_value=20, value=3
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
            # FRAME_WINDOW.image(img, channels='BGR')
            # Get predict
            img,listIdPred = get_predict(
                img, model, confidence, color_pick_list, draw_thick)
            # FRAME_WINDOW.image(img, channels='BGR')

            # listIdPred =[0,0,0,0,1]
            df_result = getDFPredict(listIdPred,food_names,calories)
            df_nf=df_nf[df_nf["id"].apply(lambda x : x in listIdPred)]
            get_save_stat(FRAME_WINDOW,save2,save3,img,df_result,df_nf)

    # Web-cam
    if options == 'Webcam':
        # cam_options = st.sidebar.selectbox('Webcam Channel',
        #                                     ('Select Channel', '0', '1', '2', '3'))

        # if not cam_options == 'Select Channel':
        #     pred = st.checkbox(f'Predict Using YOLOv7')
        #     cap = cv2.VideoCapture(int(cam_options))
        
        def video_frame_callback(frame):
            # print(frame)
            img = frame.to_ndarray(format="bgr24")
            with lock:
                kq = []
                img,listIdPred = get_predict(
                    img, model, confidence, color_pick_list, draw_thick) 
                df_result = getDFPredict(listIdPred,food_names,calories)
                # st.session_state['df_result']= df_result
                # st.session_state['listIdPred']= listIdPred
                kq.append(img)   
                kq.append(df_result)   
                kq.append(listIdPred)   
                img_container["img"] = kq
                # print(img_container['img'])
            # return frame
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        account_sid = 'ACde6e622ff59ca01ca77864061fb5c7a6'
        auth_token = '5947621c1c5cd3e90c6b6216702826a1'
        client = Client(account_sid, auth_token)

        token = client.tokens.create()

       
        ctx= webrtc_streamer(key="example" ,sendback_audio=False ,sendback_video=True,
                            #  video_processor_factory = getFrame,
                            video_frame_callback=video_frame_callback,
                          rtc_configuration={  "iceServers": token.ice_servers},
        )
        


            



if ctx != None:
    # print('323')
    save_btn = st.button('save')

    stframe1 = st.empty()
    # stframe2 = st.empty()
    FRAME_WINDOW.empty()
    # if(save_btn):
    #     st.session_state['save'] =True
    # Save img

    while ctx.state.playing :

        with lock:
            kq  =img_container["img"]
                
            
        if kq is None:
            continue
  
        # FRAME_WINDOW.image(kq[0],channels='BGR')

        get_current_detect(stframe1, kq[1])
        if(save_btn):
            df_nf=df_nf[df_nf["id"].apply(lambda x : x in kq[2])]
            get_save_stat(save1,save2,save3,
                      kq[0],
                      kq[1],
                      df_nf)
        save_btn = False
           
        
# p_time = 0
# if (cap != None) and pred:
    
#     save_btn = st.button('save')
#     # save1 = st.empty()
#     # save2 = st.empty()
#     # save3 = st.empty()
#     stframe1 = st.empty()
#     stframe2 = st.empty()


#     if(save_btn):
#         st.session_state['save'] =True
#     # Save img
#     if 'save' in st.session_state and 'img_save' in st.session_state:
#         # df = pd.read_json(r'./food.json')
#         # df = df.explode(['nf', 'value'])
#         # get_save_stat(save1,save2,save3,
#         #                 st.session_state['img_save'],
#         #                 st.session_state['df_save'],
#         #                 df
#         #             )
#         df_nf=df_nf[df_nf["id"].apply(lambda x : x in st.session_state['listIdPred'])]
#         get_save_stat(save1,save2,save3,
#                       st.session_state['img_save'],
#                       st.session_state['df_result'],
#                       df_nf)
#     while True:
#         success, img = cap.read()
       
#         if not success:
#             st.error(
#                 f"{options} NOT working\nCheck {options} properly!!",
#                 icon="🚨"
#             )
#             break
#         # img1 = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
#         # img = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)
#         # current_no_class =[]
#         # Get predict
#         img,listIdPred = get_predict(
#                 img, model, confidence, color_pick_list, draw_thick)
#         # img = GetPredict(img)
#         # FPS
#         c_time = time.time()
#         fps = 1 / (c_time - p_time)
#         p_time = c_time
#         # cv2.putText(img,f"FPS : {int(fps)}", (20,30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
#         FRAME_WINDOW.image(img, channels='BGR', use_column_width=True)
#         st.session_state['img_save']= img

#         # Current number of classes
#         # class_fq = dict(
#         #     Counter(i for sub in current_no_class for i in set(sub)))
#         # class_fq = json.dumps(class_fq, indent=4)
#         # class_fq = json.loads(class_fq)
#         # df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
        
#         # listIdPred =[0,0,0,0,1]
#         df_result = getDFPredict(listIdPred,food_names,calories)
#         st.session_state['df_result']= df_result
#         st.session_state['listIdPred']= listIdPred

#         # Updating Inference results
#         get_system_stat(stframe1, stframe2, fps, df_result)
#     cap.release()
    
# else:
#     if 'save' in st.session_state:
#         del st.session_state['save']
#     if  'img_save' in st.session_state:
#         del st.session_state['img_save']
#     if  'df_save' in st.session_state:
#         del st.session_state['df_result']
#     if  'listIdPred' in st.session_state:
#         del st.session_state['listIdPred']

