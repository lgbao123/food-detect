from utils.plots import plot_one_box
from PIL import ImageColor

import streamlit as st
import torch
import cv2
from utils.general import check_img_size,non_max_suppression,scale_coords
import pandas as pd
from utils.datasets import  letterbox
import numpy as np
import torch.backends.cudnn as cudnn
import random
from PIL import Image
import plotly.express as px
import pandas as pd
from collections import Counter


def color_picker_fn(classname, key):
    # r,g,b = color
    # color= '#{:02x}{:02x}{:02x}'.format(r, g, b)
    random.seed(key)
    color = f'#{"%06x" % random.randint(0, 0xFFFFFF)}'
    color_picke = st.sidebar.color_picker(f'{classname}:',color, key=key)
    color_rgb_list = list(ImageColor.getcolor(str(color_picke), "RGB"))
    color = [color_rgb_list[2], color_rgb_list[1], color_rgb_list[0]]
    return color



def get_predict(img, model, confidence, color_pick_list, draw_thick):

    current_no_index = []
    device = torch.device('cuda') if next(model.parameters()).is_cuda else torch.device('cpu')
    # print(next(model.parameters()).is_cuda)
   
    imgsz =640
    stride = int(model.stride.max())  # model stride
    img0 =img
    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  
    img = np.ascontiguousarray(img)
    # print(img.shape)
    img = torch.from_numpy(img).to(device)
    img =  img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    with torch.no_grad(): 
        results = model(img)[0]
    results = non_max_suppression(results, confidence)
    # Rescale boxes from img_size to img0 size
    results[0][:, :4] = scale_coords(
            img.shape[2:], results[0][:, :4], img0.shape )
    box = pd.DataFrame({
                'xmin':  [dec[0].cpu().detach().numpy() for dec in results[0]],
                'ymin':  [dec[1].cpu().detach().numpy() for dec in results[0]],
                'xmax':  [dec[2].cpu().detach().numpy() for dec in results[0]],
                'ymax':  [dec[3].cpu().detach().numpy() for dec in results[0]],
                'confidence':  [dec[4].cpu().detach().numpy() for dec in results[0]],
                'class':  [dec[5].cpu().detach().numpy().astype(int) for dec in results[0]],
                'name':  [model.names[dec[5].cpu().detach().numpy().astype(int)] for dec in results[0]],
            })
    # print(box)

        # box = results.pandas().xyxy[0]
    for i in box.index:
        xmin, ymin, xmax, ymax, conf, id, class_name = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
            int(box['ymax'][i]
                ), box['confidence'][i], box['class'][i], box['name'][i]
        if conf > confidence:
            plot_one_box([xmin, ymin, xmax, ymax], img0, label=f'{class_name} {conf:.2f}',
                            color=color_pick_list[id], line_thickness=draw_thick)
          
            current_no_index.append(int(id))
    return img0,current_no_index

@st.cache_data
def getStyleDF():
    td_props = [
        ('font-size', '14px'),
        ('text-align', 'center'),
        ('width','100vw')]
    headers_props = [
        ('text-align','center'),
        ('font-size','1em')
    ]
    styles = [
        dict(selector="td", props=td_props),
        dict(selector='th.col_heading.level0',props=headers_props),
        dict(selector='th.col_heading.level1',props=td_props),
    ]
    return styles

def get_current_detect( stframe2, df_fq):
    # Updating Inference results
    # if stframe1 :
        
    #     with stframe1.container():
    #         st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
    #         if round(fps, 4) > 1:
    #             st.markdown(
    #                 f"<h4 style='color:green;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
    #         else:
    #             st.markdown(
    #                 f"<h4 style='color:red;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)

    with stframe2.container():
        st.markdown("<h3>Detected objects in curret Frame</h3>",
                    unsafe_allow_html=True)
        # st.dataframe(df_fq, use_container_width=True)
        styles = getStyleDF()
        st.markdown(df_fq.style.set_table_styles(styles).to_html(),unsafe_allow_html=True)



def get_save_stat(stframe1, stframe2, stframe3, img, df1 ,df2):
    # Updating Inference results
    if stframe1:
        with stframe1.container():
            st.markdown("<h3>Save Image</h3>", unsafe_allow_html=True)
            st.image(img, channels='BGR')

    with stframe2.container():
        st.markdown("<h3>Detected objects in curret Frame</h3>",unsafe_allow_html=True)
        # st.dataframe(df1, use_container_width=True)
        styles = getStyleDF()
        st.markdown(df1.style.set_table_styles(styles).to_html(),unsafe_allow_html=True)
        st.write("###")
        nums =  list(df1["Number"])
        cals =  list(df1["Calories"])
        res_lt = sum([ nums[x] * cals[x] for x in range (len (nums))])

        st.success(f'Total food calories : {res_lt}')
    with stframe3.container():
        fig = px.bar(df2, x="value", y="food_name", color='nf', orientation='h',barmode="group",text=f'value',
                        labels={
                            "value": " Daily Value (gam)",
                            "food_name": "Food name",
                            "nf": "Nutrition Facts"
                        },
                        
                    )
        fig.update_layout(plot_bgcolor="#f0f2f6")
        # fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        fig.update_layout(height=600)
        st.markdown("<h3>Nutritive Value of Foods</h3>",unsafe_allow_html=True)
        st.plotly_chart(fig)
        
@st.cache_data
def loadFood():
    df = pd.read_json(r'./food.json')
    calories = [round(num) for num in list(df["nf_calories"])]
    food_names =  list(df["food_name"])
    df = df.explode(['nf', 'value'])
    df.value = df.value.astype('float').round(2)
    return food_names , calories , df

def getDFPredict(listIdPred,food_names,calories):
    data = dict(Counter(listIdPred))
    for key, values in data.items():
        name = food_names[key]
        link = f'https://www.nutritionix.com/food/{name}'
        url = f'<a target="_blank" href="{link}">More info</a>'
        new_values = [name, values,calories[key],url]
        data[key] = new_values
    df_result = pd.DataFrame(data.values(), columns=['Name', 'Number','Calories','Url'])
    return df_result


@st.cache_data
def showInfo():
    st.title(f"YOLO for Real-Time Food Detection")

    st.markdown(
        """
    The OID Data Set ([link](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=valtest&type=detection&c=%2Fm%2F014j1m)) :
    - This dataset consists of 74 food categories, with 26'176 images (21'233 training - 4943 test) 
    - Each type of food has 250-300 training samples and 50-100 test samples
    - Size : 1 gb 
    """
    )
    st.write("Workflow : ")
    image = Image.open('./assets/flow.png')
    # image1 = Image.open('./extras/before_pre.png')
    # image2 = Image.open('./extras/after_pre.png')
    # pre_image = Image.open('./extras/pre.png')
    # fit_2_image = Image.open('./extras/2_fit.png')
    # fit_10_image = Image.open('./extras/10_fit.png')
    # fit_101_image = Image.open('./extras/101_fit.png')
    st.image([image], use_column_width=True)
    # st.image([image1,image2])
    st.markdown(
        """
    Pretrained model :
    - Pretrained on the MS COCO dataset for 300 epochs.
    - The COCO (Common Objects in Context) dataset is a large-scale image recognition dataset for object detection, segmentation, and captioning tasks.
    - It contains over 330,000 images, each annotated with 80 object categories. [link](https://cocodataset.org/#home)
    """
    )

    st.markdown(
        """
    YOLOv7 Architecture :
    - Backbone: ELAN
    - Neck: FPN
    - Head: YOLOR 
    """
    )
    st.write('---')
    
    st.success("Total running time for training process: ~3900 m (~65 h) (mAP : 0.86)")
    st.markdown("<h5>Evaluate the model :</h5>", unsafe_allow_html=True)
    
    st.write(
        f"You can check out this [link](https://wandb.ai/crisbao2609/OIDv2?workspace=user-crisbao2609)")
    map = cv2.imread('./assets/map.png')
    recall = cv2.imread('./assets/recall.png')
    pre = cv2.imread('./assets/pre.png')
    st.image(map, channels='BGR')
    st.image(recall, channels='BGR')
    st.image(pre, channels='BGR')
