import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
import cv2
import av
import io
from datetime import datetime


st.set_page_config(page_title="YOLO object detection ", layout='wide')

st.title("Welcome to YOLO for Video")
st.write("Please Uploaded Image to get detections")

with st.spinner('Please wait your model is loading'):
    yolo5=YOLO_Pred(onnx_model='./models/yolov5.onnx', data_yaml='./models/data_yolov5.yaml',size=640)
    yolo8=YOLO_Pred(onnx_model='./models/yolov82.onnx', data_yaml='./models/data_yolov8_2.yaml',size=1280)
    #st.balloons()

def upload_video():
    #upload image
    video_file=st.file_uploader(label='Upload video')
    if video_file is not None:
        size_mb=video_file.size/(1024**2)
        file_details = {"videoname":video_file.name,
                    "type": video_file.type,
                    "size": "{:,.2f} MB".format(size_mb)}
        st.json(file_details)
        return {"file": video_file, "details": file_details}
    else:
        return None

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

def main():
    object = upload_video()
    

    if object:
        video=object['file']
        
        col1, col2=st.columns(2)
        str_hoy = f'{datetime.now():%Y%m%d%H%M%S%z}'
        temp_file_to_save = 'videos/temp'+  str_hoy +'.mp4'
        write_bytesio_to_file(temp_file_to_save, video)
        
        with col1:
            prediction=False
            st.write("prev video")
            print(temp_file_to_save)
            st.video(temp_file_to_save)
            
        with col2:
             st.subheader("Check below for file details")
             st.json(object['details'])
             options=("YOLOv5", "YOLOv8")
             radio_b=st.radio("YOLO version", options, index=0)
             if radio_b:
                 with st.spinner('Please wait your model is loading'): 
                        cap =cv2.VideoCapture(video.name)
                        
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frame_fps = cap.get(cv2.CAP_PROP_FPS)  
                        st.write(width, height, frame_fps)
                        
                        vid_cap = cv2.VideoCapture(temp_file_to_save)
                        st_frame = st.empty()
                        while (vid_cap.isOpened()):
                            success, frame = vid_cap.read()
                            if success:
                                fr = cv2.resize(frame, (720, int(720*(9/16))))
                                if radio_b == "YOLOv5":
                                    pred_img=yolo5.predictions_v5(frame)
                                else:
                                    pred_img=yolo8.predictions_v8(frame)
                                st_frame.image(pred_img,
                                            caption='Detected Video',
                                            channels="BGR",
                                            use_column_width=True
                                            )
                            else:
                                vid_cap.release()
                                break
                       
                    
                    
                   
                         

    else:
         st.write("no hay video")

if __name__ == "__main__":
        main()
                   

