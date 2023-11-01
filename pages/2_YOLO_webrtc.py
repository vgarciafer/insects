import streamlit as streamlit
from streamlit_webrtc import webrtc_streamer
from yolo_predictions import YOLO_Pred
import av

#load yolo models
yolo5=YOLO_Pred(onnx_model='./models/yolov5.onnx', data_yaml='./models/data.yaml',size=640)
yolo8=YOLO_Pred(onnx_model='./models/yolov8.onnx', data_yaml='./models/data.yaml',size=1280)
    
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    pred_img=yolo5.predictions_v5(img)
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video":True,"audio":False})