import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np


st.set_page_config(page_title="YOLO object detection ", layout='wide')

st.title("Welcome to YOLO for Image")
st.write("Please Uploaded Image to get detections")

with st.spinner('Please wait your model is loading'):
    yolo5=YOLO_Pred(onnx_model='./models/yolov5.onnx', data_yaml='./models/data.yaml',size=640)
    yolo8=YOLO_Pred(onnx_model='./models/yolov8.onnx', data_yaml='./models/data.yaml',size=1280)
    #st.balloons()

def upload_image():
    #upload image
    image_file=st.file_uploader(label='Upload Image')
    if image_file is not None:
        size_mb=image_file.size/(1024**2)
        file_details = {"filename":image_file.name,
                    "filetype": image_file.type,
                    "filesize": "{:,.2f} MB".format(size_mb)}
        st.json(file_details)

        #validate file
        if file_details['filetype'] in ('image/png', 'image/jpeg'):
            st.success('Valid image type')
            return {"file": image_file, "details": file_details}
        else: 
            st.error('invalid image type. Uploaded only png, jpg, jpeg')
            return None
def main():
    object = upload_image()

    if object:
        prediction=False
        image_object=Image.open(object['file'])
        

        col1, col2=st.columns(2)
        with col1:
             st.info("Preview of Image")
             st.image(image_object)
        with col2:
             st.subheader("Check below for file details")
             st.json(object['details'])

             options=("YOLOv5", "YOLOv8")
             radio_b=st.radio("YOLO version", options, index=0)
        
             if radio_b:
                  with st.spinner("""
                        Getting objects from image. Please wait
                                  """):
                    image_array=np.array(image_object)
                    if radio_b == "YOLOv5":
                        pred_img=yolo5.predictions_v5(image_array)
                    else:
                        pred_img=yolo8.predictions_v8(image_array)
                    pred_img_obj=Image.fromarray(pred_img)
                    prediction=True
            
        if prediction:
             st.subheader("Predicted Image")
             st.write("Object detection from ", radio_b)
             st.image(pred_img_obj)

if __name__ == "__main__":
        main()
                   

