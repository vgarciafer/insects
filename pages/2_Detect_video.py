import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
import os


st.set_page_config(page_title="YOLO object detection ", layout='wide')

st.title("Bienvenidos a la detección de insectos en vídeos")
st.write("Por favor, suba un vídeo para obtener detecciones de insectos")

#modelos predefinidos para la detección de videos
with st.spinner('Please wait your model is loading'):
    yolo_5=YOLO_Pred(model='5/best_yolo_640_5.torchscript', data_yaml='./models/5/data_5.yaml',size=640, type="pt")
    rtdetr_5=YOLO_Pred(model='5/best_rtdetr_640_5.pt', data_yaml='./models/5/data_5.yaml',size=640, type="rtdetr")
    yolo8_allinsects=YOLO_Pred(model='insect/best_yolo_1280_insect.pt', data_yaml='./models/insect/data_all_insect.yaml',size=1280,  type= "pt")
    rtdetr_allinsects=YOLO_Pred(model='insect/best_rtdetr_640_insect.pt', data_yaml='./models/insect/data_all_insect.yaml',size=640,  type= "rtdetr")

#función de subida de vídeo
def upload_video():

     #subida de imagen
    
    video_file=st.file_uploader(label='Seleccione un video', type=["mp4", "MOV", "avi"])
    
    
    #estado de las sesiones para comprobar si el vídeo se ha subido
    
    if st.session_state.video_tmp is not None:
     
        if video_file is not None:
         if  video_file.name != st.session_state.video_name:
             st.session_state.video_tmp = None


    #si el vídeo se ha seleccionado y no aparece en la sesión, se guarda el vídeo en la carpeta temporal
    if video_file is not None and st.session_state.video_tmp is None :
        
        size_mb=video_file.size/(1024**2)
        #se guardan los detalles del vídeo
        file_details = {"videoname":video_file.name,
                    "type": video_file.type,
                    "size": "{:,.2f} MB".format(size_mb)}
        st.json(file_details)

        #se guarda con fecha hora y segundos de subida
        str_hoy = f'{datetime.now():%Y%m%d%H%M%S%z}'
        temp_file_to_save = 'videos/temp'+  str_hoy + video_file.name

        #se guarda 
        write_bytesio_to_file(temp_file_to_save, video_file)
        #se guarda en la sesión el nombre del fichero temporal
        st.session_state.video_tmp = temp_file_to_save
        st.session_state.video_name = video_file.name


        return {"file": temp_file_to_save, "details": file_details}
    else:
        return None

# función para guardar el fichero
def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())
  

def main():
   
   #se muestra el vídeo original o el vídeo con la detección

    if "expander_state_detection" not in st.session_state:
        st.session_state["expander_state_detection"] = False
    if "expander_state_video" not in st.session_state:
        st.session_state["expander_state_video"] = True
    if "video_tmp" not in st.session_state:
        st.session_state["video_tmp"]= None
    if "video_name" not in st.session_state:
        st.session_state["video_name"]= None

    #subida del fichero
        
    object = upload_video()

    #conmutador para decidir qué se muestra, el video original o los fotogramas con las detecciones
    def toggle_closed():
        st.session_state["expander_state_detection"] = False
        st.session_state["expander_state_video"] = True
    
    def toggle():
        st.session_state["expander_state_detection"] = True
        st.session_state["expander_state_video"] = False
    
    #video temporal guardado en la sesión 
    video_uploaded=st.session_state["video_tmp"] 
    
    #si existe el vídeo se muestra
    if video_uploaded is not None:
        if object:
            video_tmp=object['file']
        else:
             video_tmp=video_uploaded
        #se muestra el video original
        with st.expander('Original',  expanded=st.session_state["expander_state_video"]):

               
                st.write("Previsualización del vídeo")
                options=("YOLOv8 Track-5 especies",  "RT-DETR-5 especies", "Yolov8 Track insecto", "RT-DETR Track insecto")
                radio_b=st.radio("Version", options)
                start=st.button("Comenzar", on_click=toggle)
                if start not in st.session_state:
                    st.session_state.button = False
                st.video(video_tmp )
        #se muestran los fotogramas con las detecciones    
        with st.expander('Video con las detecciones', expanded=st.session_state["expander_state_detection"]):
             st.write("Detección con el modelo ", radio_b)
             stop=st.button("Stop", on_click=toggle_closed)
             if start:
                 with st.spinner('La detección del vídeo se está cargando..'): 
                        
                        cap =cv2.VideoCapture(video_tmp)
                        
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frame_fps = cap.get(cv2.CAP_PROP_FPS)  
                        
                        
                        st_frame = st.empty()
                        vid_cap = cv2.VideoCapture(video_uploaded)

                        
                        num_frame=0
                            
                        while (vid_cap.isOpened()):
                            success, frame = vid_cap.read()
                            if success:
                                num_frame +=1
                                fr = cv2.resize(frame, (640, int(640*(9/16))))
                                if radio_b == "YOLOv8 Track-5 especies":
                                    pred_img=yolo_5.predictions_track_pt(frame)
                                elif radio_b == "RT-DETR-5 especies":
                                    pred_img=rtdetr_5.predictions_track_pt(frame)
                                elif radio_b== "Yolov8 Track insecto" :
                                    pred_img=yolo8_allinsects.predictions_track_pt(frame)
                                else:
                                     pred_img=rtdetr_allinsects.predictions_track_pt(frame)
                                st_frame.image(pred_img,
                                           caption='Video anotado',
                                            channels="BGR",
                                            use_column_width=True
                                            )
                                frame_save = 'videos/temp'+  str(num_frame) +".jpg"
                            
                                nombre_img_pred="fr_"+ radio_b + st.session_state["video_name"] + str(num_frame)+ "_pred.jpg"
                                #nombre_img_orig="fr_"+ st.session_state["video_name"] + str(num_frame)+ "_orig.jpg"
                                #ruta_img_pred = os.path.join(os.getcwd(),"videos", "track3",nombre_img_pred)
                                #ruta_img_orig= os.path.join(os.getcwd(),"videos/track2/", nombre_img_orig)
                    
                            
                                #cv2.imwrite(ruta_img_pred, pred_img)
                                #cv2.imwrite(ruta_img_orig, frame)
                            else:
                                vid_cap.release()
                                break
                            
                    
                   
                         

    else:
         st.write("No se ha encontrado ningún video")

if __name__ == "__main__":
        main()
                   

