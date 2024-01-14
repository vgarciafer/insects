import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
import cv2
import os


st.set_page_config(page_title="Detección de insectos ", layout='wide')

st.title("Bienvenidos a la detección de insectos en imágenes")
st.write("Por favor, suba una imagen para obtener detecciones de insectos")

#modelos predefinidos para la detección en imágenes
with st.spinner('Por favor, espere, los modelos se están cargando'):
    yolo_12=YOLO_Pred(model='12/best_yolo_1280_12.pt', data_yaml='./models/12/data_12.yaml',size=1280, type="pt")
    rtdetr_12=YOLO_Pred(model='12/best_rtdetr_1280_12.pt', data_yaml='./models/12/data_12.yaml',size=1280, type="rtdetr")
    yolo_5=YOLO_Pred(model='5/best_yolo_640_5.pt', data_yaml='./models/5/data_5.yaml',size=640, type="pt")
    rtdetr_5=YOLO_Pred(model='5/best_rtdetr_640_5.pt', data_yaml='./models/5/data_5.yaml',size=640, type="rtdetr")
    yolo8_allinsects=YOLO_Pred(model='insect/best_yolo_1280_insect.pt', data_yaml='./models/insect/data_all_insect.yaml',size=1280,  type= "pt")
    rtdetr_allinsects=YOLO_Pred(model='insect/best_rtdetr_640_insect.pt', data_yaml='./models/insect/data_all_insect.yaml',size=640,  type= "rtdetr")
    

#función de subida de imágenes
def upload_image():
    #subida de imagen
    image_file=st.file_uploader(label='Seleccione una imagen')
    if image_file is not None:
        size_mb=image_file.size/(1024**2)
        file_details = {"nombre":image_file.name,
                    "tipo": image_file.type,
                    "tamaño": "{:,.2f} MB".format(size_mb)}
        st.json(file_details)

        #validación del tipo de imagen
        if file_details['tipo'] in ('image/png', 'image/jpeg'):
            st.success('Tipo de imagen válida')
            return {"file": image_file, "details": file_details}
        else: 
            st.error('El tipo de la imagen no es válido. Tipos admitidos: png, jpg, jpeg')
            return None
def main():
    #subida de imagen
    object = upload_image()

    #si existe imagen subida
    if object:
        prediction=False
        image_object=Image.open(object['file'])
        
        #se definen dos columnas: una para previsualizar la imagen y otro para mostrar la imagen con las detecciones resultantes
        col1, col2=st.columns(2)
        with col1:
             st.info("Previsualización de imagen")
             st.image(image_object)
        with col2:
             st.subheader("Los detalles de la imagen subida son")
             st.json(object['details'])

            #opciones para detectar de Yolo8 y RTDETR
             options=("YOLOv8 12 especies", "RTDETR 12 especies","YOLOv8 5 especies",   "RTDETR 5 especies", "YOLOv8 Insecto", "RTDETR Insecto" )
             radio_b=st.radio("Tipo de detección", options, index=0)
        
             if radio_b:
                  with st.spinner("""
                        Obteniendo detecciones en la imagen. Por favor, espere...
                                  """):
                  
                    image_array=np.array(image_object)
                   
                   #según el tipo seleccionado, se escoge la predicción
                    if radio_b=="YOLOv8 12 especies":
                        pred_img=yolo_12.predictions(image_array)
                    elif radio_b=="RTDETR 12 especies":
                        pred_img=rtdetr_12.predictions(image_array)
                    elif radio_b=="YOLOv8 5 especies":
                         pred_img=yolo_5.predictions(image_array)
                    elif radio_b=="RTDETR 5 especies":   
                        pred_img=rtdetr_5.predictions(image_array)
                    elif radio_b=="YOLOv8 Insecto":
                         pred_img=yolo8_allinsects.predictions(image_array)
                    else:
                        pred_img= rtdetr_allinsects.predictions(image_array)
                    
                    prediction=True
                    #img_name=object["details"]["nombre"]
                    #nombre_img_pred=img_name+ "_pred.jpg"
                    #ruta_img_pred = os.path.join(os.getcwd(),"pred",nombre_img_pred)
  
                    #cv2.imwrite(ruta_img_pred, pred_img)

            
        if prediction:
             st.subheader("Imagen resultante con predicciones")
             st.write("Modelo de detección escogido ", radio_b)
             st.image(pred_img)

if __name__ == "__main__":
        main()
                   

