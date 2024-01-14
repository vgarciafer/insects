from ultralytics import YOLO
import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader
from ultralytics import RTDETR

#Clase para la predicción (detección y seguimiento) de YOLO o RTDETR en formato pytorch u ONNX
class YOLO_Pred:

    #inicialización según el yaml con los tipos de insectos y el tipo de detección 
    def __init__(self,model,data_yaml, size, type):
        with open(data_yaml, mode='r') as f:
            data_yaml=yaml.load(f, Loader=SafeLoader)
        self.labels=data_yaml['names']
        self.nc=data_yaml['nc']
        self.size=size

        #carga del modelo
        MODEL_DIR = os.getcwd() + '/models/'
        MODEL_ROUTE=MODEL_DIR + model 

        #tipos admitidos ONNX o Pytorch de YOLOv8 o RTDETR
        self.onnx=False
        if (type == "onnx"):
            self.onnx= True
            self.model =cv2.dnn.readNetFromONNX(MODEL_ROUTE)
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 
        elif (type == "rtdetr"):
            self.model= RTDETR(MODEL_ROUTE)
        else:
            self.model=YOLO(MODEL_ROUTE)
       
                                 
    #generación de los colores de las etiquetas
    def generate_colors(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100,255,size=(self.nc,3)).tolist()
        return tuple(colors[ID])

    #función de predicción en formato Pytorch
    def predictions_pt(self,image):
        #detección con un mínimo de confianza del 30%
        res = self.model.predict(image, conf=0.30)  
        #se devuelve la imagen detectada            
        res_plotted = res[0].plot()
        return res_plotted

    #función de seguimiento en formato Pytorch. Devuelve la imagen con la predicción
    def predictions_track_pt(self,image):
        #track con intervalo de confianza dle 50% e IOU del 50%. El rastreador por defecto es BOTSORT
        res = self.model.track(image, persist=True,conf=0.3,  iou=0.5, tracker="botsort.yaml") 
        #se devuelve la imagen de seguimiento
        res_plotted = res[0].plot()
        return res_plotted

    #función de predicción en formato ONNX. Devuelve la imagen con la predicción
    def predictions_onnx(self,image):

        row, col, d=image.shape #altura, ancho, profundidad

        #convertir la imagen en una imagen cuadrada por exigencia de yolo
        max_rc=max(row, col)
        input_image=np.zeros((max_rc, max_rc, 3), dtype=np.uint8)#matriz cuadrada de tipo imagen (uint8)
        input_image[0:row, 0:col]=image

        #se pasa la imagen a yolo y obtener predicciones
        input_width = self.size
        blob=cv2.dnn.blobFromImage(input_image, 1/255, (input_width, input_width), swapRB=True, crop=False) #se normaliza sobre 255
        self.model.setInput(blob)
         # inferencia
        outputs = self.model.forward()

        # salida en formato array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]
       
        boxes = []
        scores = []
        class_ids = []

        #altura y ancho
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/input_width
        y_factor = image_h/input_width

        #selección de los encuadres a partir del 25% de intervalo de confianza
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            #obtención del mínimo y del máximo con sus posiciones
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
 
                left = int((outputs[0][i][0] - 0.5*outputs[0][i][2])*x_factor)
                top = int((outputs[0][i][1]- 0.5* outputs[0][i][3])*y_factor)
                width = int(outputs[0][i][2]*x_factor)
                height = int(outputs[0][i][3]*y_factor)

                box=np.array([left, top, width, height])
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
             
 
            
         # NMS (Non maximal supression) para limpiar los encuadres superpuestos y dejar solo los que tengan un mínimo de intervalo de confianza dle 25% y minimo de IOU del 40% : 
        index=np.array(cv2.dnn.NMSBoxes(boxes,scores, 0.25, 0.40)).flatten()
        boxes_np=np.array(boxes).tolist()
        confidences_np=np.array(scores).tolist()

        for ind in index:
                #extraer marcos
                x, y, w,h=boxes_np[ind]
                bb_conf=int(confidences_np[ind]*100)
                classes_id=class_ids[ind]
                class_name=self.labels[classes_id]
                colors=self.generate_colors(classes_id)

                #texto de la clase predicha, se etiqueta con los colores
                text = f'{class_name}: {bb_conf}%'
                cv2.rectangle(image, (x,y), (x+w, y+h), colors, 2)
                cv2.rectangle(image, (x,y-30), (x+w, y), colors,-1)
                cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 1)

        return image

    #función de predicción 
    def predictions(self, image):
        #si está en formato ONNX
        if self.onnx:
            img=self.predictions_onnx(image)
        #si está en formato Pytorch
        else:
            img=self.predictions_pt(image)
        
        return img
    
 
    

    