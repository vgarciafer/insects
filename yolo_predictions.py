import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred:
    def __init__(self,onnx_model,data_yaml, size):
        with open(data_yaml, mode='r') as f:
            data_yaml=yaml.load(f, Loader=SafeLoader)
        self.labels=data_yaml['names']
        self.nc=data_yaml['nc']
        self.size=size

        #load yolo model
        self.yolo =cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) #CPU o CUDA

                                 
    def predictions_v5(self,image):

        row, col, d=image.shape #altura, ancho, profundidad

        #convertir la imagen en una imagen cuadrada por exigencia de yolo
        max_rc=max(row, col)
        input_image=np.zeros((max_rc, max_rc, 3), dtype=np.uint8)#matriz cuadrada de tipo imagen (uint8)
        input_image[0:row, 0:col]=image

        #se pasa la imagen a yolo y obtener predicciones
        input_width = self.size
        blob=cv2.dnn.blobFromImage(input_image, 1/255, (input_width, input_width), swapRB=True, crop=False) #se normaliza sobre 255
        self.yolo.setInput(blob)
        preds=self.yolo.forward() #detections or prediction from YOLO
     
        #non maximum suppression
        #filtrar detecciones según el intervalo de confianza (0.4) y la probabilidad(0.25)
        detections=preds[0]
        boxes=[]
        confidences=[]
        classes=[]

        #altura y ancho
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/input_width
        y_factor = image_h/input_width

        for i in range(len(detections)):
            row=detections[i]
            confidence=row[4] #confianza de obtener un objeto
            
            if confidence > 0.4:
                class_score=row[5:].max() #mayor probabilidad del objeto de todas las clases
                class_id=row[5:].argmax() #el indice en el que la mayor probabilidad está
                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
             
                    #construcción del marco
                    #left, top, width and height
                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy - 0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)

                    box=np.array([left, top, width, height])

                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        boxes_np=np.array(boxes).tolist()
        confidences_np=np.array(confidences).tolist()

        #NMS
        index=np.array(cv2.dnn.NMSBoxes(boxes_np,confidences_np, 0.25, 0.45)).flatten()


        for ind in index:
                #extraer marcos
                x, y, w,h=boxes_np[ind]
                bb_conf=int(confidences_np[ind]*100)
                classes_id=classes[ind]
                class_name=self.labels[classes_id]
                colors=self.generate_colors(classes_id)

                text = f'{class_name}: {bb_conf}%'
                cv2.rectangle(image, (x,y), (x+w, y+h), colors, 2)
                cv2.rectangle(image, (x,y-30), (x+w, y), colors,-1)
                cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 1)

        return image
    
     
    def generate_colors(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100,255,size=(self.nc,3)).tolist()
        return tuple(colors[ID])
    
    def predictions_v8(self,image):

        row, col, d=image.shape #altura, ancho, profundidad

        #convertir la imagen en una imagen cuadrada por exigencia de yolo
        max_rc=max(row, col)
        input_image=np.zeros((max_rc, max_rc, 3), dtype=np.uint8)#matriz cuadrada de tipo imagen (uint8)
        input_image[0:row, 0:col]=image

        #se pasa la imagen a yolo y obtener predicciones
        input_width = self.size
        blob=cv2.dnn.blobFromImage(input_image, 1/255, (input_width, input_width), swapRB=True, crop=False) #se normaliza sobre 255
        self.yolo.setInput(blob)
         # Perform inference
        outputs = self.yolo.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]
       
        boxes = []
        scores = []
        class_ids = []

        #altura y ancho
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/input_width
        y_factor = image_h/input_width

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
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
             
 
            
         # NMS 
        index=np.array(cv2.dnn.NMSBoxes(boxes,scores, 0.25, 0.45)).flatten()
        boxes_np=np.array(boxes).tolist()
        confidences_np=np.array(scores).tolist()

        for ind in index:
                #extraer marcos
                x, y, w,h=boxes_np[ind]
                bb_conf=int(confidences_np[ind]*100)
                classes_id=class_ids[ind]
                class_name=self.labels[classes_id]
                colors=self.generate_colors(classes_id)

                    
                text = f'{class_name}: {bb_conf}%'
                cv2.rectangle(image, (x,y), (x+w, y+h), colors, 2)
                cv2.rectangle(image, (x,y-30), (x+w, y), colors,-1)
                cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 1)

        return image
    