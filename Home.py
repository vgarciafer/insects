import streamlit as st

st.set_page_config(page_title="Home", layout='wide', page_icon='./images/house.png')
st.title("Yolo Object Detection App")
st.caption("This web application demostrate Object Detection")

#Content
st.markdown("""
     ### This App detects objects from Images
    -Automatically detects 20 types of insects from image
            
    -[Click here for App](/YOLO_for_image/)     

    


    Below give are the object that our model with detect:   
    1. unidentified
    2.  ant
    3.  apis mellifera
    4.  bee
    5.  beetle
    6.  bombus terrestris5
    7.  butterfly
    8.  danaus chrysippus
    9.  eucera
    10. fly
    11.  hoverfly
    12.  lasiommata megera
    13.  megascolia maculata
    14.  moth
    15.  oedemera barbara
    16.  papilio machaon
    17.  pieris rapaee
    18.  pieris rape
    19.  polyommatus celina
    20.  scaeva pyrastri
    21.  vanessa cardui
    22.  wasp
    23.  xylocopa violacea 
          
             """)



