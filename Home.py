import streamlit as st
import yaml
import base64

st.set_page_config(page_title="Home", layout='wide', page_icon='./images/house.png')
st.title("Detection App")

yaml_file='./models/12/data_12.yaml'
with open(yaml_file) as f:
    data = yaml.safe_load(f)

main_bg_ext = "png"
main_bg = "./images/imagen1.JPG"
st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


#Content
st.markdown(f"""
     ### Esta App detecta insectos en imágenes o en video.
    Detecta {data["nc"]} tipos de insectos. 
    Las especies que nuestro modelo puede detectar son algunas de las siguientes:   
             """)


for i, specie in enumerate(data["names"]):
   st.write(str(i + 1) + '. ' + str(specie))    




