import streamlit as st
import base64
main_bg_ext = "png"
main_bg = "./images/imagen2.JPG"
st.subheader("Proyecto TFM Ciencia de datos 2023-2024. UOC")
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