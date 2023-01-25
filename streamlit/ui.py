import io

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

# interact with FastAPI endpoint
backend = "http://85.143.174.110:8000/segmentation"

image = st.file_uploader("insert image")  # image upload widget
c1, c2= st.columns(2)

def process(image, server_url: str):

    m = MultipartEncoder(
        fields={'file': ('filename', image, 'image/jpeg')}
        )

    r = requests.post(server_url,
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000)

    return r


if st.button('Get segmentation map'):

    if image == None:
        st.write("Insert an image!")  # handle case with no image
    else:
        original_image = Image.open(image).convert("RGB")
        c1.header("Original")
        c1.image(original_image, use_column_width=True)
        segments = process(image, backend)
        c2.header('Output')
        c2.subheader('Predicted class :')
        c2.write(segments.content.decode())
