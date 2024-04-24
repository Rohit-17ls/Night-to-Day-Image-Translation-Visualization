import streamlit as st
import io
from PIL import Image
import torchvision.transforms as transforms
import model

model_store = model.GAN_Model_Store()

def update_selected_model():
    GAN_model = model_store.get_model(selected_model);    


st.header("Night to Day Image Translation using GANs");

selected_model = st.selectbox("Choose GAN model", 
                             ("CycleGAN", "DiscoGAN", "ToDayGAN", "UNITGAN"),
                             on_change = update_selected_model);

GAN_model = model_store.get_model(selected_model);

st.write(f"Selected Model : {selected_model}")


uploaded_file = st.file_uploader("Pick Image", accept_multiple_files = False);

if(uploaded_file is not None):
    image_cols = st.columns(2);
    
    img = Image.open(uploaded_file).convert('RGB')
    img = transforms.Resize((128, 128), Image.BICUBIC)(img)
    height, width = img.size

    

    with image_cols[0]:
        st.image(img);
        st.write("Input Image");



    with image_cols[1]:
        translated_img = model.translate(img, GAN_model);
        st.image(translated_img)
        st.write("Translated Image");

    
