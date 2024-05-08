import streamlit as st
import io
import time
import cv2
from PIL import Image
import torchvision.transforms as transforms
import model

model_store = model.GAN_Model_Store()
FRAMES_PER_TRANSLATION = 4

def update_selected_model():
    GAN_model = model_store.get_model(selected_model);    


st.header("Night to Day Image Translation using GANs");

selected_model = st.selectbox("Choose GAN model", 
                             ("CycleGAN", "DiscoGAN", "ToDayGAN", "UNITGAN", "DualGAN", "SegmentGAN"),
                             on_change = update_selected_model);

GAN_model = model_store.get_model(selected_model);

st.write(f"Selected Model : {selected_model}")


def render_image_model_translation(uploaded_file):

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

    
# TODO : cv2.VideoCapture(path) requires the path and does not accept an UploadedFile object 
def render_video_model_translation(uploaded_file):
    # print('HERE')
    video_object = cv2.VideoCapture(f'./public/{uploaded_file.name}');


    frames_exist, frame = video_object.read()

    if 'img' not in st.session_state:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.session_state['img'] = transforms.Resize((128, 128), Image.BICUBIC)(pil_img)

    if 'translated_img' not in st.session_state:
        st.session_state['translated_img'] = model.translate(st.session_state['img'], GAN_model)
    
    if 'run' not in st.session_state:
        st.session_state['run'] = 0

    image = translated_image = None
    input_label = output_label = None

            

    image_frames = []
    translated_image_frames = []

    frame_ind = 1;
    while True:

        frames_exist, frame = video_object.read()
        
        if not frames_exist:
            break

        if frame_ind % FRAMES_PER_TRANSLATION == 0:

            # print(f'LOG : Rendering {frame_ind}')
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB));

            image_frames.append(transforms.Resize((128, 128), Image.BICUBIC)(pil_img));
            translated_image_frames.append(model.translate(st.session_state['img'], GAN_model))

        frame_ind += 1
    
    def show_translation():
        st.session_state['run'] += 1
        st.session_state['img'] = image_frames[st.session_state['run']]
        st.session_state['translated_img'] = translated_image_frames[st.session_state['run']]
        

    image_cols = st.columns(2);
            
    with image_cols[0]:
        image = st.image(st.session_state['img'])
        input_label = st.write("Input Image");

    with image_cols[1]:
        translated_image = st.image(st.session_state['translated_img'])
        output_label = st.write("Translated Image")
    
    translated_btn = st.button('Translate', on_click = show_translation)
    
    


uploaded_file = st.file_uploader("Pick Image", accept_multiple_files = False);

if(uploaded_file is not None):

    if uploaded_file.name.split(".")[-1].lower() == "mp4":
        print(f'LOG : .mp4 file uploaded')
        render_video_model_translation(uploaded_file)
    else:
        render_image_model_translation(uploaded_file)

