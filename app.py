import numpy as np

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import streamlit as st

PATH = "model-weights/"
WEIGHTS = "durian_classifier_densenet121_20epochs_batch8_sgd0001.h5"
CLASS_DICT = {
    0: 'D24',
    1: 'JIN FENG',
    2: 'MAO SHAN WANG',
    3: 'RED PRAWN'
}


@st.cache(allow_output_mutation=True)
def load_own_model(weights):
    return load_model(weights)


def load_img(input_image, shape):
    """Load and resize image

    Arguments:
        image {.jpg, .jpeg, .png} -- Image in the relevant format

    Returns:
        resized_img -- Tensored image
    """
    img = Image.open(input_image).convert('RGB')
    img = img.resize((shape, shape))
    img = image.img_to_array(img)
    return np.reshape(img, [1, shape, shape, 3])/255


if __name__ == "__main__":
    st.image("assets/unsplash-durian.jpg", use_column_width=True)
    "# Ah Chong's Durian Classifier"

    "### Oi 老板, tell me what durian you want to jiak"

    result = st.empty()
    uploaded_img = st.file_uploader(
        label='eh what your durian looks like ah:')
    if uploaded_img:
        st.image(uploaded_img, caption="your sexy durian pic",
                 width=350)
        result.info("eh wait ah 我在 inspect 你的 liu lian..., ")
        model = load_own_model(PATH + WEIGHTS)
        pred_img = load_img(uploaded_img, 224)
        pred = CLASS_DICT[np.argmax(model.predict(pred_img))]

        result.success("Wah swee la, is my favourite " + pred + "!!!")
