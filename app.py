import numpy as np

from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    AveragePooling2D,
    Dropout
)
import streamlit as st

CLASS_DICT = {
    0: 'd24',
    1: 'jin feng',
    2: 'mao shan wang',
    3: 'red prawn'
}


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


def load_model(weights, shape):
    model = MobileNetV2(
        input_shape=(shape, shape, 3),
        include_top=False,
        weights=None
    )

    x = model.output
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(4, activation='softmax',
              kernel_regularizer=l2(.0005))(x)

    model = Model(inputs=model.inputs, outputs=x)

    optimizer = SGD(lr=1e-4, momentum=.9)
    model.load_weights(weights)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


"# Ah Beng David's Durian Classifier"

"Oi lao ban, tell me what durian you want to classify"

uploaded_img = st.file_uploader(label='Select a file:')
if uploaded_img:
    st.image(uploaded_img, caption="your sexy durian pic",
             width=350)
    model = load_model(
        "model-weights/durian_classifier_mobilenetv2_20epochs_batch8_sgd0001.h5",
        224
    )
    pred_img = load_img(uploaded_img, 224)
    pred = CLASS_DICT[np.argmax(model.predict(pred_img))]

    "This is obviously a " + pred
