 
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import numpy as np
from PIL import Image

 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

 
tf.compat.v1.reset_default_graph()

 
def load_sample_image():
    (train_images, train_labels), _ = cifar10.load_data()
    idx = np.random.randint(0, len(train_images))
    return train_images[idx], train_labels[idx]

 
model = load_model("cifar10_model.h5")

 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

 
def calculate_accuracy():
    (_, _), (test_images, test_labels) = cifar10.load_data()
    test_images = test_images / 255.0  
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    return accuracy * 100  

 
st.title("CIFAR-10 Image Classification")
st.write("This app classifies images from the CIFAR-10 dataset into 10 categories.")

 
accuracy = calculate_accuracy()
st.write(f"**Model Accuracy on Test Data: {accuracy:.2f}%**")

 
if st.button("Generate Random Image"):
     
    image, label = load_sample_image()
    image_normalized = image / 255.0  

 
    image_pil = Image.fromarray(image)

   
    target_size = 96   
    image_resized = image_pil.resize((target_size, target_size), Image.Resampling.NEAREST)

   
    st.image(image_resized, caption="Random Image from CIFAR-10 (128x128)", width=target_size)
 
    image_input = np.expand_dims(image_normalized, axis=0)
    predictions = model.predict(image_input)
    predicted_class = class_names[np.argmax(predictions)]

    
    st.subheader(f"**Predicted Class: {predicted_class}**")
