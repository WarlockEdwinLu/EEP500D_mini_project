# take classify image through streamlit page
import streamlit as st
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

st.title("Flower Identifier")
st.subheader("Please upload a flower image")
uploaded_file = st.file_uploader("Upload a Picture", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image.save("input.png")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    directory_path = 'flower_images/'
    file_list = os.listdir(directory_path)

    def extract_label_from_filename(filename):
        # Remove the file extension (.png) and leading index (e.g., "0_")
        label = filename.split('_', 1)[1].rsplit('.', 1)[0]
        # Replace underscores with spaces to get the proper flower name
        label = label.replace('_', ' ')
        return label

    # Function to load an image from a file path and preprocess it
    def load_preprocess_image(image_path, processor):
        image = Image.open(image_path)
        return processor(images=image, return_tensors="pt").pixel_values

    # Function to extract features using CLIP
    def extract_clip_features(image_path, model, processor):
        image_tensor = load_preprocess_image(image_path, processor)
        with torch.no_grad():
            features = model.get_image_features(image_tensor).cpu().numpy().flatten()
        return features

    # Function to extract the label from the filename
    def extract_label_from_filename(filename):
        label = filename.split('_', 1)[1].rsplit('.', 1)[0]
        label = label.replace('_', ' ')
        return label

    # Function to predict the class of a new image and return nearest neighbors
    def predict_image_class(image_path, knn, model, processor, nearest_neighbors, image_paths, labels):
        new_image_features = extract_clip_features(image_path, model, processor)
        prediction = knn.predict([new_image_features])
        predicted_label = prediction[0]

        # Get the distances and indices of the 4 nearest neighbors
        distances, indices = nearest_neighbors.kneighbors([new_image_features], n_neighbors=4)

        # Retrieve the paths and labels for the nearest images, excluding those with the same label
        nearest_images_paths = []
        nearest_images_labels = []
        for idx in indices.flatten():
            if labels[idx] != predicted_label:
                nearest_images_paths.append(image_paths[idx])
                nearest_images_labels.append(labels[idx])
            if len(nearest_images_paths) == 3:
                break

        return predicted_label, nearest_images_paths, nearest_images_labels

    # Function to display images given a list of image paths
    def display_images(image_paths):
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path)
            plt.subplot(1, len(image_paths), i + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()

    def display_images_streamlit(image_paths, image_labels):
        for i, (img_path, label) in enumerate(zip(image_paths, image_labels)):
            image = Image.open(img_path)
            st.image(image, caption=f"Image {i+1}: {label}", width=300)

    # Initialize lists to hold the feature vectors, labels, and image paths
    feature_list = []
    labels = []
    image_paths = []  # This will store the paths of the images

    # Extract features, labels, and store image paths for each image
    for filename in file_list:
        if filename.endswith('.png'):
            image_path = os.path.join(directory_path, filename)
            features = extract_clip_features(image_path, model, processor)
            feature_list.append(features)
            labels.append(extract_label_from_filename(filename))
            image_paths.append(image_path)  # Store the image path

    # Convert feature list and labels list to numpy arrays
    X = np.array(feature_list)
    y = np.array(labels)

    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Initialize the NearestNeighbors
    nearest_neighbors = NearestNeighbors(n_neighbors=3)
    nearest_neighbors.fit(X)

    # Path to the new image you want to classify
    new_image_path = 'input.png'

    # Predict the class of the new image and get nearest neighbors
    predicted_class, nearest_images, nearest_labels = predict_image_class(new_image_path, knn, model, processor, nearest_neighbors, image_paths, labels)
    #print("Predicted class:", predicted_class)
    st.write("Predicted class:", predicted_class)

    # Display or process nearest_images as needed
    #for img_path in nearest_images:
    #    print("Nearest image:", img_path)

    # Display nearest images
    display_images_streamlit(nearest_images, nearest_labels)

else:
    st.write("Waiting for an image to be uploaded...")