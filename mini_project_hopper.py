import streamlit as st
import os
import numpy as np
from PIL import Image
import networkx as nx
import torch
import torchvision.transforms as T
from transformers import AutoFeatureExtractor, AutoModel

# Set up your flower images directory path
FLOWER_IMAGES_DIR = 'flower_images'

# Initialize the new model and feature extractor
model_ckpt = "jafdxc/vit-base-patch16-224-finetuned-flower"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to("cuda" if torch.cuda.is_available() else "cpu")

# Data transformation chain
transformation_chain = T.Compose([
    T.Resize(int((256 / 224) * extractor.size["height"])),
    T.CenterCrop(extractor.size["height"]),
    T.ToTensor(),
    T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
])

# Function to extract the label from a filename (assuming your filenames contain labels)
def extract_label_from_filename(filename):
    label = filename.split('_', 1)[1].rsplit('.', 1)[0]
    label = label.replace('_', ' ')
    return label

# Function to get the path of an image given a flower name
def get_image_path(flower_name):
    for filename in os.listdir(FLOWER_IMAGES_DIR):
        if extract_label_from_filename(filename).lower() == flower_name.lower():
            return os.path.join(FLOWER_IMAGES_DIR, filename)
    return None

# Embedding extraction function
def extract_embeddings(image_path):
    image = Image.open(image_path)
    transformed_image = transformation_chain(image).unsqueeze(0).to(model.device)
    with torch.no_grad():
        output = model(transformed_image).last_hidden_state[:, 0].cpu().numpy()
    return output

# Function to compute the distance matrix
def compute_distance_matrix(embeddings):
    num_images = len(embeddings)
    distance_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(i, num_images):
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # Symmetric matrix
    return distance_matrix

def find_shortest_path(distance_matrix, start_index, end_index):
    # Create a graph
    G = nx.Graph()

    # Add edges to the graph based on the distance matrix
    for i in range(len(distance_matrix)):
        for j in range(i, len(distance_matrix[i])):
            if i != j and distance_matrix[i][j] != 0:  # Assuming 0 indicates no direct path
                G.add_edge(i, j, weight=distance_matrix[i][j])

    # Find valid paths with exactly two intermediate steps
    valid_paths = []
    for mid1 in G.neighbors(start_index):
        for mid2 in G.neighbors(mid1):
            if mid2 != start_index and G.has_edge(mid2, end_index):
                valid_paths.append([start_index, mid1, mid2, end_index])

    # Ensure there are valid paths
    if not valid_paths:
        return None

    # Choose the path with the minimum total distance
    shortest_path = min(valid_paths, key=lambda path: sum(distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)))

    return shortest_path

# Display function for Streamlit
def display_image_grid(image_paths):
    cols = st.columns(3)
    for idx, image_path in enumerate(image_paths):
        col = cols[idx % 3]
        image = Image.open(image_path)
        flower_name = extract_label_from_filename(image_path)
        col.image(image, caption=flower_name, use_column_width=True)

# Main function with Streamlit logic
def main():
    st.title("Flower Hopper")
    st.subheader("Enter the names of two flowers to find the shortest path")

    flower_a = st.text_input("Flower A").strip().title()
    flower_b = st.text_input("Flower B").strip().title()

    if flower_a and flower_b:
        flower_a_path = get_image_path(flower_a)
        flower_b_path = get_image_path(flower_b)
        
        if not flower_a_path or not flower_b_path:
            st.error("One or both flower images not found. Please check the names and try again.")
            return

        # Generate a list of all image paths in the directory
        all_image_paths = [os.path.join(FLOWER_IMAGES_DIR, filename) for filename in sorted(os.listdir(FLOWER_IMAGES_DIR))]
        
        # Compute the embeddings for all images
        embeddings = [extract_embeddings(path) for path in all_image_paths]

        # Compute the distance matrix
        distance_matrix = compute_distance_matrix(embeddings)

        # Get indices for start and end nodes
        start_index = all_image_paths.index(flower_a_path)
        end_index = all_image_paths.index(flower_b_path)

        # Find the shortest path using the Dijkstra algorithm
        shortest_path_indices = find_shortest_path(distance_matrix, start_index, end_index)

        # Convert indices back to image paths
        shortest_path_image_paths = [all_image_paths[idx] for idx in shortest_path_indices]

        # Display the image path as a grid
        display_image_grid(shortest_path_image_paths)

if __name__ == "__main__":
    main()
