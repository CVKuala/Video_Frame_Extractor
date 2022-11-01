import os
import cv2
import clip
import torch
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR100
from sklearn.metrics.pairwise import cosine_similarity

# This function used to classify images based on CIFAR100 classes.
def image_classifier(li, image_path, top_k):

    print("The available models are {}".format(clip.available_models()))
    try:
        # Load the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('RN50', device)

        # Prepare the inputs
        image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in li]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(top_k)

        # Print the result
        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            print(f"{li[index]:>16s}: {100 * value.item():.2f}%")
    
    except Exception as e:
        print(e)
        
def video_frame_extractor(li, video_path, text_prompt, similarity_thresh, out_path, top_k):
    
    print("The available models are {}".format(clip.available_models()))
    
    # Checking whether the text prompt by user is in list or not
    if text_prompt not in li:
        li.append(text_prompt)
    
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device)
    
    # This part is similar to the previous function where the clip model is used to tokenize all the classes in the list. 
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in li]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Creating an empty vector to use for the initial cosine similarity.
    vidOBJ=cv2.VideoCapture(video_path)
    success, image = vidOBJ.read()
    base_vector = np.zeros(image.reshape(1,-1).shape)
    
    vidOBJ=cv2.VideoCapture(video_path)
    count=0
    success=True
    
    while success==True:
        
        success, image = vidOBJ.read()
        if success == False:
            break
            
        image_input = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        
        # This part is similar to the previous the function where the model is used to encode each image.
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        
        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(top_k)
        
        # Here we dont want every image to be saved, rather we use cosine similarity to determine similar images.
        if(text_prompt in [li[x] for x in indices] and cosine_similarity(base_vector, image.reshape(1,-1))<similarity_thresh):
            cv2.imwrite(out_path + '/' + "frame{}.jpg".format(count), image)
            base_vector=image.reshape(1,-1)
            print(count)
        
        count=count+1