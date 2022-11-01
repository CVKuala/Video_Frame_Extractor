import os
import cv2
import clip
import torch
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR100
from sklearn.metrics.pairwise import cosine_similarity

import initial_list
from CLiP import image_classifier, video_frame_extractor
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--type', type=str, help=['image_classifier','video_frame_extractor'])
parser.add_argument('--path', type=str, default='input/project.avi', help='path of image or video')
parser.add_argument('--text_prompt', type=str, default="man walking in plain", help='text prompt only for video_frame_extractor')
parser.add_argument('--similarity_thresh', type=float, default=0.97, help='threshold for cosine similarity')
parser.add_argument('--output_path', type=str, default='input/', help='output for saved images according to prompt')
parser.add_argument('--top_k', type=int, default=5, help='top k prompts for each image')

args = parser.parse_args()

if __name__ == "__main__":
    
    if(args.type == 'image_classifier'):
        image_classifier(initial_list.class_list, args.path, args.top_k)
    elif(args.type == 'video_frame_extractor'):
        video_frame_extractor(initial_list.class_list, args.path, args.text_prompt, args.similarity_thresh, args.output_path, args.top_k)