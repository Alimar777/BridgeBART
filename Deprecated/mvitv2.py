import torch
import torchvision.models.video as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import torch
import torchvision.transforms as T
import cv2
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

'''This is working i3d model'''

model = models.mvit_v2_s(weights="KINETICS400_V1")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
gpt_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")

tokenizer.pad_token = tokenizer.eos_token #specifies the type of padding for gpt (end of seq)

#pulls class labels
def load_kinetics_classes():
    import requests
    url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
    response = requests.get(url)
    class_names = response.text.strip().split("\n")
    class_names = [line.split(", ") for line in class_names]
    return class_names

class_names = load_kinetics_classes() #loads i3d class names from online

def vid_preprocess(video_path, output_dir, seq_frames=16):
    video = cv2.VideoCapture(video_path) #video capture from path
    i3d_frames = []
    i3d_frames_batch = []
    transform = transforms.Compose([ #to make sure its the correct size
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    interval_count = 0
    if not video.isOpened():
        print(f"Failed to open video file: {video_path}")
        return
    print(f"Successfully opened video file: {video_path}")

    while video.isOpened(): #keep open for all frames
        ret, frame = video.read()
        if not ret: #if it couldnt read the frame
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #change from bgr to rgb color
        frame = transform(frame) #transforms using the above trans func
        i3d_frames.append(frame) #collect frames for i3d

        if len(i3d_frames) == seq_frames:
            frame_tensor = torch.stack(i3d_frames) #.unsqueeze(0) #unsqueeze adds batch dim after stacking tensors
            i3d_frames_batch.append(frame_tensor)
            i3d_frames = []
        interval_count+=1
    video.release()
    i3d_video_tensor = torch.stack(i3d_frames_batch).permute(0, 2, 1, 3, 4)# reorders to (batch, channels, frames, h, w)
    return i3d_video_tensor

def predict_actions(video_tensor):
    print(f"shape: {video_tensor.shape}")
    with torch.no_grad(): #no gradients bc we arent trying to train so we can save time and mem
        outputs = model(video_tensor) #i3d model returns classes and values
        probabilities = torch.nn.functional.softmax(outputs, dim=1) #probabililties, dim 1 is classes
        top_probability, top_classes = probabilities.topk(10) #top 10
    return top_probability, top_classes

#Gen description
def i3d_description(video_path, output_dir):
    i3d_video_tensor = vid_preprocess(video_path, output_dir) #returns tensor for i3d, frames list for blip
    if i3d_video_tensor is None:
        return "Failed to process video."
    probabilities, classes = predict_actions(i3d_video_tensor) #gets the class indices and probabilities
    
    #cls is a pytorch obj, classes[] is our tensor of class indices, probabilities[] is our tensor list of probabilities
    actions = [f"{class_names[cls.item()]}: {probability:.4f}" for cls, probability in zip(classes[0], probabilities[0])]
    actions = [action.replace("['", "").replace("']", "") for action in actions]
    return "The video shows: " + ", ".join(actions) + "."

current_dir = os.getcwd()
current_dir2 = os.path.join(current_dir, "Documents", "UNM2025")
video_filename = "testVid3.mp4"
video_path = os.path.join(current_dir2, video_filename)
output_dir = os.path.join(current_dir2, "frames")
os.makedirs(output_dir, exist_ok=True)
description = i3d_description(video_path, output_dir)

print(f"description: {description}")
