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

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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


def i3d_actions_indv(video_tensor):
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_classes = probabilities.topk(1)
    
    actions = [f"{class_names[cls.item()]} ({prob:.2f})" for cls, prob in zip(top_classes[0], top_probs[0])]
    return actions

def blip_i3d_combo_story(descriptions):
    prompt = (
    "Summarize the video based on these individual observations and actions but only return one summary."
    +" Each observation is a moment in time, combine them."+ 
    "Do not repeat the observations, combine them into a single summary. \n\n"
    + "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(descriptions))
    + "\n\n \n\nStory:"
    )


    inputs = tokenizer(prompt,
                       return_tensors="pt",
                       truncation=True,
                       max_length=2048,
                       padding=True)
    inputs['attention_mask'] = inputs['attention_mask']
    with torch.no_grad():
        outputs = gpt_model.generate(inputs['input_ids'], 
                                    attention_mask=inputs['attention_mask'],
                                    max_length=1024, 
                                    num_return_sequences=1, 
                                    temperature=0.8,         # Controls creativity
                                    top_p=0.9,               # Nucleus sampling for more natural outputs
                                    do_sample=True )

    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Combo Story:\n {story}")
    return story


def get_blip_indv_embedding(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #otherwise the colors get confused
    inputs = blip_processor(images=frame_rgb, return_tensors="pt")
    with torch.no_grad(): #dont want to waste space and time on gradients
        outputs = blip_model.generate(**inputs) #gets descriptions
        caption = blip_processor.batch_decode(outputs, skip_special_tokens=True)[0] #converts to readable text
        #print(f"Generated Blip caption Indv: {caption}")
    return caption

def vid_preprocess(video_path, output_dir, seq_frames=16):
    video = cv2.VideoCapture(video_path) #video capture from path
    i3d_frames = []
    i3d_frames_batch = []
    blip_frames = []
    blip_i3d_descriptions = []
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

    #frames for blip
    '''frames_per_sec = int(video.get(cv2.CAP_PROP_FPS)) #retreives frames per sec of vid
    frame_interval = max(1, frames_per_sec//frame_rate) #how many frames we should get'''
    #i3d needs a set number of frames per seq. I wanted Blip to match for consistency

    #note: seq frames is for i3d only, this is the number of frames to use in seq
    while video.isOpened(): #keep open for all frames
        ret, frame = video.read()
        if not ret: #if it couldnt read the frame
            break
        #blip frames
        if interval_count % seq_frames == 0: #1 frame captured per seq len
            frame_path = os.path.join(output_dir, f"frame_{interval_count}.jpg")
            cv2.imwrite(frame_path, frame) #writes frame to path
            blip_caption = get_blip_indv_embedding(frame)
            blip_frames.append(frame)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #change from bgr to rgb color
        frame = transform(frame) #transforms using the above trans func
        i3d_frames.append(frame) #collect frames for i3d

        if len(i3d_frames) == seq_frames:
            #for all
            frame_tensor = torch.stack(i3d_frames) #.unsqueeze(0) #unsqueeze adds batch dim after stacking tensors
            
            #for i3d/blip indv combo
            i3d_video_tensor = torch.stack(i3d_frames).unsqueeze(0).permute(0, 2, 1, 3, 4)
            i3d_actions = i3d_actions_indv(i3d_video_tensor)
            combined_text = f"{blip_caption}. {', '.join(i3d_actions)}."
            #print(f"Combo text: \n {combined_text}")
            blip_i3d_descriptions.append(combined_text)

            #for Batch 
            i3d_frames_batch.append(frame_tensor)
            i3d_frames = []

        interval_count+=1
    video.release()
    print("Running Combo story now....")
    #run combo descriptions
    blip_i3d_combo_story(blip_i3d_descriptions)
    print("Returning batch story now....")
    #was of dim ( frames, channels, h, w)
    i3d_video_tensor = torch.stack(i3d_frames_batch).permute(0, 2, 1, 3, 4)# reorders to (batch, channels, frames, h, w)
    return i3d_video_tensor, blip_frames
    

def predict_actions(video_tensor):
    print(f"shape: {video_tensor.shape}")
    with torch.no_grad(): #no gradients bc we arent trying to train so we can save time and mem
        outputs = model(video_tensor) #i3d model returns classes and values
        probabilities = torch.nn.functional.softmax(outputs, dim=1) #probabililties, dim 1 is classes
        top_probability, top_classes = probabilities.topk(10) #top 10
    return top_probability, top_classes




            
def get_blip_embeddings(frames):
    embeddings = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #otherwise the colors get confused
        inputs = blip_processor(images=frame_rgb, return_tensors="pt")
        with torch.no_grad(): #dont want to waste space and time on gradients
            outputs = blip_model.generate(**inputs) #gets descriptions
            caption = blip_processor.batch_decode(outputs, skip_special_tokens=True)[0] #converts to readable text
            #print(f"Generated Blip caption: {caption}")
            
            inputs = tokenizer(caption, return_tensors="pt")
            embedding = gpt_model.transformer.wte(inputs.input_ids).mean(dim=1)
            embeddings.append(embedding)
        if not embeddings:
            raise RuntimeError("No embeddings generated.")
    return torch.cat(embeddings) if embeddings else torch.tensor([])

#Gen description
def i3d_description(video_path, output_dir):
    i3d_video_tensor, blip_video_frames = vid_preprocess(video_path, output_dir) #returns tensor for i3d, frames list for blip
    if i3d_video_tensor is None:
        return "Failed to process video."

    blip_embeddings = get_blip_embeddings(blip_video_frames)
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

