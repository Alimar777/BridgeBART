import cv2
import torch
import os
import csv
import time
import statistics
import numpy as np
import nltk
import glob
from collections import Counter
from paths import *  # Ensure this is correctly set
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from transformers import BlipForQuestionAnswering
from PIL import Image
import matplotlib.pyplot as plt
import shutil

# Global variables for mode configuration
MODE = "qa"  # Options: "caption" or "qa"
QUESTION = "What is shown in this image?"  # Default question for QA mode

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Salesforce/blip-image-captioning-base" if MODE == "caption" else "Salesforce/blip-vqa-base"

print(f"Using model: {model_name} in {MODE} mode")
processor = BlipProcessor.from_pretrained(model_name)

if MODE == "caption":
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
elif MODE == "qa":
    model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
else:
    exit()

nltk.download('punkt')
nltk.download('stopwords')

# Function to clean up old attention files
def cleanup_attention_files(directory):
    """Remove all existing attention jpg files in the directory"""
    attention_files = glob.glob(os.path.join(directory, "*_attention.jpg"))
    if attention_files:
        print(f"Cleaning up {len(attention_files)} old attention files...")
        for file in attention_files:
            os.remove(file)

# Function to extract every frame from a video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_duration = total_frames / fps if fps > 0 else 0  # In seconds

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((frame_count, frame))  # Store frame number and frame
        frame_count += 1

    cap.release()
    return frames, total_frames, video_duration

# Function to compute attention rollout map
def get_attention_rollout_map(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    
    outputs = model.vision_model(**inputs, output_attentions=True)
    attentions = outputs.attentions
    
    att_mat = [att.mean(dim=1)[0] for att in attentions]
    
    rollout = None
    for a in att_mat:
        a = a + torch.eye(a.size(0)).to(a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a if rollout is None else torch.matmul(rollout, a)
    
    cls_attention = rollout[0, 1:]
    return cls_attention.detach().cpu().numpy()

# Function to visualize attention on image
def visualize_attention(image, attention_map, save_path):
    h, w, _ = image.shape
    grid_size = int(np.sqrt(len(attention_map)))
    
    attention_map = attention_map.reshape(grid_size, grid_size)
    attention_map = cv2.resize(attention_map, (w, h))
    
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map) + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(save_path, overlayed_image)

# Function to generate captions or answers and attention heatmaps
def process_frame(image, save_attention_path):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if MODE == 'caption':
        # Caption mode
        inputs = processor(images=image_pil, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=60)
        result = processor.decode(output_ids[0], skip_special_tokens=True)
    else:
        # QA mode
        inputs = processor(image_pil, QUESTION, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=50)
        result = processor.decode(output_ids[0], skip_special_tokens=True)
    
    attention_map = get_attention_rollout_map(image)
    visualize_attention(image, attention_map, save_attention_path)

    return result

# Function to create an attention visualization video
def create_attention_video(image_folder, output_video_path, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith("_attention.jpg")]
    images.sort(key=lambda x: int(x.split('_')[1]))
    
    if not images:
        print("No attention images found to create video.")
        return
    
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_image.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Attention video saved at: {output_video_path}")

# Process videos 01.mp4 to 10.mp4
output_folder = "BLIP_Caption" if MODE == 'caption' else "BLIP_QA"
BLIP_OUTPUT_DIR = os.path.join(VIDEO_PATH, "Output", output_folder)
os.makedirs(BLIP_OUTPUT_DIR, exist_ok=True)

for i in range(1, 11):
    video_filename = f"{i:02d}.mp4"
    video_path = os.path.join(VIDEO_PATH, video_filename)
    if not os.path.exists(video_path):
        print(f"Video file {video_filename} not found. Skipping...")
        continue

    video_output_dir = os.path.join(BLIP_OUTPUT_DIR, video_filename[:-4])
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Clean up any existing attention files before processing
    cleanup_attention_files(video_output_dir)

    print(f"\nProcessing {video_filename} in {MODE} mode...\n")
    if MODE == 'qa':
        print(f"Question: {QUESTION}")

    frames, total_frames, video_duration = extract_frames(video_path)
    results, csv_data, frame_times = [], [], []
    total_start_time = time.time()

    for frame_number, frame in frames:
        start_time = time.time()
        
        attention_save_path = os.path.join(video_output_dir, f"frame_{frame_number}_attention.jpg")
        result = process_frame(frame, attention_save_path)
        
        end_time = time.time()

        results.append(result)
        frame_times.append(end_time - start_time)
        
        # Add the mode-specific column name
        if MODE == 'caption':
            csv_data.append([frame_number, result, end_time - start_time])
        else:
            csv_data.append([frame_number, QUESTION, result, end_time - start_time])

    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    # Save results to CSV
    csv_filename = os.path.join(video_output_dir, f"{video_filename[:-4]}_{MODE}.csv")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if MODE == 'caption':
            writer.writerow(["Frame Number", "Caption", "Processing Time (s)"])
        else:
            writer.writerow(["Frame Number", "Question", "Answer", "Processing Time (s)"])
        writer.writerows(csv_data)
        writer.writerow(["Total Processing Time (s)", total_processing_time])

    print(f"Processing complete for {video_filename}.")

    # Generate summary statistics
    avg_time = statistics.mean(frame_times)
    fps_processing = 1 / avg_time if avg_time > 0 else 0
    
    print(f"Average processing time per frame: {avg_time:.4f} seconds")
    print(f"Processing rate: {fps_processing:.2f} frames per second")
    
    # Generate word frequency if in caption mode
    if MODE == 'caption':
        all_words = []
        for result in results:
            words = nltk.word_tokenize(result.lower())
            stopwords = nltk.corpus.stopwords.words('english')
            filtered_words = [word for word in words if word.isalnum() and word not in stopwords]
            all_words.extend(filtered_words)
        
        word_freq = Counter(all_words).most_common(10)
        print("\nMost common words in captions:")
        for word, count in word_freq:
            print(f"{word}: {count}")

# Clean up old attention videos before creating new ones
for i in range(1, 11):
    video_filename = f"{i:02d}.mp4"
    video_output_dir = os.path.join(BLIP_OUTPUT_DIR, video_filename[:-4])
    attention_video_path = os.path.join(video_output_dir, f"{video_filename[:-4]}_{MODE}_attention.mp4")
    
    # Remove old attention video if it exists
    if os.path.exists(attention_video_path):
        os.remove(attention_video_path)
        print(f"Removed old attention video: {attention_video_path}")
        
    create_attention_video(video_output_dir, attention_video_path, fps=30)