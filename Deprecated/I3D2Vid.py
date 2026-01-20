import cv2
import torch
import os
import csv
import time
import statistics
import numpy as np
from collections import Counter
from paths import *  # Ensure this is correctly set
from pytorchvideo.models.hub import i3d_r50
from torchvision.transforms import Compose, Lambda, Resize, CenterCrop, Normalize, ToTensor
from transformers import pipeline
from PIL import Image

# Define path for I3D output
I3D_OUTPUT_DIR = os.path.join(VIDEO_PATH, "Output", "I3D")  
os.makedirs(I3D_OUTPUT_DIR, exist_ok=True)

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load I3D model
model = i3d_r50(pretrained=True).to(device)
model.eval()

# Kinetics 400 class labels for mapping predictions
with open("kinetics_400_labels.txt", "r") as f:
    kinetics_labels = [line.strip() for line in f.readlines()]

# If you don't have the labels file, you can use this placeholder approach:
# This is a snippet of the labels - in practice you'd want the full list
if not os.path.exists("kinetics_400_labels.txt"):
    print("Warning: kinetics_400_labels.txt not found. Using placeholder labels.")
    kinetics_labels = [
        "abseiling", "air drumming", "answering questions", "applauding", "archery", 
        "arm wrestling", "arranging flowers", "assembling computer", "auctioning",
        "baby waking up", "baking cookies", "balloon blowing", "bandaging", 
        # ... and so on for all 400 classes
    ]
    # If none of the above, we'll just use class indices

# Define video clip duration in seconds
clip_duration = 8  # I3D typically works with 8-16 second clips

# Transform for I3D input
transform = Compose([
    Lambda(lambda x: x / 255.0),
    Resize((256, 256)),
    CenterCrop((224, 224)),
    Lambda(lambda x: x.permute(3, 0, 1, 2)),  # [T, H, W, C] -> [C, T, H, W]
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
])

# Initialize summarizer for action summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Function to extract clips from a video
def extract_clips(video_path, clip_duration=clip_duration, overlap=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0  # In seconds
    
    frames_per_clip = int(clip_duration * fps)
    stride = int(frames_per_clip * (1 - overlap))
    
    clips = []
    clip_start_frames = []
    
    frame_count = 0
    current_clip = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if len(current_clip) < frames_per_clip:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_clip.append(frame_rgb)
            
        if len(current_clip) == frames_per_clip:
            clips.append(np.array(current_clip))
            clip_start_frames.append(frame_count - frames_per_clip + 1)
            
            # For overlapping clips, remove stride frames from the beginning
            if stride < frames_per_clip:
                current_clip = current_clip[stride:]
            else:
                current_clip = []
                
        frame_count += 1
    
    cap.release()
    return clips, clip_start_frames, total_frames, video_duration, fps

# Function to process clips with I3D
def analyze_clip(clip):
    # Convert clip to tensor and apply transformations
    clip_tensor = torch.tensor(clip).float()
    clip_tensor = transform(clip_tensor)
    clip_tensor = clip_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Get model prediction
    with torch.no_grad():
        output = model(clip_tensor)
        
    # Get top predictions
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_probs, top_indices = torch.topk(probabilities, k=5)
    
    results = []
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
        label = kinetics_labels[idx] if idx < len(kinetics_labels) else f"class_{idx}"
        results.append((label, prob.item()))
    
    return results

# Function to summarize action descriptions
def summarize_actions(actions):
    start_summary_time = time.time()
    action_text = " ".join([f"{action} ({confidence:.2f})" for action, confidence in actions])
    summary = summarizer(action_text, max_length=100, min_length=15, do_sample=False)[0]["summary_text"]
    end_summary_time = time.time()
    return summary, end_summary_time - start_summary_time

# Process videos 01.mp4 to 10.mp4
for i in range(1, 11):
    video_filename = f"{i:02d}.mp4"
    video_path = os.path.join(VIDEO_PATH, video_filename)
    if not os.path.exists(video_path):
        print(f"Video file {video_filename} not found. Skipping...")
        continue

    video_output_dir = os.path.join(I3D_OUTPUT_DIR, video_filename[:-4])
    os.makedirs(video_output_dir, exist_ok=True)

    print(f"\nProcessing {video_filename}...\n")

    # Extract clips
    clips, clip_start_frames, total_frames, video_duration, fps = extract_clips(video_path)
    
    if not clips:
        print(f"No clips extracted from {video_filename}. Skipping...")
        continue
    
    csv_data = []
    clip_times = []
    all_actions = []
    total_start_time = time.time()

    # Process each clip
    for clip_idx, (clip, start_frame) in enumerate(zip(clips, clip_start_frames)):
        start_time = time.time()
        
        # Analyze clip with I3D
        actions = analyze_clip(clip)
        all_actions.extend([action for action, _ in actions])
        
        end_time = time.time()
        processing_time = end_time - start_time
        clip_times.append(processing_time)
        
        # Calculate timestamp
        start_sec = start_frame / fps
        end_sec = (start_frame + len(clip)) / fps
        
        # Save top actions to CSV
        for rank, (action, confidence) in enumerate(actions):
            csv_data.append([
                clip_idx, 
                f"{start_sec:.2f}-{end_sec:.2f}", 
                f"{action} ({confidence:.4f})", 
                rank + 1,
                processing_time
            ])
    
    # Calculate time statistics
    mean_time = statistics.mean(clip_times)
    median_time = statistics.median(clip_times)
    mode_time = statistics.mode(clip_times) if len(clip_times) > 1 else clip_times[0]
    
    # Get most common actions
    action_counter = Counter(all_actions)
    most_common_actions = action_counter.most_common(10)
    
    # Generate summary
    action_summary = [(action, count/len(all_actions)) for action, count in most_common_actions]
    final_summary, summary_time = summarize_actions(action_summary)
    
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    # Save results to CSV
    csv_filename = os.path.join(video_output_dir, f"{video_filename[:-4]}_actions.csv")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Clip Number", "Time Range (s)", "Action (Confidence)", "Rank", "Processing Time (s)"])
        writer.writerows(csv_data)
        writer.writerow([])
        writer.writerow(["Video Stats"])
        writer.writerow(["Total Frames", total_frames])
        writer.writerow(["Video Duration (s)", video_duration])
        writer.writerow(["FPS", fps])
        writer.writerow(["Clips Processed", len(clips)])
        writer.writerow(["Mean Processing Time (s)", mean_time])
        writer.writerow(["Median Processing Time (s)", median_time])
        writer.writerow(["Mode Processing Time (s)", mode_time])
        writer.writerow(["Summary Processing Time (s)", summary_time])
        writer.writerow(["Total Processing Time (s)", total_processing_time])
        writer.writerow([])
        writer.writerow(["Most Common Actions"])
        for action, count in most_common_actions:
            writer.writerow([action, count, f"{count/len(all_actions):.4f}"])
        writer.writerow([])
        writer.writerow(["Summary", final_summary])

    print("\n=== Video Activity Summary ===")
    print(final_summary)
    print("\n=== Most Common Actions ===")
    for action, count in most_common_actions:
        print(f"{action}: {count} occurrences ({count/len(all_actions):.2%})")
    print("\n=== Processing Time Stats ===")
    print(f"Total Frames: {total_frames}")
    print(f"Video Duration: {video_duration:.2f} seconds")
    print(f"Clips Processed: {len(clips)}")
    print(f"Mean Clip Processing Time: {mean_time:.4f} seconds")
    print(f"Median Clip Processing Time: {median_time:.4f} seconds")
    print(f"Mode Clip Processing Time: {mode_time:.4f} seconds")
    print(f"Summary Processing Time: {summary_time:.4f} seconds")
    print(f"Total Processing Time: {total_processing_time:.4f} seconds")