import cv2
import torch
import os
import csv
import time
import statistics
import nltk
from collections import Counter
from paths import *  # Ensure this is correctly set
from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

nltk.download('punkt')
nltk.download('stopwords')

interval = 1  # Capture frame every 1 second

# Function to extract frames from a video at intervals
def extract_frames(video_path, interval=interval):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = interval * fps  # Capture every 'interval' seconds
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0  # In seconds

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append((frame_count, frame))  # Store frame number and frame
        frame_count += 1

    cap.release()
    return frames, total_frames, video_duration

# Define candidate captions
candidate_captions = [
    "A person walking in a park",
    "A car driving down the street",
    "A dog playing with a ball",
    "A group of people having a conversation",
    "A man working on a computer",
    "A woman holding a phone",
    "A child riding a bicycle",
    "A cat sitting on a windowsill",
    "A bus stopping at a station",
    "A person eating at a restaurant"
]

# Function to generate captions using CLIP similarity scoring
def generate_caption(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Tokenize text and preprocess image
    inputs = processor(text=candidate_captions, images=image_pil, return_tensors="pt", padding=True).to(device)
    
    # Compute similarity scores
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        best_caption_idx = logits_per_image.argmax().item()
    
    return candidate_captions[best_caption_idx]

# Initialize summarizer before timing
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Function to summarize captions
def summarize_captions(captions):
    start_summary_time = time.time()
    summary = summarizer(" ".join(captions), max_length=100, min_length=15, do_sample=False)[0]["summary_text"]
    end_summary_time = time.time()
    return summary, end_summary_time - start_summary_time

# Define CLIP output directory
CLIP_OUTPUT_DIR = os.path.join(VIDEO_PATH, "Output", "CLIP")  
os.makedirs(CLIP_OUTPUT_DIR, exist_ok=True)

# Process videos 01.mp4 to 10.mp4
for i in range(1, 11):
    video_filename = f"{i:02d}.mp4"
    video_path = os.path.join(VIDEO_PATH, video_filename)
    if not os.path.exists(video_path):
        print(f"Video file {video_filename} not found. Skipping...")
        continue

    video_output_dir = os.path.join(CLIP_OUTPUT_DIR, video_filename[:-4])
    os.makedirs(video_output_dir, exist_ok=True)

    print(f"\nProcessing {video_filename}...\n")

    frames, total_frames, video_duration = extract_frames(video_path, interval)
    captions, csv_data, frame_times = [], [], []
    total_start_time = time.time()

    for frame_number, frame in frames:
        start_time = time.time()
        caption = generate_caption(frame)
        end_time = time.time()

        captions.append(caption)
        frame_times.append(end_time - start_time)
        csv_data.append([frame_number, f"frame_{frame_number}.jpg", caption, end_time - start_time])

    mean_time = statistics.mean(frame_times)
    median_time = statistics.median(frame_times)
    mode_time = statistics.mode(frame_times) if len(frame_times) > 1 else frame_times[0]

    final_summary, summary_time = summarize_captions(captions)
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    csv_filename = os.path.join(video_output_dir, f"{video_filename[:-4]}_captions.csv")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Number", "Image File", "Caption", "Processing Time (s)"])
        writer.writerows(csv_data)
        writer.writerow([])
        writer.writerow(["Video Stats"])
        writer.writerow(["Total Frames", total_frames])
        writer.writerow(["Video Duration (s)", video_duration])
        writer.writerow(["Mean Processing Time (s)", mean_time])
        writer.writerow(["Median Processing Time (s)", median_time])
        writer.writerow(["Mode Processing Time (s)", mode_time])
        writer.writerow(["Summary Processing Time (s)", summary_time])
        writer.writerow(["Total Processing Time (s)", total_processing_time])
        writer.writerow([])
        writer.writerow(["Summary", final_summary])

    print("\n=== Final Video Summary ===")
    print(final_summary)
    print("\n=== Processing Time Stats ===")
    print(f"Total Frames: {total_frames}")
    print(f"Video Duration: {video_duration:.2f} seconds")
    print(f"Mean Frame Processing Time: {mean_time:.4f} seconds")
    print(f"Median Frame Processing Time: {median_time:.4f} seconds")
    print(f"Mode Frame Processing Time: {mode_time:.4f} seconds")
    print(f"Summary Processing Time: {summary_time:.4f} seconds")
    print(f"Total Processing Time: {total_processing_time:.4f} seconds")
