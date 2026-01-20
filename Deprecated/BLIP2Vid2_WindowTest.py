# Full version of BLIP2Vid2.py with Hierarchical Windows integration

import cv2
import torch
import os
import csv
import time
import statistics
import nltk
from collections import Counter, deque
from paths import *  # Ensure this is correctly set
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === ANSI Color Codes ===
DARK_GREEN = "\033[38;2;0;128;0m"
CYAN = "\033[38;2;0;255;255m"
YELLOW = "\033[38;2;255;255;0m"
RED = "\033[38;2;255;0;0m"
MAGENTA = "\033[38;2;255;0;255m"
RESET = "\033[0m"

# === Global Settings ===
VIDEO_SELECTION_MODE = "single"  # Options: "all", "single"
SELECTED_VIDEO_INDEX = 3  # Only used if VIDEO_SELECTION_MODE == "single"
DETAIL_MODE = True  # Print captions for each frame in single mode
SUMMARY_MODEL = "tinyllama"  # Options: "bart", "gpt2", "llama", "mistral", "tinyllama", "nous"

# === Prompt Engineering (for GPT2/LLaMA transitions) ===
PROMPT_STYLE = (
    "\n\nDescribe the visual change between the two scenes below "
    "in one concise sentence. \n\nPrevious: {prev}\nCurrent: {curr}\nTransition:"
)

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

nltk.download('punkt')
nltk.download('stopwords')

interval = 1  # Capture frame every 1 second

# === Hierarchical Windows ===
scene_window = deque(maxlen=20)
context_window = deque(maxlen=5)
transition_window = deque(maxlen=2)
noise_window = deque(maxlen=5)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_sim(a, b):
    emb_a = embed_model.encode([a])[0]
    emb_b = embed_model.encode([b])[0]
    return cosine_similarity([emb_a], [emb_b])[0][0]

SCENE_UPDATE_THRESHOLD = 0.7
NOISE_THRESHOLD = 0.5


def load_summarizer():
    global tokenizer, model
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    def load_quantized_model(model_id):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto"
        )
        return tokenizer, model

    if SUMMARY_MODEL == "bart":
        return pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

    elif SUMMARY_MODEL == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    elif SUMMARY_MODEL == "llama":
        from huggingface_hub import login
        token_path = os.path.join(os.path.dirname(__file__), "hf_access_token.txt")
        if not os.path.exists(token_path):
            raise FileNotFoundError("Hugging Face access token file not found at hf_access_token.txt")
        with open(token_path, "r") as f:
            token = f.read().strip()
        login(token)
        tokenizer, model = load_quantized_model("meta-llama/Llama-2-7b-hf")

    elif SUMMARY_MODEL == "mistral":
        tokenizer, model = load_quantized_model("mistralai/Mistral-7B-Instruct-v0.1")

    elif SUMMARY_MODEL == "tinyllama":
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)

    elif SUMMARY_MODEL == "nous":
        tokenizer, model = load_quantized_model("NousResearch/Llama-2-7b-chat-hf")

    else:
        raise ValueError("Unsupported SUMMARY_MODEL")

    def summarize_transformer(text):
        prompt = (
            "Summarize the following scene descriptions from a video.\n"
            "Only include what is explicitly described. Do not add people, objects, or locations that are not mentioned.\n\n"
            "Captions:\n" + text + "\n\nSummary:"
        )
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(model.device)
        attention_mask = torch.ones_like(inputs)
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summarize_transformer

summarizer = load_summarizer()

def extract_frames(video_path, interval=interval):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = interval * fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append((frame_count, frame))
        frame_count += 1

    cap.release()
    return frames, total_frames, video_duration

def generate_caption(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    generated_ids = blip_model.generate(pixel_values=inputs["pixel_values"], max_new_tokens=60)
    return processor.decode(generated_ids[0], skip_special_tokens=True)

def summarize_captions(captions):
    start_summary_time = time.time()
    joined = "\n".join(captions)
    if SUMMARY_MODEL == "bart":
        summary = summarizer(joined, max_length=100, min_length=15, do_sample=False)[0]["summary_text"]
    else:
        summary = summarizer(joined)
    end_summary_time = time.time()
    return summary, end_summary_time - start_summary_time

def summarize_transition(prev_caption, curr_caption):
    if SUMMARY_MODEL not in ["gpt2", "llama"]:
        raise ValueError("Transition summarization is only supported for gpt2 or llama.")
    prompt = PROMPT_STYLE.format(prev=prev_caption, curr=curr_caption)
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(device)
    attention_mask = torch.ones_like(inputs)
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=60,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Transition:" in decoded:
        transition = decoded.split("Transition:")[-1].strip()
    else:
        transition = decoded.strip()
    transition = transition.split(".")[0].strip() + "."
    return transition

BLIP_OUTPUT_DIR = os.path.join(VIDEO_PATH, "Output", "BLIP")
os.makedirs(BLIP_OUTPUT_DIR, exist_ok=True)

video_indices = range(1, 11) if VIDEO_SELECTION_MODE == "all" else [SELECTED_VIDEO_INDEX]

for i in video_indices:
    video_filename = f"{i:02d}.mp4"
    video_path = os.path.join(VIDEO_PATH, video_filename)
    if not os.path.exists(video_path):
        print(f"{RED}Video file {video_filename} not found. Skipping...{RESET}")
        continue

    video_output_dir = os.path.join(BLIP_OUTPUT_DIR, video_filename[:-4])
    os.makedirs(video_output_dir, exist_ok=True)

    summary_output_dir = os.path.join(video_output_dir, "Summaries", SUMMARY_MODEL)
    os.makedirs(summary_output_dir, exist_ok=True)

    print(f"{CYAN}\nProcessing {video_filename}...{RESET}\n")

    frames, total_frames, video_duration = extract_frames(video_path, interval)
    captions, csv_data, frame_times = [], [], []
    total_start_time = time.time()
    transitions = []

    for frame_number, frame in frames:
        start_time = time.time()
        caption = generate_caption(frame)
        end_time = time.time()

        captions.append(caption)
        frame_times.append(end_time - start_time)
        csv_data.append([frame_number, f"frame_{frame_number}.jpg", caption, end_time - start_time])

        if VIDEO_SELECTION_MODE == "single" and DETAIL_MODE:
            print(f"{DARK_GREEN}Frame {frame_number}:{RESET} {caption}")

        # === Hierarchical Window Logic ===
        caption_sim_scene = max([cosine_sim(caption, sc) for sc in scene_window], default=0.0)
        caption_sim_context = max([cosine_sim(caption, cc) for cc in context_window], default=0.0)

        if caption_sim_scene < SCENE_UPDATE_THRESHOLD:
            print(f"{YELLOW}Scene shift detected:{RESET} {caption}")
            scene_window.clear()
        scene_window.append(caption)
        context_window.append(caption)

        if caption_sim_scene < NOISE_THRESHOLD and caption_sim_context < NOISE_THRESHOLD:
            noise_window.append(caption)
            print(f"{RED}Noisy frame detected:{RESET} {caption}")

        transition_window.append(caption)
        if len(transition_window) == 2 and SUMMARY_MODEL in ["gpt2", "llama"]:
            prev_caption, curr_caption = transition_window
            transition = summarize_transition(prev_caption, curr_caption)
            transitions.append((prev_caption, curr_caption, transition))
            print(f"{YELLOW}Transition:{RESET} {transition}")

        if len(noise_window) == noise_window.maxlen:
            noise_common = Counter(noise_window).most_common(1)[0]
            if noise_common[1] > 3:
                print(f"{MAGENTA}Noise override using:{RESET} {noise_common[0]}")
                context_window.append(noise_common[0])

    mean_time = statistics.mean(frame_times)
    median_time = statistics.median(frame_times)
    mode_time = statistics.mode(frame_times) if len(frame_times) > 1 else frame_times[0]

    if transitions and SUMMARY_MODEL in ["gpt2", "llama"]:
        transition_sentences = sorted(set(t[2] for t in transitions))
        final_summary = " ".join(transition_sentences)
        summary_time = 0.0
    else:
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

    summary_filename = os.path.join(summary_output_dir, f"{video_filename[:-4]}_summary.csv")
    with open(summary_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Summary Model", SUMMARY_MODEL])
        writer.writerow(["Total Captions", len(captions)])
        writer.writerow(["Summary Text", final_summary])
        writer.writerow(["Summary Processing Time (s)", summary_time])
        writer.writerow(["Total Video Processing Time (s)", total_processing_time])

    captions_txt_path = os.path.join(summary_output_dir, f"{video_filename[:-4]}_captions.txt")
    with open(captions_txt_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(captions))

    if transitions:
        transitions_csv_path = os.path.join(summary_output_dir, f"{video_filename[:-4]}_transitions.csv")
        with open(transitions_csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Previous Caption", "Current Caption", "Transition Summary"])
            writer.writerows(transitions)

    print(f"\n{MAGENTA}=== All Unique Transitions ==={RESET}")
    for unique_transition in sorted(set(t[2] for t in transitions)):
        print(f"- {unique_transition}")

    print(f"\n{MAGENTA}=== Final Video Summary ==={RESET}")
    print(f"{final_summary}")

    print(f"\n{MAGENTA}=== Processing Time Stats ==={RESET}")
    print(f"{CYAN}Total Frames:{RESET} {total_frames}")
    print(f"{CYAN}Video Duration:{RESET} {video_duration:.2f} seconds")
    print(f"{CYAN}Mean Frame Processing Time:{RESET} {mean_time:.4f} seconds")
    print(f"{CYAN}Median Frame Processing Time:{RESET} {median_time:.4f} seconds")
    print(f"{CYAN}Mode Frame Processing Time:{RESET} {mode_time:.4f} seconds")
    print(f"{CYAN}Summary Processing Time:{RESET} {summary_time:.4f} seconds")
    print(f"{CYAN}Total Processing Time:{RESET} {total_processing_time:.4f} seconds")
