from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline

# Model selector
TRANSITION_MODEL_NAME = "gpt2"  # Options: "llama", "gpt2", "bart"

def load_transition_model():
    if TRANSITION_MODEL_NAME == "llama":
        return pipeline("text2text-generation", model="meta-llama/Llama-2-7b-hf")
    elif TRANSITION_MODEL_NAME == "gpt2":
        return pipeline("text-generation", model="gpt2")
    elif TRANSITION_MODEL_NAME == "bart":
        return pipeline("text2text-generation", model="facebook/bart-large-cnn")
    else:
        raise ValueError("Unsupported transition model.")

class SceneWindow:
    def __init__(self, max_size=20, similarity_threshold=0.7):
        self.captions = []
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

    def is_similar(self, caption):
        if not self.captions:
            return True
        vectorizer = TfidfVectorizer().fit(self.captions + [caption])
        vectors = vectorizer.transform(self.captions + [caption])
        sim_scores = cosine_similarity(vectors[-1], vectors[:-1])
        return np.mean(sim_scores) > self.similarity_threshold

    def add(self, caption):
        self.captions.append(caption)
        if len(self.captions) > self.max_size:
            self.captions.pop(0)

    def reset_with(self, caption):
        self.captions = [caption]

    def get_keywords(self):
        return " ".join(self.captions)

    def get_summary(self):
        return " ".join(self.captions[:5])


class ContextWindow:
    def __init__(self, max_size=5):
        self.captions = deque(maxlen=max_size)

    def add(self, caption):
        self.captions.append(caption)

    def reset(self):
        self.captions.clear()

    def last(self):
        return self.captions[-1] if self.captions else None


class TransitionWindow:
    def __init__(self):
        self.previous = None
        self.current = None

    def set_previous(self, caption):
        self.previous = caption

    def set_current(self, caption):
        self.current = caption

    def get_transition_pair(self):
        return (self.previous, self.current)

    def describe_transition(self, model):
        if self.previous and self.current:
            prompt = f"Describe the visual change from: '{self.previous}' to '{self.current}'"
            if TRANSITION_MODEL_NAME == "gpt2":
                prompt += "\n"
                response = model(prompt, max_length=60, do_sample=True)[0]['generated_text']
            else:
                response = model(prompt, max_length=60, do_sample=False)[0]['generated_text']
            return response
        return None


class NoiseWindow:
    def __init__(self, max_size=10, novelty_threshold=0.5):
        self.captions = deque(maxlen=max_size)
        self.novelty_threshold = novelty_threshold

    def add(self, caption):
        self.captions.append(caption)

    def should_filter(self, caption, reference_text):
        vectorizer = TfidfVectorizer().fit([reference_text, caption])
        vectors = vectorizer.transform([reference_text, caption])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return similarity < self.novelty_threshold

    def get_common_caption(self):
        return max(set(self.captions), key=self.captions.count) if self.captions else None


class RollingWindowManager:
    def __init__(self):
        self.scene = SceneWindow()
        self.context = ContextWindow()
        self.transition = TransitionWindow()
        self.noise = NoiseWindow()

    def update(self, caption, frame_index):
        if self.scene.is_similar(caption):
            self.scene.add(caption)
            self.context.add(caption)
            self.transition.set_previous(caption)
        elif self.noise.should_filter(caption, self.scene.get_keywords()):
            self.noise.add(caption)
        else:
            self.transition.set_current(caption)
            self.scene.reset_with(caption)
            self.context.reset()

    def get_transition_text(self):
        return self.transition.get_transition_pair()

    def describe_transition(self, model):
        return self.transition.describe_transition(model)

    def get_scene_summary(self):
        return self.scene.get_summary()

    def get_common_noise(self):
        return self.noise.get_common_caption()
