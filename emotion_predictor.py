import pandas as pd
import nltk
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
from datetime import datetime, timedelta

# Download required NLTK resources
nltk.download("punkt")

# Step 1: Simulated User Data (Replace with Real Data Collection)
user_data = [
    {"date": "2025-01-20", "text": "Achieving my goal after months of hard work is so fulfilling."},
    {"date": "2025-01-22", "text": "No matter how hard I try, nothing seems to get better."},
    {"date": "2025-01-25", "text": "It's so frustrating when people dont respect my boundaries."},
    {"date": "2025-01-27", "text": "Feeling grateful for the little things in life."},
    {"date": "2025-01-28", "text": "My heart races every time I hear that strange noise at night."},
    {"date": "2025-01-29", "text": "Your kindness makes my world a better place."},
    {"date": "2025-01-30", "text": "I'm so excited to start this new chapter in my life."},
    {"date": "2025-02-01", "text": "I feel so lonely and isolated from everyone."},
    {"date": "2025-02-03", "text": "The thought of failing makes me so anxious."},
    {"date": "2025-02-04", "text": "This is the most unexpected gift I’ve ever received."},
    {"date": "2025-02-05", "text": "I can't stop smiling after watching that hilarious movie."},
    {"date": "2025-01-28", "text": "I never expected to see you here today!"}
]

df = pd.DataFrame(user_data)
df["date"] = pd.to_datetime(df["date"])

# Step 2: Load Sentiment & Emotion Analysis Model
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Step 3: Perform Emotion Analysis
def analyze_emotion(text):
    result = emotion_pipeline(text)[0][0]
    return result["label"], result["score"]

df["emotion"], df["confidence"] = zip(*df["text"].apply(analyze_emotion))

# Step 4: Plot Emotional Trends Over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="date", y="confidence", hue="emotion", marker="o")
plt.title("Emotional Evolution Over Time")
plt.xlabel("Date")
plt.ylabel("Emotion Confidence")
plt.legend(title="Emotion")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()