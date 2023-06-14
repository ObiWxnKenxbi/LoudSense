#this is the integrated loudness of the whole audios to compare with RX8
import csv 
import os 
import pandas as pd
import soundfile as sf
import pyloudnorm as pyln

audio_folder_path = '/../../output/non_speech'

for filename in os.listdir(audio_folder_path):
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        filepath = os.path.join(audio_folder_path, filename)
        data, rate = sf.read(filepath) 
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        print(f"The integrated loudness of {filename} is {loudness:.2f} LUFS")

file_path = os.path.join("/../../results/audio_values_loudness.csv", "audio_values_loudness.csv") 

with open(file_path, mode="w") as csv_file:
    fieldnames = ["filename", "loudness"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for filename in os.listdir(audio_folder_path):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            filepath = os.path.join(audio_folder_path, filename)
            data, rate = sf.read(filepath)
            meter = pyln.Meter(rate)
            loudness = meter.integrated_loudness(data) 

            writer.writerow({"filename": filename, "loudness": loudness})

#this is the difference between pyln and RX8. I think we cna trust pyln since the median difference is -0.05

df = pd.read_csv("/../../results/audio_values_loudness.csv")
diff = df["pyln"] - df["RX8"]
for i, d in enumerate(diff):
  print(f"Difference for video {i+1}: {d:.2f}")
median_diff = diff.median()
print("-------- --------")
print(f"The median difference between pyln and RX8 is {median_diff:.2f}.")
