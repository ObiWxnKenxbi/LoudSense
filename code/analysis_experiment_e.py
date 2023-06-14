import os
import csv
from utilities import analyze_audio
import pandas as pd

#VoS = Voice on Set
#SS = Source Separation
#D = Demucs

folder_path = "/../../audios/experiment_e"
vad_param = 3
hop_length_param = 10
csv_file_name = "/../..results/results_exp_e.csv"

with open(csv_file_name, mode='w', newline='') as csv_file:
    fieldnames = ['name of the clip', 'language', 'music', 'speech', 'speech_to_music_ratio']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader() 
    for filename in os.listdir(folder_path):
      if filename.endswith(".wav"):
          file_path = os.path.join(folder_path, filename)
          (pyln_we_avg_loudness, speech_avg_loudness, speech_to_music_ratio) = analyze_audio(file_path, vad_param, hop_length_param)
          if filename.startswith("1"):
              language = "SPA"
          elif filename.startswith("2"):
              language = "ENG"
          print("chosen audio:", file_path)
          print("Average Weighted Loudness (Speech):", speech_avg_loudness )
          print("Average Weighted Loudness (Non Speech):", pyln_we_avg_loudness)
          print("Speech/Music ratio:", speech_to_music_ratio)
          print("Language:", language)
          print("--------------")
          writer.writerow({'name of the clip': filename, 'language': language, 'speech' : pyln_we_avg_loudness, 'music': speech_avg_loudness, 'speech_to_music_ratio': speech_to_music_ratio})

#Difference between SS and D
csv_file_name = "/../../results/results_exp_e.csv"
df = pd.read_csv(csv_file_name)

df['SS-Music vs D-Music'] = df['SS-Music'] - df['D-Music']
df['SS-Speech vs D-Speech'] = df['SS-Speech'] - df['D-Speech']
df['SS-SMR vs D-SMR'] = df['SS-SMR'] - df['D-SMR']

df = df.drop(columns=["SS-Music","D-Music","SS-Speech", "D-Speech", "SS-SMR", "D-SMR", "VoS-Music", "VoS-Speech", "VoS-SMR"])

df.head()

#Difference between SS and VoS
df['SS-Music vs D-Music'] = df['SS-Music'] - df['VoS-Music']
df['SS-Speech vs D-Speech'] = df['SS-Speech'] - df['VoS-Speech']
df['SS-SMR vs D-SMR'] = df['SS-SMR'] - df['VoS-SMR']

df = df.drop(columns=["SS-Music","D-Music","SS-Speech", "D-Speech", "SS-SMR", "D-SMR", "VoS-Music", "VoS-Speech", "VoS-SMR"])

df.head()

#Difference between D and VoS
df['SS-Music vs D-Music'] = df['D-Music'] - df['VoS-Music']
df['SS-Speech vs D-Speech'] = df['D-Speech'] - df['VoS-Speech']
df['SS-SMR vs D-SMR'] = df['D-SMR'] - df['VoS-SMR']

df = df.drop(columns=["SS-Music","D-Music","SS-Speech", "D-Speech", "SS-SMR", "D-SMR", "VoS-Music", "VoS-Speech", "VoS-SMR"])

df.head()