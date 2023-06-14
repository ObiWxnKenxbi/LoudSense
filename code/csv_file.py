import os
import csv
from utilities import analyze_audio

csv_file_name = "/../../results/msr_results.csv"
folder_path = "/../../audios"

with open(csv_file_name, mode='w', newline='') as csv_file:
    #fieldnames = ['name of the clip', 'language', 'non speech loudness', 'speech loudness', 'speech_to_music_ratio']
    #writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #writer.writeheader() 
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
          #writer.writerow({'name of the clip': filename, 'language': language, 'speech loudness' : pyln_we_avg_loudness, 'non speech loudness': speech_avg_loudness, 'speech_to_music_ratio': speech_to_music_ratio})