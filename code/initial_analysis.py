from utilities import analyze_audio

vad_param = 3
hop_length_param = 10

file_path = "/../../audios/10003-es_antena3_20210703_002150.wav"

print("chosen audio:", file_path)
pyln_we_avg_loudness, speech_avg_loudness, non_speech_avg_loudness, speech_to_music_ratio = analyze_audio(file_path, vad_param, hop_length_param)
print("Average Weighted Loudness (Speech):", speech_avg_loudness)
print("Average Weighted Loudness (Non Speech):", non_speech_avg_loudness)
print("Speech/Music ratio:", speech_to_music_ratio)