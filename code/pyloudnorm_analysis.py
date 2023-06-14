from utilities import vad, calculate_loudness
import pandas as pd
import os
import wave
import webrtcvad
import statistics
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import pandas as pd
import pyloudnorm as pyln
from pyloudnorm import Meter
from pydub import AudioSegment
import subprocess
import shlex

audio_folder_path = '/../../audios'

file_names = os.listdir(audio_folder_path)
wav_file_names = [f for f in file_names if f.endswith('.wav')]
for file_name in wav_file_names:
    print(f"Processing file: {file_name}")

#csv creation
columns = ['filename', 'num_channels', 'average_loudness', 'non_speech_weighted_loudness_average', 'speech_weighted_loudness_average', 'margin_of_difference', 'mean_loudness', 'median_loudness', 'min_loudness', 'max_loudness', 'nonspeech_audio_length']
df = pd.DataFrame(columns=columns)

hop_length_param = 10 #Choose either 10, 20 or 30
vad_param = 3 #Choose 1, 2 or 3. 3 being the hardest.

output_folder = "/../../output/non_speech"
file_paths = [os.path.join(audio_folder_path, file_name) for file_name in os.listdir(audio_folder_path)]
audio_df = pd.DataFrame(columns=["filename", "duration", "sampling_rate"])

audio_df = pd.DataFrame(columns=["filename", "duration", "sampling_rate"])
for filename in os.listdir(audio_folder_path):
    if filename.endswith(".wav"):
        filepath = os.path.join(audio_folder_path, filename)
        audio_data, sr = librosa.load(filepath, sr=None)
        duration = librosa.get_duration(audio_data, sr=sr)
        audio_df = audio_df.append({"filename": filename, "duration": duration, "sampling_rate": sr}, ignore_index=True)

for file_path in file_paths:
  vad = webrtcvad.Vad()
  vad.set_mode(3)
  with wave.open(file_path, 'rb') as wf:
      sample_rate = wf.getframerate()
      num_channels = wf.getnchannels()
      audio_data = wf.readframes(wf.getnframes())
  audio_array = np.frombuffer(audio_data, dtype=np.int16)
  if num_channels > 1:
      audio_array = audio_array[::num_channels]
  audio_array = audio_array / np.abs(audio_array).max()
  time = np.linspace(0, len(audio_array) / sample_rate, len(audio_array))
  print("audio to analyze: ", os.path.basename(file_path))
  print("number of channels: ", num_channels)
  print("sample rate ", sample_rate)
  from pyvad import vad, trim, split
  audio_file = AudioSegment.from_file(file_path)
  target_sr = 16000
  if sample_rate != target_sr:
      audio_file = audio_file.set_frame_rate(target_sr)
  if num_channels > 1:
      audio_file = audio_file.set_channels(1)
  audio_file = audio_file.set_sample_width(2)
  data = np.array(audio_file.get_array_of_samples())
  data = data.astype(np.float32)
  data /= 32768.0
  vact = vad(data, fs=target_sr, fs_vad=48000, hop_length=hop_length_param, vad_mode=vad_param)
  time = np.arange(len(data)) / target_sr

  fig, (ax1, ax3) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
  ax1.plot(time, data, label='speech waveform')
  ax1.set_ylabel("Amplitude")
  ax1.set_xlim([time[0], time[-1]])
  ax2 = ax1.twinx()
  ax2.plot(time, vact, color="r", label='vad')
  ax2.set_ylim([-0.01, 1.01])
  ax2.set_yticks([0, 1])
  ax2.set_ylabel("VAD")
  Pxx, freqs, bins, im = ax3.specgram(data, Fs=target_sr, NFFT=256, noverlap=128, cmap='magma')
  ax3.set_xlabel("Time [s]")
  ax3.set_ylabel("Frequency [Hz]")
  vad_overlay = np.ma.masked_where(vact == 1, vact)
  vad_overlay = np.expand_dims(vad_overlay, axis=0)
  vad_overlay = np.repeat(vad_overlay, len(freqs), axis=0)
  ax3.imshow(vad_overlay, cmap='Reds', origin='lower', aspect='auto', alpha=0.5, extent=[time[0], time[-1], freqs[0], freqs[-1]])
  plt.tight_layout()
  plt.subplots_adjust(hspace=0.05)
  ax1.legend(loc='upper left')
  ax2.legend(loc='upper right')
  plt.show()

  start_times = []
  end_times = []
  is_active = False
  for i in range(len(vact)):
      if vact[i] == 1 and not is_active:
          start_times.append(i)
          is_active = True
      elif vact[i] == 0 and is_active:
          end_times.append(i)
          is_active = False
  audio_file = AudioSegment.from_file(file_path)
  speech_audio = AudioSegment.empty()
  non_speech_audio = AudioSegment.empty()
  speech_durations = []
  pyln_loudness_values = []

  for i, (start, end) in enumerate(zip(start_times, end_times)):
      start_time_sec = start / sample_rate
      end_time_sec = end / sample_rate
      pyln_loudness, ffmpeg_loudness = calculate_loudness(file_path, start_time_sec, end_time_sec)
      if pyln_loudness is None:
          pyln_message = "Unable to calculate loudness with pyln"
      else:
          pyln_message = f"{pyln_loudness:.2f}"
  pyln_loudness_values = []
  ffmpeg_loudness_values = []
  for start, end in zip(start_times, end_times):
      start_time_sec = start / sample_rate
      end_time_sec = end / sample_rate
      pyln_loudness, ffmpeg_loudness = calculate_loudness(file_path, start_time_sec, end_time_sec)
      if pyln_loudness is None:
          pyln_message = "Unable to calculate loudness with pyln"
      else:
          pyln_message = f"{pyln_loudness:.2f}"
          pyln_loudness_values.append(pyln_loudness)
      print(f"Silence from {start_time_sec:.2f}s to {end_time_sec:.2f}s:, pyln Loudness = {pyln_message}")
  for start, end in zip(start_times, end_times):
      start_time_ms = start * 1000 / sample_rate
      end_time_ms = end * 1000 / sample_rate
      non_speech_segment = audio_file[start_time_ms:end_time_ms]
      non_speech_audio = non_speech_audio + non_speech_segment
      out_file_name = f"non_speech_{i}.wav"
      out_file_path = os.path.join(output_folder, out_file_name)
      # Create a WAV file with the non-speech segment
      # with wave.open(out_file_path, 'w') as out_file:
      #   out_file.setparams((audio_file.channels, audio_file.sample_width, audio_file.frame_rate, len(non_speech_segment), 'NONE', 'not compressed'))
      #   out_file.writeframes(non_speech_segment._data)
  audio_length_s = len(non_speech_audio) / sample_rate
  audio_length_ms = len(non_speech_audio) / sample_rate * 1000
  print(f"Total non-speech audio length: {audio_length_ms:.2f} ms or {audio_length_s:.2f} s")
  speech_audio = np.array([])
  for i, (end, start) in enumerate(zip(end_times[:-1], start_times[1:])):
      end_time_sec = end / sample_rate
      start_time_sec = start / sample_rate
      pyln_loudness, ffmpeg_loudness = calculate_loudness(file_path, end_time_sec, start_time_sec)
      if pyln_loudness is None:
          pyln_message = "Unable to calculate loudness with pyln"
      else:
          pyln_message = f"{pyln_loudness:.2f}"
          pyln_loudness_values.append(pyln_loudness)
      print(f"Speech from {end_time_sec:.2f}s to {start_time_sec:.2f}s:, pyln Loudness = {pyln_message}")
      end_time_ms = end * 1000 / sample_rate
      start_time_ms = start * 1000 / sample_rate
      speech_segment = audio_file[end_time_ms:start_time_ms]
      speech_audio = np.concatenate((speech_audio, np.ravel(speech_segment)))
      out_file_name = f"speech_{i}.wav"
      out_file_path = os.path.join(output_folder, out_file_name)
  non_speech_loudness_values = []
  for i, (start, end) in enumerate(zip(start_times, end_times)):
      start_time_sec = start / sample_rate
      end_time_sec = end / sample_rate
      pyln_loudness, ffmpeg_loudness = calculate_loudness(file_path, start_time_sec, end_time_sec)
      if pyln_loudness is None:
          pyln_message = "Unable to calculate loudness with pyln"
      else:
          pyln_message = f"{pyln_loudness:.2f}"
          non_speech_loudness_values.append(pyln_loudness)
      print(f"Silence from {start_time_sec:.2f}s to {end_time_sec:.2f}s:, pyln Loudness = {pyln_message}")

  if len(non_speech_loudness_values) > 0:
      non_speech_avg_loudness = sum(non_speech_loudness_values) / len(non_speech_loudness_values)
      print(f"Average loudness of non-speech segments: {non_speech_avg_loudness:.2f} LUFS")
  else:
      print("No non-speech segments found")
  speech_loudness_values = []
  for i in range(len(start_times)-1):
      start = end_times[i]
      end = start_times[i+1]
      start_time_sec = start / sample_rate
      end_time_sec = end / sample_rate
      pyln_loudness, ffmpeg_loudness = calculate_loudness(file_path, start_time_sec, end_time_sec)
      if pyln_loudness is None:
          pyln_message = "Unable to calculate loudness with pyln"
      else:
          pyln_message = f"{pyln_loudness:.2f}"
          speech_loudness_values.append(pyln_loudness)
      print(f"Speech from {start_time_sec:.2f}s to {end_time_sec:.2f}s:, pyln Loudness = {pyln_message}")

  if len(speech_loudness_values) > 0:
      speech_avg_loudness = sum(speech_loudness_values) / len(speech_loudness_values)
      print(f"Average loudness of speech segments: {speech_avg_loudness:.2f} LUFS")
  else:
      print("No speech segments found")

  if len(pyln_loudness_values) > 0:
    pyln_avg_loudness = sum(pyln_loudness_values) / len(pyln_loudness_values)
    durations = []

    for start, end in zip(start_times, end_times):
        duration = (end - start) / sample_rate
        durations.append(duration)

    total_duration = sum(durations)
    weights = [duration / total_duration for duration in durations]
    weighted_loudness_values = [loudness * weight for loudness, weight in zip(pyln_loudness_values, weights)]
    pyln_we_avg_loudness = sum(weighted_loudness_values)
    pyln_margin_of_difference = max(pyln_loudness_values) - min(pyln_loudness_values)
    pyln_mean = statistics.mean(pyln_loudness_values)
    pyln_median = statistics.median(pyln_loudness_values)
    pyln_min = min(pyln_loudness_values)
    pyln_max = max(pyln_loudness_values)
  #music_to_speech_ratio = round(pyln_we_avg_loudness / avg_weighted_loudness, 2)
  #print("music_to_speech_ratio", music_to_speech_ratio)
  df = df.append({
      'filename': os.path.basename(file_path),
      'num_channels': num_channels,
      'non_speech_weighted_loudness_average': pyln_we_avg_loudness,
      'speech_weighted_loudness_average' : speech_avg_loudness,
      'average_loudness': weighted_loudness_values,
      'margin_of_difference': pyln_margin_of_difference,
      'mean_loudness': pyln_mean,
      'median_loudness': pyln_median,
      'min_loudness': pyln_min,
      'max_loudness': pyln_max,
      'nonspeech_audio_length': audio_length_ms
  }, ignore_index=True)
df.to_csv('/../../audio_data.csv', index=False)

df.head()

folder_path_output = "/../../output/non_speech"
folder_lufs = []
for file_name in os.listdir(folder_path_output):
    if file_name.endswith(".wav"):
        file_path = os.path.join(folder_path_output, file_name)
        command = 'ffmpeg -i {} -af ebur128=framelog=verbose -f null -'.format(file_path)
        proc = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tmp = proc.stdout.read()
        text = str(tmp)
        ind = text.find('I:')
        lufs = float(text[(ind+11):(ind+16)])
        folder_lufs.append([file_path, lufs])
        print(f"LUFS of {file_name} is {lufs}")

#I believe the issue with FFMPEG lies in the duration of the audio files being too short. 
# In an attempt to address this, I experimented with merging all the non-speech segments from the videos and treating them as a single, larger audio file. 
# However, even with this approach, only one audio file, coincidentally the longest one, yielded any successful analysis.

#On a positive note, I discovered that when it comes to analyzing audio file #3 (the lengthy one), both pyln, RX8, 
# and FFMPEG produced identical results.