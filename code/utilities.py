import subprocess
import webrtcvad
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pydub import AudioSegment
import wave
import statistics
import pyvad
import pandas as pd
import matplotlib.pyplot as plt

#The following code defines a function called vad that performs Voice Activity Detection (VAD) on an audio signal. 
# It takes the audio signal data, the sampling frequency fs, and optional parameters such as fs_vad (the VAD algorithm's sampling frequency), 
# hop_length (the frame hop length in milliseconds), and vad_mode (the VAD mode) as inputs. 
# The function performs various checks on the input parameters and converts the data type if necessary. 
# It then resamples the data if the sampling frequencies are different, applies padding and framing to the resampled data, and initializes the WebRTC VAD algorithm. 
# The VAD algorithm is applied to each frame of the audio signal, and the resulting binary speech/non-speech flags are post-processed using a convolution and thresholding operation.
# Finally, the function reshapes and returns the VAD flags corresponding to the original input data size.

def vad(data, fs, fs_vad=16000, hop_length=30, vad_mode=0):
    """
    Args:
    - data: 1D array of the audio signal to be analyzed
    - fs: the sampling frequency of the audio signal
    - fs_vad (optional): the sampling frequency to be used by the VAD algorithm (default is 16000)
    - hop_length (optional): the hop length (in milliseconds) of the frames used in the VAD analysis (default is 30)
    - vad_mode (optional): the VAD mode (0, 1, 2, or 3) to be used by the WebRTC VAD algorithm (default is 0)

    Returns:
    - va_framed: a 2D binary array indicating whether each frame contains speech (1) or not (0) 
    """

    if fs_vad not in [8000, 16000, 32000, 48000]:
        raise ValueError("8000, 16000, 32000 or 48000.")
    if hop_length not in [10, 20, 30]:
        raise ValueError("10, 20, or 30.")
    if vad_mode not in [0, 1, 2, 3]:
        raise ValueError("0, 1, 2 or 3.")
    if data.dtype.kind == "i":
        if data.max() > 2 ** 15 - 1 or data.min() < -(2 ** 15):
            raise ValueError(
                "When data.type is int, data must be -32768 < data < 32767."
            )
        data = data.astype("f") / 2.0 ** 15
    elif data.dtype.kind == "f":
        if np.abs(data).max() > 1:
            raise ValueError(
                "When data.type is float, data must be -1.0 <= data <= 1.0."
            )
        data = data.astype("f")
    else:
        raise ValueError("data.dtype must be int or float.")
    data = data.squeeze()
    if not data.ndim == 1:
        raise ValueError("data must be mono (1 ch).")
    if fs != fs_vad:
        resampled = resample(data, orig_sr=fs, target_sr=fs_vad)
        if np.abs(resampled).max() > 1.0:
            resampled *= 0.99 / np.abs(resampled).max()
            warn("Resampling causes data clipping. data was rescaled.")
    else:
        resampled = data
    resampled = (resampled * 2.0 ** 15).astype("int16")
    hop = fs_vad * hop_length // 1000
    framelen = resampled.size // hop + 1
    padlen = framelen * hop - resampled.size
    paded = np.lib.pad(resampled, (0, padlen), "constant", constant_values=0)
    framed = frame(paded, frame_length=hop, hop_length=hop).T
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)
    valist = [vad.is_speech(tmp.tobytes(), fs_vad) for tmp in framed]
    valist = np.asarray(valist).astype("float")
    valist = np.convolve(valist, np.ones(3) / 3, mode="same") > 0
    hop_origin = fs * hop_length // 1000
    va_framed = np.zeros([len(valist), hop_origin])
    va_framed[valist] = 0
    return va_framed.reshape(-1)[: data.size]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#The code defines a function called calculate_loudness that calculates the integrated loudness of an audio segment. 
# It takes a file path, start time, and end time as inputs. The function reads the audio data from the specified file, 
# selects the desired segment, and uses the pyln.Meter from the pyloudnorm library to calculate the integrated loudness. 
# Additionally, it constructs an FFmpeg command to calculate the loudness using the ebur128 filter and executes it using subprocess.Popen. 
# The output is captured and the loudness value is extracted and returned as pyln_loudness. 
# There is an optional step that extracts the loudness value using a different method and assigns it to ffmpeg_loudness. 
# Finally, the function returns the calculated integrated loudness values as a tuple.

def calculate_loudness_NOTUSED(file_path, start_time, end_time):
      """
      Args:
      - file_path: the path to the audio file
      - start_time: the start time (in seconds) of the audio segment to be analyzed
      - end_time: the end time (in seconds) of the audio segment to be analyzed

      Returns:
      - pyln_loudness: the integrated loudness (in LUFS) of the audio segment, calculated using pyln.Meter

      """
      data, rate = sf.read(file_path)
      start_index = int(start_time * rate)
      end_index = int(end_time * rate)
      data = data[start_index:end_index]
      meter = pyln.Meter(rate, block_size=0.004)
      pyln_loudness = meter.integrated_loudness(data)
      command = f'/usr/bin/ffmpeg -nostats -i {file_path} -filter_complex ebur128=peak=true -f null -'

      #comands used that dont work
      #command = "ffmpeg -i {file_path} -af ebur128=framelog=verbose -f null - 2>&1 | awk '/I:/{print $2}"
      #command = "ffmpeg -i {file_path} -hide_banner -filter_complex ebur128 -f null - 2<&1 | grep -E 'I:.+ LUFS$' | tail -1 | grep -E '\-[0-9\.]+"
      #command = "ffmpeg -i {file_path} -hide_banner -filter_complex ebur128 -f null -"
      #command = "ffmpeg -nostats -i {} -filter_complex ebur128=peak=true -f null -".format(file_path)
      #command = ['ffmpeg', '-nostats', '-i', file_path, '-filter_complex', '[0:a]ebur128=framelog=verbose:peak=0.0', '-f', 'null', '-']
      #command = ['ffmpeg', '-nostats', '-i', file_path, '-filter_complex', 'ebur128=peak=true', '-f', 'null', '-']
      #command = f'ffmpeg -nostats -i {file_path} -filter_complex ebur128=peak=true -f null - 2>&1'
      #command = f'/usr/bin/ffmpeg -nostats -i {file_path} -filter_complex ebur128=peak=true -f null -'

      proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      output, _ = proc.communicate()
      output = output.decode('utf-8')
      lufs_index = output.find('I:  ')
      if lufs_index == -1:
          raise Exception('FFmpeg failed to calculate loudness')
      lufs_str = None
      try:
          output = subprocess.check_output(
              command,
              stderr=subprocess.STDOUT,
              shell=True
          ).decode()
          lufs_index = output.index('I:')
          lufs_str = output[lufs_index+4:lufs_index+10].strip()
      except:
          pass
      if lufs_str is not None and lufs_str != '':
          ffmpeg_loudness = int(lufs_str)
      else:
          ffmpeg_loudness = None
      return pyln_loudness, ffmpeg_loudness

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#The code defines a function called calculate_loudness that calculates the integrated loudness (in LUFS) of an audio segment. 
# It takes three inputs: file_path (the path to the audio file), start_time (the start time of the audio segment in seconds), 
# and end_time (the end time of the audio segment in seconds). 
# The function reads the audio file using sf.read and extracts the desired segment based on the start and end times. 
# It then initializes a loudness meter using pyln.Meter with the given sample rate and a block size of 0.00004. 
# The integrated_loudness method is called on the segment data using the meter, and the result is stored in the variable pyln_loudness. 
# Finally, the function returns the calculated loudness value as pyln_loudness and None for the second value.

def calculate_loudness(file_path, start_time, end_time):
      """
      Args:
      - file_path: the path to the audio file
      - start_time: the start time (in seconds) of the audio segment to be analyzed
      - end_time: the end time (in seconds) of the audio segment to be analyzed

      Returns:
      - pyln_loudness: the integrated loudness (in LUFS) of the audio segment, calculated using pyln.Meter
      """
      data, rate = sf.read(file_path)
      start_index = int(start_time * rate)
      end_index = int(end_time * rate)
      data = data[start_index:end_index]
      meter = pyln.Meter(rate, block_size=0.00004)
      pyln_loudness = meter.integrated_loudness(data)
      return pyln_loudness, None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def analyze_audio(file_path, vad_param, hop_length_param):
  """
  Args:
  - file_path: the path to the audio file
  - vad_param: the VAD mode (0, 1, 2, or 3) to be used by the WebRTC VAD algorithm (default is 0)
  - hop_length: the hop length (in milliseconds) of the frames used in the VAD analysis (must be either 10, 20 or 20)

  Returns:
  - pyln_we_avg_loudness: the weighted loudness (in LUFS) of the non-speech segments, calculated using pyln.Meter
  """

  #general analysis
  webrtc_vad = webrtcvad.Vad()
  webrtc_vad.set_mode(3)
  with wave.open(file_path, 'rb') as wf:
      sample_rate = wf.getframerate()
      num_channels = wf.getnchannels()
      audio_data = wf.readframes(wf.getnframes())
  audio_array = np.frombuffer(audio_data, dtype=np.int16)
  if num_channels > 1:
      audio_array = audio_array[::num_channels]
  audio_array = audio_array / np.abs(audio_array).max()
  time = np.linspace(0, len(audio_array) / sample_rate, len(audio_array))
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
  vact = pyvad.vad(data, fs=target_sr, fs_vad=48000, hop_length=hop_length_param, vad_mode=vad_param)
  time = np.arange(len(data)) / target_sr

  #plotting (rn its commented cuz i dont wanna get the spectrogram for the selected)
#  fig, (ax1, ax3) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
#  ax1.plot(time, data, label='speech waveform')
#  ax1.set_ylabel("Amplitude")
#  ax1.set_xlim([time[0], time[-1]])
#  ax2 = ax1.twinx()
#  ax2.plot(time, vact, color="r", label='vad')
#  ax2.set_ylim([-0.01, 1.01])
#  ax2.set_yticks([0, 1])
#  ax2.set_ylabel("VAD")
#  Pxx, freqs, bins, im = ax3.specgram(data, Fs=target_sr, NFFT=256, noverlap=128, cmap='magma')
#  ax3.set_xlabel("Time [s]")
#  ax3.set_ylabel("Frequency [Hz]")
#  vad_overlay = np.ma.masked_where(vact == 1, vact)
#  vad_overlay = np.expand_dims(vad_overlay, axis=0) 
#  vad_overlay = np.repeat(vad_overlay, len(freqs), axis=0) 
#  ax3.imshow(vad_overlay, cmap='Reds', origin='lower', aspect='auto', alpha=0.5, extent=[time[0], time[-1], freqs[0], freqs[-1]])
#  plt.tight_layout()
#  plt.subplots_adjust(hspace=0.05)
#  ax1.legend(loc='upper left')
#  ax2.legend(loc='upper right')
#  plt.show()

  #start / end times segmentation
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

  #pyln analysis of the non-speech and speech segments
  pyln_loudness_values = []
  ffmpeg_loudness_values = []
  speech_loudness_values = []

  for i, (start, end) in enumerate(zip(start_times, end_times)):
      start_time_sec = start / sample_rate
      end_time_sec = end / sample_rate
      pyln_loudness, ffmpeg_loudness = calculate_loudness(file_path, start_time_sec, end_time_sec)

      if pyln_loudness is None:
          pyln_message = "Unable to calculate loudness with pyln"
      else:
          pyln_message = f"{pyln_loudness:.2f}"
          pyln_loudness_values.append(pyln_loudness)

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

  if len(speech_loudness_values) > 0:
      speech_avg_loudness = sum(speech_loudness_values) / len(speech_loudness_values)
  else:
      print("No speech segments found")
      
  total_duration = sum(speech_durations)
  pyln_we_avg_loudness = []

  #statistics
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
      speech_to_music_ratio = speech_avg_loudness - pyln_we_avg_loudness

  return pyln_we_avg_loudness, speech_avg_loudness, speech_to_music_ratio

#This code defines a function called analyze_audio that takes three parameters: file_path (the path to an audio file), vad_param (the voice activity detection mode), and hop_length_param (the hop length in milliseconds).
#The function performs various operations on the audio file. 
# It uses the WebRTC VAD algorithm to analyze the audio and extract non-speech segments based on the specified VAD mode. 
# It then calculates the weighted loudness of the non-speech segments using the pyln.Meter library.
# After the analysis, the function plots the speech waveform, VAD activity, and spectrogram of the audio. 
# It also performs additional loudness calculations using the calculate_loudness function for different time intervals within the audio.
# Finally, the function computes various statistics and returns the weighted average loudness (pyln_we_avg_loudness) of the non-speech segments. 
# The function also prints a message indicating the audibility level based on the loudness value.