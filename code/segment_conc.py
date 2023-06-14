import webrtcvad
import wave
import numpy as np
import pyvad
from pydub import AudioSegment
import matplotlib.pyplot as plt
import pyloudnorm as pyln


def audio_separation(file_path, output_folder, vad_param=3, hop_length_param=30):
    # general analysis
    webrtc_vad = webrtcvad.Vad()
    webrtc_vad.set_mode(vad_param)
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
    # start / end times segmentation
    speech_start_times = []
    speech_end_times = []
    non_speech_start_times = []
    non_speech_end_times = []
    is_speech = False

    for i in range(len(vact)):
        if vact[i] == 1 and not is_speech:
            speech_start_times.append(i)
            is_speech = True
        elif vact[i] == 0 and is_speech:
            speech_end_times.append(i)
            is_speech = False
        elif vact[i] == 0 and not is_speech:
            non_speech_end_times.append(i)
            is_speech = False
        elif vact[i] == 1 and is_speech:
            non_speech_start_times.append(i)
            is_speech = True

    # Calculate non-speech end and start times
    print("calculating non-speech end and start times")
    non_speech_mask = list(zip(non_speech_start_times, non_speech_end_times))
    non_speech_start_times = [start for start, end in non_speech_mask]
    non_speech_end_times = [end for start, end in non_speech_mask]

    # Concatenate speech segments
    print("concatenating speech segments")
    speech_audio = AudioSegment.empty()
    for start, end in zip([0] + non_speech_end_times, non_speech_start_times):
        start_time_ms = start * 1000 / target_sr
        end_time_ms = end * 1000 / target_sr
        if start_time_ms < end_time_ms:
            segment = audio_file[start_time_ms:end_time_ms]
            speech_audio += segment

    # Concatenate non-speech segments
    print("concatenating non-speech segments")
    non_speech_audio = AudioSegment.empty()
    for start, end in zip(non_speech_end_times, non_speech_start_times[1:] + [None]):
        if end is not None:
            start_time_ms = start * 1000 / target_sr
            end_time_ms = end * 1000 / target_sr
            segment = audio_file[start_time_ms:end_time_ms]
            non_speech_audio += segment

    # Save separated audio files
    print("saving wav files")
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    speech_audio.export(os.path.join(output_folder, f"{file_name}_speech.wav"), format="wav")
    non_speech_audio.export(os.path.join(output_folder, f"{file_name}_non_speech.wav"), format="wav")
    speech_length = len(speech_audio)
    non_speech_length = len(non_speech_audio)

    return non_speech_length, speech_length