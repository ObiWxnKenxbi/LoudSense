import wave
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import webrtcvad
import pyvad

def analyze_audio(output_path, file_path, vad_param, hop_length_param):
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
    ffmpeg_loudness_values = []
    speech_loudness_values = []
    audio_segments = []  # List to store audio segments
    
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        start_time_sec = start / sample_rate
        end_time_sec = end / sample_rate
        pyln_loudness, ffmpeg_loudness = calculate_loudness(file_path, start_time_sec, end_time_sec)
    
        if pyln_loudness is None:
            pyln_message = "Unable to calculate loudness with pyln"
        else:
            pyln_message = f"{pyln_loudness:.2f}"
            pyln_loudness_values.append(pyln_loudness)
        
        # Extract audio segment based on start and end times
        segment = audio_file[start_time_sec * 1000:end_time_sec * 1000]
        audio_segments.append(segment)
        
        # Save the audio segment as a WAV file
        segment.export(f"{output_path}/segment_{i+1}.wav", format="wav")
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
    return audio_segments

folder_path = "/../../audios"
audio_files = os.listdir(folder_path)
#random_video = random.choice(audio_files)
vad_param = 3
hop_length_param = 10
file_path = "/../../audios/10986-es_cuatro_20210705_132545.wav"
output_path = "/../../output"
print("chosen audio:", file_path)
audio_segments = analyze_audio(output_path, file_path, vad_param, hop_length_param)
print("Audio Segments", audio_segments)