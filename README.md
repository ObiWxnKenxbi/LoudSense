<h1 style="text-align: justify;">LoudSense</h1>
<p style="text-align: justify;">This repository contains the code and documentation for a project focused on measuring the loudness of background music in audiovisual productions.</p>
<p style="text-align: justify;">The goal of this project is to develop an algorithm that accurately measures the loudness of background music, facilitating compliance with copyright regulations.</p>
<p style="text-align: justify;">For more information about the project, please refer to:
  <a href="https://www.upf.edu/web/mtg/ongoing-projects/-/asset_publisher/DneGVrJZ7tmE/content/id/245414020/maximized" target="_blank" style="text-align: justify;">
    https://www.upf.edu/web/mtg/ongoing-projects/-/asset_publisher/DneGVrJZ7tmE/content/id/245414020/maximized
  </a>
</p>
<h2 style="text-align: justify;">Introduction</h2>
<p style="text-align: justify;">In recent years, copyright regulations related to the use of background music in audiovisual productions have gained significant attention. To provide standardized measurements, a new framework was introduced that categorizes the audibility of background music into three levels: audible, barely audible, and not audible. These measurements aim to provide clearer guidelines for copyright enforcement and licensing purposes, ensuring proper compensation for rights holders.</p>

<p style="text-align: justify;">The central objective of this project is to develop an algorithm capable of accurately measuring the loudness of background music, considering the new copyright regulations. The algorithm utilizes Voice Activity Detection (VAD) to detect the loudness of background music during non-speech segments. By focusing on these specific intervals, more precise measurements can be extracted, ensuring a robust analysis of audibility levels.</p>

<h2 style="text-align: justify;">Methodologies Explored</h2>
<p style="text-align: justify;">Several approaches were considered for measuring the loudness of background music. One method involved using audio signal processing techniques such as Mel Frequency Cepstral Coefficients (MFCCs) or Constant-Q Transform (CQT) to separate the audio into different clusters based on their spectral features. Adaptive filtering and pre-trained models like WaveNet and Unet were also explored for audio separation.</p>

<p style="text-align: justify;">Voice Activity Detection (VAD) models were investigated to identify which portions of the audio contain speech. Deep learning models, including convolutional neural networks (CNNs) and recurrent neural networks (RNNs), were explored for VAD. The py-webrtcvad library was identified as the most suitable option for accurate VAD results in the presence of background noise.</p>

<h2 style="text-align: justify;">Project Progress</h2>
<p style="text-align: justify;">The project progressed through various stages and iterations to address the challenges and considerations encountered. The conceptual narrative provides an overview of the progress made during the internship, including the exploration of VAD models, code improvements, and the incorporation of weighted loudness average into the system.</p>

<h2 style="text-align: justify;">Results</h2>
<p style="text-align: justify;">The study's results demonstrate the promising performance of the Voice Activity Detection (VAD) system in accurately detecting speech segments within audio files, even in the presence of background noise. A perceptual analysis was conducted, aligning with the numerical evaluations and highlighting the consistency between the VAD system and source separation results.</p>

<p style="text-align: justify;">Although the overall performance was positive, challenges were encountered when background music with vocals or certain musical instruments was present. The VAD algorithm occasionally misclassified these instances as speech. However, these limitations did not significantly deviate from the results obtained through the previous source separation approach.</p>
<p style="text-align: justify;">For more information, check the <a href="https://github.com/ObiWxnKenxbi/LoudSense/tree/main/results" style="text-align: justify;">results</a> page.</p>

<h2>Repository Structure</h2>
<pre>
/code
  - analysis_experiment_e/ (code for analyzing the audios from experiment e)
  - csv_file/ (code related to CSV file processing)
  - differences/ (code for computing differences between demucs, voice on set and source separation)
  - initial_analysis/ (code for initial analysis)
  - pyloudnorm_analysis/ (code for analyzing loudness using pyloudnorm)
  - RX_PYLN_Comparison/ (code for comparing iztopoes RX7 and pyloudnorm)
  - segment_conc/ (code for segment concatenation)
  - segment_separation/ (code for audio segment separation)
  - utilities/ (utility functions)
/audios
  - [selected audio files used]
  /experiment_e (selected clips from the e experiment)
    - [audio files]
  /original audios (original audio files used for the analysis)
    - [audio files]
/output (Output audios of the speech/non-speech segmentation)
/results (results in CSV format)
  /spectrograms (spectrograms showing the VAD algorithm)
    - [spectrogram images]
</pre>

<h2>Usage</h2>
<p>The code provided in the repository can be used as a reference or starting point for developing algorithms to measure the loudness of background music in audiovisual productions. The documentation provides detailed insights into the methodologies, implementation, and results obtained.</p>

<p style="text-align: justify;">Please refer to the specific directories within the repository for detailed instructions on running the code, accessing datasets, and utilizing the provided models.</p>

<h2>Contributors</h2>
<p><a href="https://www.bmat.com/innovation-lab/">BMAT</a> and <a href="https://www.upf.edu/web/mtg">Music Technology Group</a></p>

<h2>License</h2>
<p style="text-align: justify;">This project is licensed under the GNU Affero General Public License</p>
