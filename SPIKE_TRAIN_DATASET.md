# In-Depth Guide to Spike Train Dataset Creation

This document provides a detailed explanation of how the spike train dataset is created from raw audio files. The process, handled by `create_dataset.py`, is a crucial first step in preparing the data for our Spiking Neural Network.

## 1. The Goal: From Sound Wave to Spike Train

Our objective is to convert a continuous audio waveform into a format that a Spiking Neural Network (SNN) can understand: a **spike train**. A spike train is a sequence of binary events (spikes) that occur over time. This process mimics how our own auditory system converts sound waves into neural signals.

## 2. Audio Loading and Preprocessing

### Sample Rate: Why 16000 Hz?

We resample the audio to **16000 Hz**. This is a standard choice for speech recognition for two main reasons:

1.  **Covers the Speech Spectrum:** The most important frequencies for understanding human speech lie between 300 Hz and 8 kHz. According to the Nyquist theorem, a sample rate of 16000 Hz allows us to accurately capture frequencies up to 8 kHz, which is sufficient for our task.
2.  **Efficiency:** Higher sample rates (like 44100 Hz for music) would capture more detail, but much of that detail is irrelevant for speech and would significantly increase the computational load and memory requirements.

### Duration and Mono Conversion

- **Duration (1.0 second):** All audio files are standardized to a 1-second length. This ensures that every sample fed into the network has a consistent size.
- **Mono:** We convert the audio to a single channel (mono) as spatial audio information is not needed for this task.

## 3. Spectrogram Generation: Visualizing Sound

Next, we convert the 1D audio waveform into a 2D **spectrogram**. A spectrogram shows how the frequency content of the audio signal changes over time.

### Filterbanks: Mel vs. Gammatone

We use a filterbank to create the spectrogram. A filterbank is a set of filters that separates the audio signal into different frequency bands. We support two psychoacoustically-motivated filterbanks:

- **Mel Spectrogram:** This is the most common type of spectrogram in speech recognition. The Mel scale is based on how humans perceive pitch. It gives more resolution to lower frequencies (which are more important for speech) and less to higher frequencies.
- **Gammatone Spectrogram (Default):** This filterbank is inspired by the filtering that occurs in the human cochlea (the inner ear). Gammatone filters are known to be very effective at modeling the auditory system and have shown excellent performance in noisy speech recognition tasks. This is why we use it as the default.

The output of the filterbank is a spectrogram, which is then converted to a logarithmic scale (dB) and normalized to a range of [0, 1].

## 4. Spike Encoding: From Analog to Digital Spikes

This is the core of the process: converting the continuous-valued spectrogram into a binary spike train.

### Hysteresis Thresholding: A Robust Approach

We use **hysteresis thresholding** to generate spikes. This method is more robust to noise than a simple threshold.

1.  **Spike ON:** A spike is generated (turns ON) only when the signal's amplitude _crosses an "ON" threshold in an upward direction_.
2.  **Spike OFF:** The spike remains ON until the signal's amplitude _crosses a lower "OFF" threshold in a downward direction_. The "OFF" threshold is defined as `ON_threshold - HYSTERESIS_GAP`.

This prevents the "flickering" of spikes that can occur if the signal hovers right around a single threshold.

### Parameters in Detail

- **Time Bins (100):** We divide the 1-second audio clip into 100 discrete time steps. This gives us a temporal resolution of 10 ms per time bin, which is a good trade-off between capturing the dynamics of speech and keeping the data size manageable.
- **Spike Thresholds (`[0.70, 0.80, 0.90, 0.95]`):** We use four thresholds. This means that for each of the 128 frequency bins from the Gammatone filterbank, we generate four separate spike trains. This creates a rich, multi-channel representation of the audio.
- **Hysteresis Gap (0.1):** This value was chosen empirically to provide good noise immunity without losing too much detail.

## 5. Pure Redundancy: Leveraging Randomness

The **redundancy factor** (default is 1) allows us to repeat the spike train for each neuron. For example, with a redundancy factor of 3, each of the 128 frequency channels would have its spike train repeated 3 times.

This may seem counterintuitive, but it's a powerful technique for SNNs. Even though the repeated spike trains are identical, they connect to _different_ random neurons in the Liquid State Machine. This allows the LSM to learn a more robust and generalizable representation of the input.

## 6. Final Dataset

The final dataset is saved as a compressed NumPy file (`.npz`) containing:

- `X_spikes`: A 3D array of shape `(n_samples, n_neurons, n_time_bins)` containing the spike trains.
- `y_labels`: A 1D array of shape `(n_samples,)` containing the corresponding labels.
