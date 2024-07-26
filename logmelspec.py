import tensorflow as tf
import librosa
import numpy as np

def _tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def power_to_db(magnitude, amin=1e-16, top_db=80.0):
    """
        https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """
    ref_value = tf.reduce_max(magnitude)
    log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
    return log_spec

class LogMelSpectrogram(tf.keras.layers.Layer):
    """Compute log-magnitude mel-scaled spectrograms."""

    def __init__(self, sample_rate=16000, fft_size=512, hop_size=512, n_mels=256,
                 f_min=30.0, f_max=8000, **kwargs):
        super(LogMelSpectrogram, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max)
        # self.mel_filterbank = librosa.filters.mel(self.sample_rate, 
        #     self.fft_size, 
        #     n_mels=256, 
        #     fmin=0.0, fmax=None, htk=False, norm="slaney").T

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(LogMelSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        """Forward pass.
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples)
            A Batch of mono waveforms.
        Returns
        -------
        log_mel_spectrograms : (tf.Tensor), shape = (None, time, freq, ch)
            The corresponding batch of log-mel-spectrograms
        """

        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=self.fft_size,
                                      frame_step=self.hop_size,
                                      fft_length=self.fft_size)
        
        magnitude_spectrograms = tf.abs(spectrograms)

        mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                    self.mel_filterbank)
        

        
        return mel_spectrograms
