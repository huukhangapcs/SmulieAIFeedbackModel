import tensorflow as tf
from tensorflow.python.ops.signal import window_ops
import math
import numpy as np
np.set_printoptions(suppress=True)

N = 2048

inputs = tf.keras.Input((N,))
spectrograms = tf.signal.stft(inputs,
                            frame_length=N,
                            frame_step=N,
                            fft_length=N, window_fn=window_ops.hamming_window)

magnitude_spectrograms = tf.abs(spectrograms)

fft = tf.squeeze(magnitude_spectrograms,1)

fft2 = fft[:,6:540]

arr2 = np.reshape(np.loadtxt("4ocarr2.txt", dtype=float),[1,50])

arr3 = np.zeros((60, 534))
arr4 = np.reshape(np.array([0.5] * 60), [1, 60])

for i in range(48):
    arr4[0,i] = arr2[0,i]

note_idx = np.arange(0, 60)
base_frequency = np.array([261.6256, 277.1826, 293.6648, 311.1270, 329.6276, 349.2282, 369.9944, 391.9954, 415.3047, 440.0000, 466.1638, 493.8833])
base2_freq = base_frequency / 4.0
base3_freq = base_frequency / 2.0
base5_freq = base_frequency * 2.0
base6_freq = base_frequency * 4.0

# concat_freq = base_frequency
concat_freq = np.concatenate((base2_freq, base3_freq, base_frequency, base5_freq, base6_freq))

sample_rate = 16000.0
frequencies = np.asarray(np.fft.fftfreq(N, d=1/sample_rate), dtype=float)

# 1024: std = 15
# 2048: std: 10
def normal_transform(arr, exp_sum: float, std: float = 11):
    normals = np.exp(-0.5 * (arr / std)**2) / (std * np.sqrt(2 * math.pi))
    diff = exp_sum / np.sum(normals)
    return normals * diff

max_freq = frequencies[300]
for i, x in enumerate(concat_freq):
    arr = np.zeros(N, dtype=float)
    num_freq = max_freq // x

    exp_weight = np.arange(num_freq, 0, -1) ** 2
    exp_weight = exp_weight / np.sum(exp_weight)
    F_list = [x * i for i in range(1, len(exp_weight) + 1)]
    for f, w in zip(F_list, exp_weight):
        diff = np.abs(frequencies - f)
        diff = np.asarray(diff, dtype=float)
        idx = np.where(diff < 25)[0]
        arr[idx] += normal_transform(diff[idx], w)
    arr3[note_idx[i]] = arr[6:540]

matrix12 = tf.convert_to_tensor(arr3, tf.float32)
matrix22 = tf.transpose(matrix12)
f2 = tf.convert_to_tensor(arr4, tf.float32)
pVar12 = tf.expand_dims(fft2, -1)
pVar22 = tf.ones([1,60])
pVar32 = tf.matmul(pVar12, pVar22)
pVar32 = tf.math.multiply(pVar32, matrix22)


for i in range(20):
    pVar42 = tf.matmul(f2, matrix12)
    pVar52 = tf.math.pow(pVar42+1e-6,-1.5)
    pVar62 = tf.matmul(pVar52, pVar32)
    pVar62 = tf.math.divide(pVar62, tf.matmul(tf.math.multiply(pVar42, pVar52),matrix22))
    f2 = tf.math.multiply(f2, pVar62)


model = tf.keras.Model(inputs, f2)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True
import pathlib

tflite_models_dir = pathlib.Path("./")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/"MODEL_nmf_test_C2B6_2048.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)