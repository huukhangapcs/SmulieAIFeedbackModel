import tensorflow as tf
from tensorflow.python.ops.signal import window_ops
import numpy as np
import math

N = 1024

inputs = tf.keras.Input((N,))
spectrograms = tf.signal.stft(inputs,
                            frame_length=N,
                            frame_step=N,
                            fft_length=N, window_fn=window_ops.hamming_window)
        
magnitude_spectrograms = tf.abs(spectrograms)

fft = tf.squeeze(magnitude_spectrograms,1)

fft1 = fft[:,1:156]
fft2 = fft[:,8:150]

arr1 = np.reshape(np.loadtxt("3ocarr1.txt", dtype=float), [37,142])
arr2 = np.reshape(np.loadtxt("3ocarr2.txt", dtype=float),[1,37]) # from C3
arr3 = np.reshape(np.loadtxt("3ocarr1.txt", dtype=float), [37,142])
arr4 = np.reshape(np.loadtxt("3ocarr2.txt", dtype=float),[1,37]) # from C3
arr3 = arr3[12:, :]
arr4 = arr4[:,12:]


note_idx_1 = np.arange(0, 36)
note_idx = np.arange(0, 24)
base_frequency = np.array([261.6256, 277.1826, 293.6648, 311.1270, 329.6276, 349.2282, 369.9944, 391.9954, 415.3047, 440.0000, 466.1638, 493.8833])
# base2_freq = base_frequency / 4.0
base3_freq = base_frequency / 2.0
base5_freq = base_frequency * 2.0
# base6_freq = base_frequency * 4.0

# concat_freq = base_frequency
# concat_freq = np.concatenate((base2_freq, base3_freq, base_frequency, base5_freq, base6_freq))
concat_freq_1 = np.concatenate((base3_freq, base_frequency, base5_freq))
concat_freq = np.concatenate((base_frequency, base5_freq))

sample_rate = 16000.0
frequencies = np.asarray(np.fft.fftfreq(N, d=1/sample_rate), dtype=float)

def normal_transform(arr, exp_sum: float, std: float = 11):
    normals = np.exp(-0.5 * (arr / std)**2) / (std * np.sqrt(2 * math.pi))
    diff = exp_sum / np.sum(normals)
    return normals * diff


max_freq = frequencies[150]

for i, x in enumerate(concat_freq_1):
    arr = np.zeros(N, dtype=float)
    num_freq = max_freq // x

    exp_weight = np.arange(num_freq, 0, -1)
    exp_weight = exp_weight/ np.sum(exp_weight)
    F_list = [x * i for i in range(1, len(exp_weight) + 1)]
    for f, w in zip(F_list, exp_weight):
        diff = np.abs(frequencies - f)
        diff = np.asarray(diff, dtype=float)
        idx = np.where(diff < 25)[0]
        arr[idx] += normal_transform(diff[idx], w)

    arr1[note_idx_1[i]] = arr[8:150]

arr3[:,] = arr1[12:,]

matrix1 = tf.convert_to_tensor(arr1, tf.float32)
matrix2 = tf.transpose(matrix1)
f = tf.convert_to_tensor(arr2, tf.float32)
pVar = tf.expand_dims(fft2, -1)
pVar2 = tf.ones([1,37])
pVar3 = tf.matmul(pVar, pVar2)
pVar3 = tf.math.multiply(pVar3, matrix2)


for i in range(20):
    pVar4 = tf.matmul(f, matrix1)
    pVar5 = tf.math.pow(pVar4+1e-6,-1.5)
    pVar10 = tf.math.pow(pVar4 + 1e-6, -0.5)
    pVar6 = tf.matmul(pVar5, pVar3)
    pVar6 = tf.math.divide(pVar6, tf.matmul(tf.math.multiply(pVar4, pVar10),matrix2))
    f = tf.math.multiply(f, pVar6)

matrix12 = tf.convert_to_tensor(arr3, tf.float32)
matrix22 = tf.transpose(matrix12)
f2 = tf.convert_to_tensor(arr4, tf.float32)
pVar12 = tf.expand_dims(fft2, -1)
pVar22 = tf.ones([1,25])
pVar32 = tf.matmul(pVar12, pVar22)
pVar32 = tf.math.multiply(pVar32, matrix22)
for i in range(20):
    pVar42 = tf.matmul(f2, matrix12)
    pVar52 = tf.math.pow(pVar42+1e-6,-1.5)
    pVar62 = tf.matmul(pVar52, pVar32)
    pVar62 = tf.math.divide(pVar62, tf.matmul(tf.math.multiply(pVar42, pVar52),matrix22))
    f2 = tf.math.multiply(f2, pVar62)

model = tf.keras.Model(inputs, [f, f2])
print(model.summary())

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True
import pathlib

tflite_models_dir = pathlib.Path("./")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/"MODEL_nmf_test_2SubModule_1024.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)