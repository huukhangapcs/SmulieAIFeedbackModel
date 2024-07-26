import tensorflow as tf
import numpy as np
# import coremltools as ct
from tensorflow.python.ops.signal import window_ops
# import tensorflow_probability as tfp

# inputs = tf.keras.Input((2048,))
input0 = tf.keras.Input((155,))
input1 = tf.keras.Input((142,))
# spectrograms = tf.signal.stft(inputs,
#                                       frame_length=2048,
#                                       frame_step=1024,
#                                       fft_length=1024, window_fn=window_ops.hamming_window)
        
# magnitude_spectrograms = tf.abs(spectrograms)*5

# fft = tf.squeeze(magnitude_spectrograms,1)

# fft1 = fft[:,1:156]

# pc1 = tfp.stats.percentile(fft1, 80)
# fft1 = tf.where(fft1>, fft1*1.5, fft1)
# pc2 = tfp.stats.percentile(fft1, 20)
# fft1 = tf.where(fft1<0.1, fft1/2, fft1)
# fft1 = tf.where(fft1>1, fft1*2, fft1)

arr1 = np.reshape(np.loadtxt("4ocarr1.txt", dtype=float), [50,155])
# arr1 = np.reshape(np.load("fftChange.npz")["x"], [50,155])
arr2 = np.reshape(np.loadtxt("4ocarr2.txt", dtype=float),[1,50])
arr3 = np.reshape(np.loadtxt("3ocarr1.txt", dtype=float), [37,142])
arr4 = np.reshape(np.loadtxt("3ocarr2.txt", dtype=float),[1,37])


# sum = 0
# for j in range (50):
#     for i in range(155):
#         if arr1[j,i] <0.005:
#             sum += arr1[j,i]
#             arr1[j,i] = 0

#     print(np.sum(arr1[j,:]))

print(arr1[24,:])

# print(arr1.shape)

# for i in range (49):
#     sum = 0
#     cnt = 0
#     for j in range (155):
#         if arr1[i,j]<0.005:
#             sum += arr1[i,j]
#             cnt += 1
#     print(sum)
#     val = sum/cnt
#     for j in range (155):
#         if (arr1[i,j]<0.005):
#             arr1[i,j] =0
#         else:
#             arr1[i,j] += val

# for i in range (36):
#     sum = 0
#     cnt = 0
#     for j in range (142):
#         if arr3[i,j]<0.005:
#             sum += arr3[i,j]
#             cnt += 1
#     print(sum)
#     val = sum/cnt
#     for j in range (142):
#         if (arr3[i,j]<0.005):
#             arr3[i,j] =0
#         else:
#             arr3[i,j] += val


# for i in range (49):



# for i in range(12,24):
#     sum = 0
#     cnt = 0
#     for j in range (50):
#         if (arr3[i,j]<0.02):
#             sum += arr3[i,j]
#             cnt +=1
#     print(sum)
#     val = sum/cnt
#     for j in range (50):
#         if (arr3[i,j]<0.02):
#             arr3[i,j] =0
#         else:
#             arr3[i,j] += val

# for i in range(12,24):
#     sum = 0
#     for j in range (50):
#         if (arr3[i,j]<0.02):
#             sum += arr3[i,j]
#     print(sum)        


# matrix1 = tf.convert_to_tensor(arr1, tf.float32)
# matrix2 = tf.transpose(matrix1)
# f = tf.convert_to_tensor(arr2, tf.float32)
# pVar = tf.expand_dims(input0, -1)
# pVar2 = tf.ones([1,50])
# pVar3 = tf.matmul(pVar, pVar2)
# pVar3 = tf.math.multiply(pVar3, matrix2)
# for i in range(20):
#     pVar4 = tf.matmul(f, matrix1)
#     pVar5 = tf.math.pow(pVar4+1e-6,-1.5)
#     pVar6 = tf.matmul(pVar5, pVar3)
#     pVar6 = tf.math.divide(pVar6, tf.matmul(tf.math.multiply(pVar4, pVar5),matrix2))
#     f = tf.math.multiply(f, pVar6)



# matrix12 = tf.convert_to_tensor(arr3, tf.float32)
# matrix22 = tf.transpose(matrix12)
# f2 = tf.convert_to_tensor(arr4, tf.float32)
# pVar12 = tf.expand_dims(input1, -1)
# pVar22 = tf.ones([1,37])
# pVar32 = tf.matmul(pVar12, pVar22)
# pVar32 = tf.math.multiply(pVar32, matrix22)
# for i in range(20):
#     pVar42 = tf.matmul(f2, matrix12)
#     pVar52 = tf.math.pow(pVar42+1e-6,-1.5)
#     pVar62 = tf.matmul(pVar52, pVar32)
#     pVar62 = tf.math.divide(pVar62, tf.matmul(tf.math.multiply(pVar42, pVar52),matrix22))
#     f2 = tf.math.multiply(f2, pVar62)

# model = tf.keras.Model([input0, input1], [f,f2])
# print(model.summary())



# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.allow_custom_ops = True
# import pathlib

# tflite_models_dir = pathlib.Path("./")
# tflite_models_dir.mkdir(exist_ok=True, parents=True)

# # converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_fp16_model = converter.convert()
# tflite_model_fp16_file = tflite_models_dir/"MODEL_nmf_full.tflite"
# tflite_model_fp16_file.write_bytes(tflite_fp16_model)

# mlmodel = ct.convert(model, convert_to="mlprogram",compute_precision=ct.precision.FLOAT32)
# mlmodel.save("SoundRecogntion.mlpackage")