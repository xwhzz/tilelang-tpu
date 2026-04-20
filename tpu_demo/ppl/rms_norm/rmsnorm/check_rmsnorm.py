import numpy as np

input_vector = np.fromfile("./test/rms_norm_input0.in", dtype=np.float32).reshape(128, 128)

eps = 1e-12

result = input_vector / np.sqrt(np.mean(np.square(input_vector), axis=1, keepdims=True) + eps)

res = np.fromfile("./test/rms_norm_output1.out", dtype=np.float32).reshape(128, 128)

print("input vector:")
print(input_vector)
print("numpy result:")
print(result)
print("tilelang-tpu result:")
print(res)
print(np.allclose(result, res, rtol=1e-3))
