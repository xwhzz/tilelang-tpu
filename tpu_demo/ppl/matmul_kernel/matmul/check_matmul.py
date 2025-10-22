import numpy as np

left_matrix = np.fromfile("./test/matmul_left0.in", dtype=np.float32).reshape(384, 768)
right_matrix = np.fromfile("./test/matmul_right1.in", dtype=np.float32).reshape(768, 768)

result = np.matmul(left_matrix, right_matrix)

res = np.fromfile("./test/matmul_res2.out", dtype=np.float32).reshape(384, 768)


print(result)
print(res)
print(np.allclose(result, res, rtol=1e-2, atol=1e-2))
