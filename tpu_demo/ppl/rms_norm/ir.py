# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((8192, 8192), "float32"), B: T.Buffer((8192, 8192), "float32")):
        T.func_attr({"A_pow2": 0, "A_powsum": 114688, "A_shared": 49152, "target": T.target({"arch": "sm_89", "host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tmp_buffer": 98304})
        A_shared = T.decl_buffer((16, 8192), scope="shared.dyn")
        A_pow2 = T.decl_buffer((16, 8192), scope="shared.dyn")
        A_powsum = T.decl_buffer((16, 1), scope="shared.dyn")
        tmp_buffer = T.decl_buffer((16, 32), scope="shared.dyn")
        for bx in range(512):
            T.call_extern("handle", "ppl.copy", T.region(A[bx * 16:bx * 16 + 16, 0:8192], 1, 16, 8192), T.region(A_shared[0, 0], 2, 16, 8192))
            T.call_extern("handle", "ppl.mul", T.tvm_access_ptr(T.type_annotation("float32"), A_pow2.data, 0, 131072, 2), T.tvm_access_ptr(T.type_annotation("float32"), A_shared.data, 0, 131072, 1), T.tvm_access_ptr(T.type_annotation("float32"), A_shared.data, 0, 131072, 1))
            T.call_extern("handle", "ppl.reduce_sum", T.tvm_access_ptr(T.type_annotation("float32"), A_pow2.data, 0, 131072, 3), T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 16, 3), T.tvm_access_ptr(T.type_annotation("float32"), tmp_buffer.data, 0, 512, 3), 32, 8192, 8192)
            T.call_extern("handle", "ppl.mul_C", T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 16, 2), T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 16, 1), T.float32(0.0001220703125))
            T.call_extern("handle", "ppl.add_C", T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 16, 2), T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 16, 1), T.float32(9.9999999999999998e-13))
            T.call_extern("handle", "ppl.rsqrt", T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 16, 2), T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 16, 1))
            T.call_extern("handle", "ppl.mul", T.tvm_access_ptr(T.type_annotation("float32"), A_shared.data, 0, 131072, 2), T.tvm_access_ptr(T.type_annotation("float32"), A_shared.data, 0, 131072, 1), T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 16, 1))
            T.call_extern("handle", "ppl.copy", T.region(A_shared[0, 0], 1, 16, 8192), T.region(B[bx * 16:bx * 16 + 16, 0:8192], 2, 16, 8192))