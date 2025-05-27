# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((8192, 8192), "float32"), B: T.Buffer((8192, 8192), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_89", "host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        A_shared = T.decl_buffer((512, 8192), scope="shared.dyn")
        A_pow2 = T.decl_buffer((512, 8192), scope="shared.dyn")
        A_powsum = T.decl_buffer((512, 1), scope="shared.dyn")
        tmp_buffer = T.decl_buffer((512, 32), scope="shared.dyn")
        for bx in range(16):
            T.call_extern("handle", "ppl.copy", T.region(A[bx * 512:bx * 512 + 512, 0:8192], 1, 512, 8192), T.region(A_shared[0, 0], 2, 512, 8192))
            T.call_extern("handle", "ppl.mul", T.tvm_access_ptr(T.type_annotation("float32"), A_pow2.data, 0, 4194304, 2), T.tvm_access_ptr(T.type_annotation("float32"), A_shared.data, 0, 4194304, 1), T.tvm_access_ptr(T.type_annotation("float32"), A_shared.data, 0, 4194304, 1))
            T.call_extern("handle", "ppl.reduce_sum", T.tvm_access_ptr(T.type_annotation("float32"), A_pow2.data, 0, 4194304, 3), T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 512, 3), T.tvm_access_ptr(T.type_annotation("float32"), tmp_buffer.data, 0, 16384, 3), 32, 8192, 65536)
            T.call_extern("handle", "ppl.mul_C", T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 512, 2), T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 512, 1), T.float32(0.0001220703125))
            T.call_extern("handle", "ppl.add_C", T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 512, 2), T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 512, 1), T.float32(9.9999999999999998e-13))
            T.call_extern("handle", "ppl.rsqrt", T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 512, 2), T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 512, 1))
            T.call_extern("handle", "ppl.mul", T.tvm_access_ptr(T.type_annotation("float32"), A_shared.data, 0, 4194304, 2), T.tvm_access_ptr(T.type_annotation("float32"), A_shared.data, 0, 4194304, 1), T.tvm_access_ptr(T.type_annotation("float32"), A_powsum.data, 0, 512, 1))
            T.call_extern("handle", "ppl.copy", T.region(A_shared[0, 0], 1, 512, 8192), T.region(B[bx * 512:bx * 512 + 512, 0:8192], 2, 512, 8192))