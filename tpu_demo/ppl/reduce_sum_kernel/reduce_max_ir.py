# from tvm.script import tir as T

@T.prim_func
def main(X: T.Buffer((8192, 8192), "float32"), Y: T.Buffer((8192, 1), "float32")):
    # with T.block("root"):
    for bx, by in T.grid(1, 16):
        with T.block(""):
            T.reads(X[by * 512, 0])
            T.writes(Y[by * 512, 0])
            T.block_attr({"tilelang.is_cpu_kernel_frame": T.bool(True)})
            X_shared = T.alloc_buffer((512, 8192), scope="shared.dyn")
            Y_shared = T.alloc_buffer((512, 1), scope="shared.dyn")
            T.call_extern("handle", "ppl.fill", T.tvm_access_ptr(T.type_annotation("float32"), Y_shared.data, 0, 512, 2), T.float32(0))
            T.call_extern("handle", "ppl.copy", T.region(X[by * 512, 0], 1, 512, 8192), T.region(X_shared[0, 0], 2, 512, 8192))
            inpptr: T.handle = T.tvm_access_ptr(T.type_annotation("float32"), X_shared.data, 0, 4194304, 3)
            outptr: T.handle = T.tvm_access_ptr(T.type_annotation("float32"), Y_shared.data, 0, 512, 3)
            with T.block("reduce_sum"):
                T.reads()
                T.writes()
                tmp_buffer = T.alloc_buffer((512, 32), scope="shared.dyn")
                tmp_shape_0: T.int32 = 512
                tmp_shape_1: T.int32 = 32
                tmp_ptr: T.handle = T.tvm_access_ptr(T.type_annotation("float32"), tmp_buffer.data, 0, 16384, 3)
                eu_num: T.int32 = 32
                channel: T.int32 = 64
                align_w: T.int32 = (8192 + eu_num - 1) // eu_num * eu_num
                stride: T.int32 = (512 + channel - 1) // channel * align_w
                T.call_extern("handle", "ppl.reduce_sum", inpptr, outptr, tmp_ptr, eu_num, align_w, stride)
            T.call_extern("handle", "ppl.copy", T.region(Y_shared[0, 0], 1, 512, 1), T.region(Y[by * 512, 0], 2, 512, 1))