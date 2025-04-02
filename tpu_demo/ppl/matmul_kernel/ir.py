@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((384, 786), "float16"), B: T.Buffer((786, 786), "float16"), C: T.Buffer((384, 786), "float32")):
        T.func_attr({"A_shared_0": 65536, "A_shared_1": 32768, "B_shared_0": 49152, "B_shared_1": 16384, "C_shared": 0, "target": T.target({"arch": "sm_89", "host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        C_shared = T.decl_buffer((128, 128), scope="shared.dyn")
        A_shared_0 = T.decl_buffer((128, 128), "float16", scope="shared.dyn")
        B_shared_0 = T.decl_buffer((128, 128), "float16", scope="shared.dyn")
        A_shared_1 = T.decl_buffer((128, 128), "float16", scope="shared.dyn")
        B_shared_1 = T.decl_buffer((128, 128), "float16", scope="shared.dyn")
        for bx, by in T.grid(7, 3):
            T.call_extern("handle", "ppl.fill", T.tvm_access_ptr(T.type_annotation("float32"), C_shared.data, 0, 16384, 2), T.float32(0))
            with T.attr(0, "async_scope", 1):
                A_shared: T.handle("float16", "shared.dyn") = A_shared_0.data
                A_shared_2 = T.Buffer((128, 128), "float16", data=A_shared, scope="shared.dyn")
                T.call_extern("handle", "ppl.copy", T.region(A[by * 128, 0], 1, 128, 128), T.region(A_shared_2[0, 0], 2, 128, 128))
            with T.attr(0, "async_commit_queue_scope", 0):
                T.attr(0, "async_scope", 1)
                B_shared: T.handle("float16", "shared.dyn") = B_shared_0.data
                B_shared_2 = T.Buffer((128, 128), "float16", data=B_shared, scope="shared.dyn")
                T.call_extern("handle", "ppl.copy", T.region(B[0, bx * 128], 1, 128, 128), T.region(B_shared_2[0, 0], 2, 128, 128))
            for k in range(6):
                with T.attr(0, "tpu_parallel_start", 0):
                    T.evaluate(0)
                with T.attr(0, "async_scope", 1):
                    A_shared: T.handle("float16", "shared.dyn") = T.if_then_else((k + 1) % 2 == 0, A_shared_0.data, A_shared_1.data)
                    A_shared_2 = T.Buffer((128, 128), "float16", data=A_shared, scope="shared.dyn")
                    T.call_extern("handle", "ppl.copy", T.region(A[by * 128, k * 128 + 128], 1, 128, 128), T.region(A_shared_2[0, 0], 2, 128, 128))
                with T.attr(0, "async_commit_queue_scope", 0):
                    T.attr(0, "async_scope", 1)
                    B_shared: T.handle("float16", "shared.dyn") = T.if_then_else((k + 1) % 2 == 0, B_shared_0.data, B_shared_1.data)
                    B_shared_2 = T.Buffer((128, 128), "float16", data=B_shared, scope="shared.dyn")
                    T.call_extern("handle", "ppl.copy", T.region(B[k * 128 + 128, bx * 128], 1, 128, 128), T.region(B_shared_2[0, 0], 2, 128, 128))
                with T.attr(0, "async_wait_queue_scope", 0):
                    T.attr(0, "async_wait_inflight_count", 1)
                    B_shared: T.handle("float16", "shared.dyn") = T.if_then_else(k % 2 == 0, B_shared_0.data, B_shared_1.data)
                    A_shared: T.handle("float16", "shared.dyn") = T.if_then_else(k % 2 == 0, A_shared_0.data, A_shared_1.data)
                    T.call_extern("handle", "ppl.gemm", T.tvm_access_ptr(T.type_annotation("float16"), A_shared, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_shared, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_shared.data, 0, 16384, 3), T.bool(False), T.bool(False), 128, 128, 128)
                T.attr(0, "tpu_parallel_end", 0)
                T.evaluate(0)
            with T.attr(0, "async_wait_queue_scope", 0):
                T.attr(0, "async_wait_inflight_count", 0)
                T.call_extern("handle", "ppl.gemm", T.tvm_access_ptr(T.type_annotation("float16"), A_shared_0.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_shared_0.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_shared.data, 0, 16384, 3), T.bool(False), T.bool(False), 128, 128, 128)
            T.call_extern("handle", "ppl.copy", T.region(C_shared[0, 0], 1, 128, 128), T.region(C[by * 128, bx * 128], 2, 128, 128))