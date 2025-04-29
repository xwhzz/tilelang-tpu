@I.ir_module
class Module:
    @T.prim_func
    def main(G_out: T.Buffer((1024, 1024), "float32"), G_in: T.Buffer((1024, 1024), "float32"), G_cos: T.Buffer((1024, 1024), "float32"), G_sin: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"in_cos": 98304, "in_sin": 81920, "in_x": 114688, "out": 0, "target": T.target({"arch": "sm_89", "host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "x_cos": 65536, "x_neg": 32768, "x_neg_sin": 16384, "x_sin": 49152})
        in_x = T.decl_buffer((64, 16), scope="shared.dyn")
        in_cos = T.decl_buffer((64, 16), scope="shared.dyn")
        in_sin = T.decl_buffer((64, 16), scope="shared.dyn")
        x_cos = T.decl_buffer((64, 16), scope="shared.dyn")
        x_sin = T.decl_buffer((64, 16), scope="shared.dyn")
        x_neg = T.decl_buffer((64, 16), scope="shared.dyn")
        x_neg_sin = T.decl_buffer((64, 16), scope="shared.dyn")
        out = T.decl_buffer((64, 16), scope="shared.dyn")
        for bx, by in T.grid(16, 64):
            T.call_extern("handle", "ppl.copy", T.region(G_in[bx * 64, by * 16], 1, 64, 16), T.region(in_x[0, 0], 2, 64, 16))
            T.call_extern("handle", "ppl.copy", T.region(G_cos[bx * 64, by * 16], 1, 64, 16), T.region(in_cos[0, 0], 2, 64, 16))
            T.call_extern("handle", "ppl.copy", T.region(G_sin[bx * 64, by * 16], 1, 64, 16), T.region(in_sin[0, 0], 2, 64, 16))
            T.call_extern("handle", "ppl.mul", T.tvm_access_ptr(T.type_annotation("float32"), x_cos.data, 0, 1024, 2), T.tvm_access_ptr(T.type_annotation("float32"), in_x.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), in_cos.data, 0, 1024, 1))
            T.call_extern("handle", "ppl.mul", T.tvm_access_ptr(T.type_annotation("float32"), x_sin.data, 0, 1024, 2), T.tvm_access_ptr(T.type_annotation("float32"), in_x.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), in_sin.data, 0, 1024, 1))
            T.call_extern("handle", "ppl.mul_C", T.tvm_access_ptr(T.type_annotation("float32"), x_neg.data, 0, 1024, 2), T.tvm_access_ptr(T.type_annotation("float32"), in_x.data, 0, 1024, 1), T.float32(-1))
            T.call_extern("handle", "ppl.mul", T.tvm_access_ptr(T.type_annotation("float32"), x_neg_sin.data, 0, 1024, 2), T.tvm_access_ptr(T.type_annotation("float32"), x_neg.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), in_sin.data, 0, 1024, 1))
            T.call_extern("handle", "ppl.rope_add", T.tvm_access_ptr(T.type_annotation("float32"), out.data, 0, 1024, 2), T.tvm_access_ptr(T.type_annotation("float32"), x_cos.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), x_neg_sin.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), x_cos.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), x_sin.data, 0, 1024, 1))
            T.call_extern("handle", "ppl.copy", T.region(out[0, 0], 1, 64, 16), T.region(G_out[bx * 64, by * 16], 2, 64, 16))
