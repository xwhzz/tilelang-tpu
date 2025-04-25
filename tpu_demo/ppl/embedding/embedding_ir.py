# from tvm.script import tir as T

@T.prim_func
def main(output: T.Buffer((7, 17), "float16"), params: T.Buffer((65, 17), "float16"), indices: T.Buffer((7, 1), "uint16")):
    # with T.block("root"):
    params_shared = T.alloc_buffer((65, 17), "float16", scope="shared.dyn")
    indices_shared = T.alloc_buffer((7, 1), "uint16", scope="shared.dyn")
    output_shared = T.alloc_buffer((7, 17), "float16", scope="shared.dyn")
    T.call_extern("handle", "ppl.copy", T.region(params[0, 0], 1, 65, 17), T.region(params_shared[0, 0], 2, 65, 17))
    T.call_extern("handle", "ppl.copy", T.region(indices[0, 0], 1, 7, 1), T.region(indices_shared[0, 0], 2, 7, 1))
    T.call_extern("handle", "ppl.fill", T.tvm_access_ptr(T.type_annotation("float16"), output_shared.data, 0, 119, 2), T.float32(0))
    outptr: T.handle = T.tvm_access_ptr(T.type_annotation("float16"), output_shared.data, 0, 119, 3)
    paramptr: T.handle = T.tvm_access_ptr(T.type_annotation("float16"), params_shared.data, 0, 1105, 1)
    indexptr: T.handle = T.tvm_access_ptr(T.type_annotation("uint16"), indices_shared.data, 0, 7, 1)
    with T.block("embedding"):
        T.reads()
        T.writes()
        params_tmp_buffer = T.alloc_buffer((17, 65), "float16", scope="shared.dyn")
        output_tmp_buffer = T.alloc_buffer((17, 7), "float16", scope="shared.dyn")
        params_tmp_ptr: T.handle = T.tvm_access_ptr(T.type_annotation("float16"), params_tmp_buffer.data, 0, 1105, 3)
        output_tmp_ptr: T.handle = T.tvm_access_ptr(T.type_annotation("float16"), output_tmp_buffer.data, 0, 119, 3)
        const_val: T.float32 = T.float32(0)
        T.call_extern("handle", "ppl.embedding", outptr, paramptr, indexptr, params_tmp_ptr, output_tmp_ptr, 1, 17, 65, 7, const_val)
    T.call_extern("handle", "ppl.copy", T.region(output_shared[0, 0], 1, 7, 17), T.region(output[0, 0], 2, 7, 17))