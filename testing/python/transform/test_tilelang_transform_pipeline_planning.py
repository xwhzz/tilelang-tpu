# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import tempfile

from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T

try:
    import tilelang.testing as tilelang_testing
except ModuleNotFoundError:
    tilelang_testing = None

auto_target = tvm.target.Target(determine_target("auto"))
tpu_target = tvm.target.Target("tpu")


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    mod = tl.transform.Simplify()(mod)
    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    transformed = tvm.tir.transform.BindTarget(auto_target)(transformed)
    tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


def test_simple_pipeline():

    @T.prim_func
    def before(A: T.Tensor((1024, 32), "float32"), B: T.Tensor((32, 1024), "float32"), C: T.Tensor(
        (1024, 1024), "float32")):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float32")
            B_shared = T.alloc_shared((32, 128), "float32")
            C_local = T.alloc_fragment((128, 128), "float32")

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    @T.prim_func
    def after(A: T.Tensor((1024, 32), "float32"), B: T.Tensor((32, 1024), "float32"), C: T.Tensor(
        (1024, 1024), "float32")):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float32")
            B_shared = T.alloc_shared((32, 128), "float32")
            C_local = T.alloc_fragment((128, 128), "float32")

            T.clear(C_local)

            for ko in T.serial(
                    32,
                    annotations={
                        "software_pipeline_order": [0, 1, 2],
                        "software_pipeline_stage": [3, 3, 3]
                    }):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    _check(before, after)


def test_pipeline_certificate_dump():

    @T.prim_func
    def before(A: T.Tensor((1024, 32), "float32"), B: T.Tensor((32, 1024), "float32"), C: T.Tensor(
        (1024, 1024), "float32")):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float32")
            B_shared = T.alloc_shared((32, 128), "float32")
            C_local = T.alloc_fragment((128, 128), "float32")

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    fd, path = tempfile.mkstemp(prefix="tl_pipeline_certificate_", suffix=".txt")
    os.close(fd)
    old_value = os.environ.get("TL_TPU_PIPELINE_CERTIFICATE")
    os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = path
    try:
        mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.BindTarget(auto_target)(mod)
        tl.transform.PipelinePlanning()(mod)
        with open(path, "r", encoding="utf-8") as f:
            certificate = f.read()
    finally:
        if old_value is None:
            os.environ.pop("TL_TPU_PIPELINE_CERTIFICATE", None)
        else:
            os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = old_value
        if os.path.exists(path):
            os.remove(path)

    assert "PIPELINE_CERTIFICATE_BEGIN" in certificate
    assert "function: main" in certificate
    assert "num_stages: 3" in certificate
    assert "planned_stage:" in certificate
    assert "planned_order:" in certificate
    assert "reads:" in certificate
    assert "writes:" in certificate


def test_pipeline_manual_stage_order_annotations():

    @T.prim_func
    def before(A: T.Tensor((1024, 32), "float32"), B: T.Tensor((32, 1024), "float32"), C: T.Tensor(
        (1024, 1024), "float32")):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float32")
            B_shared = T.alloc_shared((32, 128), "float32")
            C_local = T.alloc_fragment((128, 128), "float32")

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3, order=[1, 0, 2], stage=[0, 0, 1]):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    text = str(mod)

    assert '"software_pipeline_order": [1, 0, 2]' in text
    assert '"software_pipeline_stage": [0, 0, 1]' in text


def test_tpu_stage_search_and_buffer_version_hints():

    @T.prim_func
    def before(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64, 64), "float16"), C: T.Tensor(
        (64, 64), "float32")):
        with T.Kernel(2, 2, is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((32, 32), "float16")
            B_shared = T.alloc_shared((32, 32), "float16")
            C_shared = T.alloc_shared((32, 32), "float32")

            T.ppl_fill(C_shared, T.float32(0))
            for ko in T.Pipelined(2, num_stages=2):
                T.ppl_copy(A[by * 32, ko * 32], A_shared)
                T.ppl_copy(B[ko * 32, bx * 32], B_shared)
                T.ppl_gemm(A_shared, B_shared, C_shared)

            T.ppl_copy(C_shared, C[by * 32, bx * 32])

    fd, path = tempfile.mkstemp(prefix="tl_tpu_stage_search_", suffix=".txt")
    os.close(fd)
    old_value = os.environ.get("TL_TPU_PIPELINE_CERTIFICATE")
    os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = path
    try:
        mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.BindTarget(tpu_target)(mod)
        mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        planned = tl.transform.PipelinePlanning()(mod)
        injected = tl.transform.InjectSoftwarePipeline()(planned)
        with open(path, "r", encoding="utf-8") as f:
            certificate = f.read()
    finally:
        if old_value is None:
            os.environ.pop("TL_TPU_PIPELINE_CERTIFICATE", None)
        else:
            os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = old_value
        if os.path.exists(path):
            os.remove(path)

    planned_text = str(planned)
    injected_text = str(injected)
    assert "optimizer_decision: tpu_stage_search_exact" in certificate, certificate
    assert "pipeline_policy: gemm_heavy" in certificate, certificate
    assert "policy_family: pure_gemm" in certificate, certificate
    assert "evidence_tier: tier_2_model_local" in certificate, certificate
    assert "policy_stage_optimization: enabled" in certificate, certificate
    assert "policy_buffer_version_hints: enabled" in certificate, certificate
    assert "policy_stage_aware_lmem: enabled" in certificate, certificate
    assert "engine_timeline:" in certificate, certificate
    assert "profitability_guard:" in certificate, certificate
    assert "status: model_bound_non_regression" in certificate, certificate
    assert "trip_count_static: 2" in certificate, certificate
    assert "recommendation: model_local_claim_only" in certificate, certificate
    assert "phase: gemm" in certificate, certificate
    assert "tpu_cost_model:" in certificate, certificate
    assert "model: bm1690_relative_v2_phase_cost" in certificate, certificate
    assert "legacy_dma_cost_units:" in certificate, certificate
    assert "phase_gdma_cost_units=32" in certificate, certificate
    assert "phase_bdc_cost_units=64" in certificate, certificate
    assert "status: active" in certificate, certificate
    assert "s2l_bytes: 4096" in certificate, certificate
    assert "bdc_ops: 65536" in certificate, certificate
    assert "versioned_lmem_bytes: 384" in certificate, certificate
    assert "max_raw_stage_distance: 1" in certificate, certificate
    assert "expanded_prefetch_edges: 0" in certificate, certificate
    assert "prefetch_distance_penalty: 0" in certificate, certificate
    assert "local_buffer_footprints:" in certificate, certificate
    assert "bm1690_tpu_align_bytes: 64" in certificate, certificate
    assert "tpu_resource: {s2l_bytes=2048" in certificate, certificate
    assert "tpu_resource: {s2l_bytes=0,l2s_bytes=0,local_copy_bytes=0,bdc_ops=65536" in certificate, certificate
    assert '"software_pipeline_stage": [0, 0, 1]' in planned_text, planned_text
    assert "tl_pipeline_buffer_versions" in planned_text, planned_text
    assert '"A_shared": 2' in planned_text, planned_text
    assert '"B_shared": 2' in planned_text, planned_text
    assert "A_shared_0" in injected_text, injected_text
    assert "A_shared_1" in injected_text, injected_text
    assert "B_shared_0" in injected_text, injected_text
    assert "B_shared_1" in injected_text, injected_text


def test_tpu_store_in_loop_stage_search_policy_guard():

    @T.prim_func
    def before(A: T.Tensor((128, 32), "float16"), B: T.Tensor((128, 32), "float16"), C: T.Tensor(
        (128, 32), "float32")):
        with T.Kernel(1, 1, is_cpu=True):
            A_shared = T.alloc_shared((32, 32), "float16")
            B_shared = T.alloc_shared((32, 32), "float16")
            C_shared = T.alloc_shared((32, 32), "float32")

            for tile in T.Pipelined(4, num_stages=3):
                T.ppl_copy(A[tile * 32, 0], A_shared)
                T.ppl_copy(B[tile * 32, 0], B_shared)
                T.ppl_fill(C_shared, T.float32(0))
                T.ppl_gemm(A_shared, B_shared, C_shared)
                T.ppl_copy(C_shared, C[tile * 32, 0])

    fd, path = tempfile.mkstemp(prefix="tl_tpu_store_loop_policy_", suffix=".txt")
    os.close(fd)
    old_value = os.environ.get("TL_TPU_PIPELINE_CERTIFICATE")
    os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = path
    try:
        mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.BindTarget(tpu_target)(mod)
        mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        planned = tl.transform.PipelinePlanning()(mod)
        with open(path, "r", encoding="utf-8") as f:
            certificate = f.read()
    finally:
        if old_value is None:
            os.environ.pop("TL_TPU_PIPELINE_CERTIFICATE", None)
        else:
            os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = old_value
        if os.path.exists(path):
            os.remove(path)

    planned_text = str(planned)
    assert "optimizer_decision: tpu_stage_search_exact" in certificate, certificate
    assert "pipeline_policy: gemm_heavy" in certificate, certificate
    assert "tpu_pipeline_role: load" in certificate, certificate
    assert "tpu_pipeline_role: compute" in certificate, certificate
    assert "tpu_pipeline_role: store" in certificate, certificate
    assert "s2l_bytes: 4096" in certificate, certificate
    assert "l2s_bytes: 4096" in certificate, certificate
    assert "bdc_ops: 66560" in certificate, certificate
    assert "max_raw_stage_distance: 1" in certificate, certificate
    assert "expanded_prefetch_edges: 0" in certificate, certificate
    assert "prefetch_distance_penalty: 0" in certificate, certificate
    assert "unknown_static_cost: false" in certificate, certificate
    assert "tpu_resource: {s2l_bytes=0,l2s_bytes=0,local_copy_bytes=0,bdc_ops=1024" in certificate, certificate
    assert "max_required_versions: 2" in certificate, certificate
    assert '"software_pipeline_stage": [0, 0, 1, 1, 2]' in planned_text, planned_text


def test_tpu_copy_transaction_pre_codegen_falls_back_without_error():

    @T.prim_func
    def before(A: T.Tensor((32, 32), "float16"), C: T.Tensor((32, 32), "float16")):
        with T.Kernel(1, 1, is_cpu=True):
            scratch = T.alloc_shared((32, 32), "float16")
            for ko in T.Pipelined(1, num_stages=2):
                T.ppl_copy(A, C)
                T.ppl_fill(scratch, T.float32(0))

    fd, path = tempfile.mkstemp(prefix="tl_tpu_copy_fallback_", suffix=".txt")
    os.close(fd)
    old_value = os.environ.get("TL_TPU_PIPELINE_CERTIFICATE")
    os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = path
    try:
        mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.BindTarget(tpu_target)(mod)
        planned = tl.transform.PipelinePlanning()(mod)
        with open(path, "r", encoding="utf-8") as f:
            certificate = f.read()
    finally:
        if old_value is None:
            os.environ.pop("TL_TPU_PIPELINE_CERTIFICATE", None)
        else:
            os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = old_value
        if os.path.exists(path):
            os.remove(path)

    planned_text = str(planned)
    assert "pipeline_policy: unsupported_tpu_role" in certificate, certificate
    assert "policy_stage_optimization: disabled" in certificate, certificate
    assert "policy_buffer_version_hints: disabled" in certificate, certificate
    assert "optimizer_decision: legacy_heuristic:policy_stage_optimization_disabled:unsupported_tpu_role" in certificate, certificate
    assert "tl_pipeline_buffer_versions" not in planned_text, planned_text


def test_tpu_stage_aware_address_assign_dump():

    @T.prim_func
    def before(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64, 64), "float16"), C: T.Tensor(
        (64, 64), "float32")):
        with T.Kernel(2, 2, is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((32, 32), "float16")
            B_shared = T.alloc_shared((32, 32), "float16")
            C_shared = T.alloc_shared((32, 32), "float32")

            T.ppl_fill(C_shared, T.float32(0))
            for ko in T.Pipelined(2, num_stages=2):
                T.ppl_copy(A[by * 32, ko * 32], A_shared)
                T.ppl_copy(B[ko * 32, bx * 32], B_shared)
                T.ppl_gemm(A_shared, B_shared, C_shared)

            T.ppl_copy(C_shared, C[by * 32, bx * 32])

    fd, path = tempfile.mkstemp(prefix="tl_tpu_address_assign_", suffix=".txt")
    os.close(fd)
    old_value = os.environ.get("TL_TPU_ADDRESS_ASSIGN_DUMP")
    os.environ["TL_TPU_ADDRESS_ASSIGN_DUMP"] = path
    try:
        mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.BindTarget(tpu_target)(mod)
        mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tl.transform.PipelinePlanning()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        injected_text = str(mod)
        mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
        tl.transform.AddressAssign()(mod)
        with open(path, "r", encoding="utf-8") as f:
            dump = f.read()
    finally:
        if old_value is None:
            os.environ.pop("TL_TPU_ADDRESS_ASSIGN_DUMP", None)
        else:
            os.environ["TL_TPU_ADDRESS_ASSIGN_DUMP"] = old_value
        if os.path.exists(path):
            os.remove(path)

    assert "stage_aware_lmem: enabled" in dump, dump
    assert "pipeline_base: A_shared" in dump, dump
    assert "pipeline_base: B_shared" in dump, dump
    assert "stage_sibling_conflict_degree: 1" in dump, dump


def test_tpu_mixed_gemm_policy_disables_gemm_heavy_optimizations():

    @T.prim_func
    def before(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64, 64), "float16"), C: T.Tensor(
        (64, 64), "float32")):
        with T.Kernel(2, 2, is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((32, 32), "float16")
            B_shared = T.alloc_shared((32, 32), "float16")
            C_shared = T.alloc_shared((32, 32), "float32")
            tmp = T.alloc_shared((32, 32), "float32")

            T.ppl_fill(C_shared, T.float32(0))
            for ko in T.Pipelined(2, num_stages=2):
                T.ppl_copy(A[by * 32, ko * 32], A_shared)
                T.ppl_copy(B[ko * 32, bx * 32], B_shared)
                T.ppl_gemm(A_shared, B_shared, C_shared)
                T.ppl_mul(tmp, C_shared, C_shared)

            T.ppl_copy(C_shared, C[by * 32, bx * 32])

    cert_fd, cert_path = tempfile.mkstemp(prefix="tl_tpu_mixed_policy_", suffix=".txt")
    dump_fd, dump_path = tempfile.mkstemp(prefix="tl_tpu_mixed_address_", suffix=".txt")
    os.close(cert_fd)
    os.close(dump_fd)
    old_cert = os.environ.get("TL_TPU_PIPELINE_CERTIFICATE")
    old_dump = os.environ.get("TL_TPU_ADDRESS_ASSIGN_DUMP")
    os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = cert_path
    os.environ["TL_TPU_ADDRESS_ASSIGN_DUMP"] = dump_path
    try:
        mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.BindTarget(tpu_target)(mod)
        mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tl.transform.PipelinePlanning()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        injected_text = str(mod)
        mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
        tl.transform.AddressAssign()(mod)
        with open(cert_path, "r", encoding="utf-8") as f:
            certificate = f.read()
        with open(dump_path, "r", encoding="utf-8") as f:
            dump = f.read()
    finally:
        if old_cert is None:
            os.environ.pop("TL_TPU_PIPELINE_CERTIFICATE", None)
        else:
            os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = old_cert
        if old_dump is None:
            os.environ.pop("TL_TPU_ADDRESS_ASSIGN_DUMP", None)
        else:
            os.environ["TL_TPU_ADDRESS_ASSIGN_DUMP"] = old_dump
        for path in (cert_path, dump_path):
            if os.path.exists(path):
                os.remove(path)

    assert "pipeline_policy: mixed_gemm" in certificate, certificate
    assert "policy_family: attention_mixed" in certificate, certificate
    assert "policy_stage_optimization: disabled" in certificate, certificate
    assert "policy_buffer_version_hints: disabled" in certificate, certificate
    assert "policy_stage_aware_lmem: disabled" in certificate, certificate
    assert "optimizer_decision: legacy_heuristic:policy_stage_optimization_disabled:mixed_gemm" in certificate, certificate
    assert "status: inactive_policy_stage_optimization_disabled" in certificate, certificate
    assert "parallel_guard_region:" in certificate, certificate
    assert "enabled: true" in certificate, certificate
    assert "annotation: tl_tpu_parallel_guard_region" in certificate, certificate
    assert "profitability_guard:" in certificate, certificate
    assert "status: legality_or_opt_in_materializer_only" in certificate, certificate
    assert "recommendation: keep_materializers_opt_in_until_positive_timing_region" in certificate, certificate
    assert "tpu_parallel_start" in injected_text, injected_text
    assert "tpu_parallel_end" in injected_text, injected_text
    assert "tl_tpu_parallel_guard_region" not in injected_text, injected_text
    assert "stage_aware_lmem: disabled" in dump, dump
    assert "stage_aware_lmem_policy: mixed_gemm" in dump, dump
    assert "stage_aware_lmem_reason: gemm_function_with_elementwise_or_reduction_ops" in dump, dump


def test_tpu_attention_mixed_overlap_window_certificate():

    @T.prim_func
    def before(K: T.Tensor((4, 8), "float16"), V: T.Tensor((4, 8), "float16"), O: T.Tensor(
        (2, 8), "float32")):
        with T.Kernel(1, is_cpu=True):
            Q_shared = T.alloc_shared((2, 8), "float16")
            K_shared = T.alloc_shared((2, 8), "float16")
            V_shared = T.alloc_shared((2, 8), "float16")
            acc_s = T.alloc_shared((2, 2), "float32")
            acc_s_cast = T.alloc_shared((2, 2), "float16")
            acc_o = T.alloc_shared((2, 8), "float32")
            scores_sum = T.alloc_shared((2, 1), "float32")

            T.ppl_fill(acc_o, T.float32(0))
            for k in T.Pipelined(2, num_stages=1):
                with T.block("load_k"):
                    T.reads(K[k * 2:k * 2 + 2, 0:8])
                    T.writes(K_shared[0:2, 0:8])
                    T.ppl_copy(K[k * 2, 0], K_shared)
                with T.block("fill_scores"):
                    T.reads()
                    T.writes(acc_s[0:2, 0:2])
                    T.ppl_fill(acc_s, T.float32(0))
                with T.block("score_gemm"):
                    T.reads(Q_shared[0:2, 0:8], K_shared[0:2, 0:8], acc_s[0:2, 0:2])
                    T.writes(acc_s[0:2, 0:2])
                    T.ppl_gemm(Q_shared, K_shared, acc_s, transpose_B=True)
                with T.block("softmax_reduce"):
                    T.reads(acc_s[0:2, 0:2])
                    T.writes(scores_sum[0:2, 0:1])
                    T.ppl_reduce_sum(acc_s, scores_sum, dim=1)
                with T.block("cast_scores"):
                    T.reads(acc_s[0:2, 0:2])
                    T.writes(acc_s_cast[0:2, 0:2])
                    T.ppl_copy(acc_s, acc_s_cast)
                with T.block("load_v"):
                    T.reads(V[k * 2:k * 2 + 2, 0:8])
                    T.writes(V_shared[0:2, 0:8])
                    T.ppl_copy(V[k * 2, 0], V_shared)
                with T.block("output_gemm"):
                    T.reads(acc_s_cast[0:2, 0:2], V_shared[0:2, 0:8], acc_o[0:2, 0:8])
                    T.writes(acc_o[0:2, 0:8])
                    T.ppl_gemm(acc_s_cast, V_shared, acc_o)

            T.ppl_copy(acc_o, O)

    fd, path = tempfile.mkstemp(prefix="tl_tpu_attention_windows_", suffix=".txt")
    os.close(fd)
    old_value = os.environ.get("TL_TPU_PIPELINE_CERTIFICATE")
    old_materializer = os.environ.get("TL_TPU_ATTENTION_WINDOW_B_MATERIALIZE")
    os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = path
    os.environ["TL_TPU_ATTENTION_WINDOW_B_MATERIALIZE"] = "1"
    try:
        mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.BindTarget(tpu_target)(mod)
        mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        planned = tl.transform.PipelinePlanning()(mod)
        injected = tl.transform.InjectSoftwarePipeline()(planned)
        with open(path, "r", encoding="utf-8") as f:
            certificate = f.read()
    finally:
        if old_value is None:
            os.environ.pop("TL_TPU_PIPELINE_CERTIFICATE", None)
        else:
            os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = old_value
        if old_materializer is None:
            os.environ.pop("TL_TPU_ATTENTION_WINDOW_B_MATERIALIZE", None)
        else:
            os.environ["TL_TPU_ATTENTION_WINDOW_B_MATERIALIZE"] = old_materializer
        if os.path.exists(path):
            os.remove(path)

    assert "policy_family: attention_mixed" in certificate, certificate
    assert "overlap_windows:" in certificate, certificate
    assert "name: WindowA_K_prefetch_next_iter" in certificate, certificate
    assert "status: candidate_needs_split_or_versioning" in certificate, certificate
    assert "buffer: K_shared" in certificate, certificate
    assert "versions: 2" in certificate, certificate
    assert "name: WindowB_V_prefetch_same_iter" in certificate, certificate
    assert "status: selected_for_materialization" in certificate, certificate
    assert "materialization_kind: parallel_marker_inside_iteration" in certificate, certificate
    assert "attention_materialization_plan:" in certificate, certificate
    assert "status: selected" in certificate, certificate
    assert "load_statement: 5" in certificate, certificate
    assert "overlap_statements: [3, 4]" in certificate, certificate
    injected_text = str(injected)
    assert "tpu_parallel_start" in injected_text, injected_text
    assert "tpu_parallel_end" in injected_text, injected_text
    assert "tl_tpu_attention_window_b" not in injected_text, injected_text


def test_tpu_attention_window_b_skips_fp32_local_load():

    @T.prim_func
    def before(K: T.Tensor((4, 8), "float32"), V: T.Tensor((4, 8), "float32"), O: T.Tensor(
        (2, 8), "float32")):
        with T.Kernel(1, is_cpu=True):
            Q_shared = T.alloc_shared((2, 8), "float32")
            K_shared = T.alloc_shared((2, 8), "float32")
            V_shared = T.alloc_shared((2, 8), "float32")
            acc_s = T.alloc_shared((2, 2), "float32")
            acc_s_cast = T.alloc_shared((2, 2), "float32")
            acc_o = T.alloc_shared((2, 8), "float32")
            scores_sum = T.alloc_shared((2, 1), "float32")

            T.ppl_fill(acc_o, T.float32(0))
            for k in T.Pipelined(2, num_stages=1):
                with T.block("load_k"):
                    T.reads(K[k * 2:k * 2 + 2, 0:8])
                    T.writes(K_shared[0:2, 0:8])
                    T.ppl_copy(K[k * 2, 0], K_shared)
                with T.block("fill_scores"):
                    T.reads()
                    T.writes(acc_s[0:2, 0:2])
                    T.ppl_fill(acc_s, T.float32(0))
                with T.block("score_gemm"):
                    T.reads(Q_shared[0:2, 0:8], K_shared[0:2, 0:8], acc_s[0:2, 0:2])
                    T.writes(acc_s[0:2, 0:2])
                    T.ppl_gemm(Q_shared, K_shared, acc_s, transpose_B=True)
                with T.block("softmax_reduce"):
                    T.reads(acc_s[0:2, 0:2])
                    T.writes(scores_sum[0:2, 0:1])
                    T.ppl_reduce_sum(acc_s, scores_sum, dim=1)
                with T.block("cast_scores"):
                    T.reads(acc_s[0:2, 0:2])
                    T.writes(acc_s_cast[0:2, 0:2])
                    T.ppl_copy(acc_s, acc_s_cast)
                with T.block("load_v"):
                    T.reads(V[k * 2:k * 2 + 2, 0:8])
                    T.writes(V_shared[0:2, 0:8])
                    T.ppl_copy(V[k * 2, 0], V_shared)
                with T.block("output_gemm"):
                    T.reads(acc_s_cast[0:2, 0:2], V_shared[0:2, 0:8], acc_o[0:2, 0:8])
                    T.writes(acc_o[0:2, 0:8])
                    T.ppl_gemm(acc_s_cast, V_shared, acc_o)

            T.ppl_copy(acc_o, O)

    fd, path = tempfile.mkstemp(prefix="tl_tpu_attention_fp32_window_", suffix=".txt")
    os.close(fd)
    old_value = os.environ.get("TL_TPU_PIPELINE_CERTIFICATE")
    old_materializer = os.environ.get("TL_TPU_ATTENTION_WINDOW_B_MATERIALIZE")
    os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = path
    os.environ["TL_TPU_ATTENTION_WINDOW_B_MATERIALIZE"] = "1"
    try:
        mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.BindTarget(tpu_target)(mod)
        mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        planned = tl.transform.PipelinePlanning()(mod)
        injected = tl.transform.InjectSoftwarePipeline()(planned)
        with open(path, "r", encoding="utf-8") as f:
            certificate = f.read()
    finally:
        if old_value is None:
            os.environ.pop("TL_TPU_PIPELINE_CERTIFICATE", None)
        else:
            os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = old_value
        if old_materializer is None:
            os.environ.pop("TL_TPU_ATTENTION_WINDOW_B_MATERIALIZE", None)
        else:
            os.environ["TL_TPU_ATTENTION_WINDOW_B_MATERIALIZE"] = old_materializer
        if os.path.exists(path):
            os.remove(path)

    assert "policy_family: attention_mixed" in certificate, certificate
    assert "name: WindowB_V_prefetch_same_iter" in certificate, certificate
    assert "status: candidate_legal_certificate_only" in certificate, certificate
    assert "attention_materialization_plan:" in certificate, certificate
    assert "status: not_selected" in certificate, certificate
    injected_text = str(injected)
    assert "tpu_parallel_start" in injected_text, injected_text
    assert "tpu_parallel_end" in injected_text, injected_text
    assert "tl_tpu_attention_window_b" not in injected_text, injected_text


def test_tpu_reduction_phase_model_certificate():

    @T.prim_func
    def before(A: T.Tensor((32, 64), "float16"), B: T.Tensor((32, 1), "float32")):
        with T.Kernel(1, is_cpu=True):
            A_shared = T.alloc_shared((32, 32), "float16")
            A_shared_fp32 = T.alloc_shared((32, 32), "float32")
            A_pow2 = T.alloc_shared((32, 32), "float32")
            A_powsum = T.alloc_shared((32, 1), "float32")
            A_temp = T.alloc_shared((32, 1), "float32")

            T.ppl_fill(A_powsum, T.float32(0))
            for k in T.Pipelined(2, num_stages=2):
                T.ppl_copy(A[0, k * 32], A_shared)
                T.ppl_copy(A_shared, A_shared_fp32)
                T.ppl_mul(A_pow2, A_shared_fp32, A_shared_fp32)
                T.ppl_reduce_sum(A_pow2, A_temp, dim=1)
                T.ppl_add(A_powsum, A_powsum, A_temp)

            T.ppl_copy(A_powsum, B)

    fd, path = tempfile.mkstemp(prefix="tl_tpu_reduction_phase_", suffix=".txt")
    os.close(fd)
    old_value = os.environ.get("TL_TPU_PIPELINE_CERTIFICATE")
    os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = path
    try:
        mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.BindTarget(tpu_target)(mod)
        mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        planned = tl.transform.PipelinePlanning()(mod)
        injected = tl.transform.InjectSoftwarePipeline()(planned)
        with open(path, "r", encoding="utf-8") as f:
            certificate = f.read()
    finally:
        if old_value is None:
            os.environ.pop("TL_TPU_PIPELINE_CERTIFICATE", None)
        else:
            os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = old_value
        if os.path.exists(path):
            os.remove(path)

    planned_text = str(planned)
    injected_text = str(injected)
    assert "optimizer_decision: tpu_stage_search_exact" in certificate, certificate
    assert "pipeline_policy: reduction_norm" in certificate, certificate
    assert "policy_family: reduction_norm" in certificate, certificate
    assert "evidence_tier: tier_2_family_model_local" in certificate, certificate
    assert "policy_stage_optimization: enabled" in certificate, certificate
    assert "policy_buffer_version_hints: enabled" in certificate, certificate
    assert "policy_stage_aware_lmem: disabled" in certificate, certificate
    assert "profitability_guard:" in certificate, certificate
    assert "status: legality_and_materialization_only" in certificate, certificate
    assert "recommendation: require_serialized_runtime_evidence_before_speedup_claim" in certificate, certificate
    assert "engine_timeline:" in certificate, certificate
    assert "op: ppl.reduce_sum" in certificate, certificate
    assert "phase: reduce_h" in certificate, certificate
    assert "phase: reduce_w" in certificate, certificate
    assert "hidden_workspace_lmem:" in certificate, certificate
    assert "tmp_buffer_sum" in certificate, certificate
    assert "total_bm1690_tpu_align_bytes: 128" in certificate, certificate
    assert "loop_carried_dependencies:" in certificate, certificate
    assert "buffer: A_powsum" in certificate, certificate
    assert "tpu_cost_model:" in certificate, certificate
    assert "status: active" in certificate, certificate
    assert "local_copy_bytes: 4096" in certificate, certificate
    assert "bdc_ops: 1056" in certificate, certificate
    assert "unknown_static_cost: true" in certificate, certificate
    assert "bdc_ops=1024" in certificate, certificate
    assert "bdc_ops=32" in certificate, certificate
    assert '"software_pipeline_stage": [0, 1, 1, 1]' in planned_text, planned_text
    assert "tl_pipeline_buffer_versions" in planned_text, planned_text
    assert '"A_shared": 2' in planned_text, planned_text
    assert "A_shared_0" in injected_text, injected_text
    assert "A_shared_1" in injected_text, injected_text


def test_tpu_reduction_outer_alloc_buffer_version_lowering():

    @T.prim_func
    def before(A: T.Tensor((64, 512), "float16"), B: T.Tensor((64, 512), "float16")):
        reciprocal_N = T.float32(1.0 / 512)

        with T.Kernel(2, is_cpu=True) as (bx,):
            A_shared = T.alloc_shared((32, 32), "float16")
            A_shared_fp32 = T.alloc_shared((32, 32), "float32")
            A_pow2 = T.alloc_shared((32, 32), "float32")
            A_powsum = T.alloc_shared((32, 1), "float32")
            A_temp = T.alloc_shared((32, 1), "float32")

            T.ppl_fill(A_powsum, T.float32(0.0))
            num_k_step = T.ceildiv(512, 32)

            for k in T.Pipelined(num_k_step, num_stages=2):
                T.ppl_copy(A[bx * 32, k * 32], A_shared)
                T.ppl_copy(A_shared, A_shared_fp32)
                T.ppl_mul(A_pow2, A_shared_fp32, A_shared_fp32)
                T.ppl_reduce_sum(A_pow2, A_temp, dim=1)
                T.ppl_add(A_powsum, A_powsum, A_temp)

            T.ppl_mul_C(A_powsum, A_powsum, reciprocal_N)
            T.ppl_add_C(A_powsum, A_powsum, T.float32(1e-12))
            T.ppl_rsqrt(A_powsum, A_powsum)

            for k in T.Pipelined(num_k_step, num_stages=0):
                block_k_idx = num_k_step - 1 - k
                T.ppl_copy(A[bx * 32, block_k_idx * 32], A_shared)
                T.ppl_copy(A_shared, A_shared_fp32)
                T.ppl_mul(A_shared_fp32, A_shared_fp32, A_powsum)
                T.ppl_copy(A_shared_fp32, A_shared)
                T.ppl_copy(A_shared, B[bx * 32, block_k_idx * 32])

    fd, path = tempfile.mkstemp(prefix="tl_tpu_reduction_outer_alloc_", suffix=".txt")
    os.close(fd)
    old_value = os.environ.get("TL_TPU_PIPELINE_CERTIFICATE")
    os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = path
    try:
        artifact = tl.lower(before, target="tpu")
        with open(path, "r", encoding="utf-8") as f:
            certificate = f.read()
    finally:
        if old_value is None:
            os.environ.pop("TL_TPU_PIPELINE_CERTIFICATE", None)
        else:
            os.environ["TL_TPU_PIPELINE_CERTIFICATE"] = old_value
        if os.path.exists(path):
            os.remove(path)

    kernel_source = str(artifact.kernel_source)
    assert "optimizer_decision: tpu_stage_search_exact" in certificate, certificate
    assert "pipeline_policy: reduction_norm" in certificate, certificate
    assert "buffer: A_shared" in certificate, certificate
    assert "required_versions: 2" in certificate, certificate
    assert "bdc_ops: 2112" in certificate, certificate
    assert "unknown_static_cost: false" in certificate, certificate
    assert "A_shared_0" in kernel_source, kernel_source
    assert "A_shared_1" in kernel_source, kernel_source
    assert "A_shared_pipe" in kernel_source, kernel_source


if __name__ == "__main__":
    if tilelang_testing is not None:
        tilelang_testing.main()
    else:
        for name, value in sorted(globals().items()):
            if name.startswith("test_") and callable(value):
                print(f"RUN {name}")
                value()
                print(f"PASS {name}")
