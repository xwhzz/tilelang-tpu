# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
from contextlib import contextmanager
from typing import List, Union, Any, Callable, Literal, Optional, Dict
from tvm.target import Target
import tilelang
from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm import tir

from tilelang.jit.adapter import (
    TorchDLPackKernelAdapter,
    BaseKernelAdapter,
    CtypesKernelAdapter,
    CythonKernelAdapter,
)
from tilelang.utils.target import determine_target, AVALIABLE_TARGETS
from tilelang.profiler import Profiler, TensorSupplyType
from tilelang.engine.param import KernelParam, CompiledArtifact


def _normalize_tpu_core_parallel(config):
    """Normalize TPU core-parallel compile option.

    Supported forms:
      None  -> disabled
      bool  -> enable/disable with default loop/core_num
      dict  -> {"enable": True, "loop": "by", "core_num": 8}
            or {"enable": True, "core_num": "auto", "max_core_num": 8}
    """
    if config is None:
        return {
            "enable": False,
            "loop": "by",
            "core_num": 8,
            "max_core_num": 8,
        }

    if isinstance(config, bool):
        return {
            "enable": config,
            "loop": "by",
            "core_num": 8,
            "max_core_num": 8,
        }

    if isinstance(config, dict):
        core_num = config.get("core_num", 8)

        if isinstance(core_num, str):
            if core_num != "auto":
                raise ValueError(
                    'tpu_core_parallel["core_num"] string only supports "auto"'
                )
        else:
            core_num = int(core_num)

        normalized = {
            "enable": bool(config.get("enable", True)),
            "loop": str(config.get("loop", "by")),
            "core_num": core_num,
            "max_core_num": int(config.get("max_core_num", 8)),
        }

        if "_resolved_core_num" in config:
            normalized["_resolved_core_num"] = int(config["_resolved_core_num"])

        return normalized

    raise TypeError(
        "tpu_core_parallel must be None, bool, or dict, "
        'e.g. {"enable": True, "loop": "by", "core_num": 8}'
    )

def _try_get_static_int(value):
    """Best-effort conversion from a TIR expression to a Python int."""
    if value is None:
        return None

    if isinstance(value, int):
        return int(value)

    try:
        analyzer = tvm.arith.Analyzer()
        value = analyzer.simplify(value)
    except Exception:
        pass

    if isinstance(value, tir.IntImm):
        return int(value.value)

    if hasattr(value, "value"):
        try:
            return int(value.value)
        except Exception:
            return None

    try:
        return int(value)
    except Exception:
        return None


def _has_tpu_core_parallel_annotation(for_node):
    """Check whether a TIR For node is marked as TPU core-parallel."""
    annotations = getattr(for_node, "annotations", None)
    if annotations is None:
        return False

    try:
        if "tilelang.tpu_core_parallel" in annotations:
            return True
    except Exception:
        pass

    try:
        for key in annotations.keys():
            if str(key) == "tilelang.tpu_core_parallel":
                return True
    except Exception:
        pass

    return False


def _infer_tpu_core_num_from_tir(func, max_core_num=8):
    """Infer TPU core number from the annotated parallel loop extent.

    Short-term rule:
      core_num = min(max_core_num, parallel_loop_extent)

    This version only targets static loop extents, such as:
      for m_blk in T.serial(T.ceildiv(M, block_M),
                            annotations={"tilelang.tpu_core_parallel": True})
    """
    inferred_extent = None

    def visit(stmt):
        nonlocal inferred_extent

        if inferred_extent is not None:
            return

        if isinstance(stmt, tir.For) and _has_tpu_core_parallel_annotation(stmt):
            extent = _try_get_static_int(stmt.extent)
            if extent is not None and extent > 0:
                inferred_extent = extent

    try:
        if isinstance(func, tir.PrimFunc):
            tir.stmt_functor.post_order_visit(func.body, visit)
        elif isinstance(func, tvm.IRModule):
            for _, prim_func in func.functions.items():
                if isinstance(prim_func, tir.PrimFunc):
                    tir.stmt_functor.post_order_visit(prim_func.body, visit)
                    if inferred_extent is not None:
                        break
        elif hasattr(func, "body"):
            tir.stmt_functor.post_order_visit(func.body, visit)
    except Exception:
        inferred_extent = None

    if inferred_extent is None:
        return int(max_core_num)

    return max(1, min(int(max_core_num), int(inferred_extent)))

@contextmanager
def _temporary_tpu_core_parallel_env(config):
    """Temporarily map compile option to environment variables.

    The current TPU PPL codegen and host template still read:
      TL_TPU_ENABLE_CORE_PARALLEL
      TL_TPU_CORE_PARALLEL_LOOP
      TL_TPU_CORE_NUM

    This context manager allows tilelang.compile(...) to control them
    without requiring users to export shell environment variables manually.
    """
    config = _normalize_tpu_core_parallel(config)

    keys = [
        "TL_TPU_ENABLE_CORE_PARALLEL",
        "TL_TPU_CORE_PARALLEL_LOOP",
        "TL_TPU_CORE_NUM",
    ]
    old_values = {key: os.environ.get(key) for key in keys}

    try:
        if config["enable"]:
            os.environ["TL_TPU_ENABLE_CORE_PARALLEL"] = "1"
            os.environ["TL_TPU_CORE_PARALLEL_LOOP"] = config["loop"]

            core_num = config.get("_resolved_core_num", config.get("core_num", 8))
            if core_num == "auto":
                core_num = config.get("max_core_num", 8)

            os.environ["TL_TPU_CORE_NUM"] = str(int(core_num))
        else:
            os.environ.pop("TL_TPU_ENABLE_CORE_PARALLEL", None)
            os.environ.pop("TL_TPU_CORE_PARALLEL_LOOP", None)
            os.environ.pop("TL_TPU_CORE_NUM", None)

        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class JITKernel(object):
    """
    A wrapper class for compiling and invoking TileLang (TVM TIR) functions as PyTorch-compatible functions.

    Attributes
    ----------
    artifact : CompiledArtifact
        The compiled artifact containing the runtime module and parameters.
    adapter : BaseKernelAdapter
        The adapter for the compiled function.
    torch_function : Callable
        The compiled function that can be invoked as a PyTorch-compatible function.
    """

    artifact: CompiledArtifact = None
    adapter: BaseKernelAdapter = None
    torch_function: Callable = None

    def __init__(
        self,
        func: PrimFunc = None,
        out_idx: Union[List[int], int] = None,
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        verbose: bool = False,
        pass_configs: Optional[Dict[str, Any]] = None,
        from_database: bool = False,
        mode: Literal["pcie", "cmodel"] = "pcie",
        tpu_core_parallel: Optional[Union[bool, Dict[str, Any]]] = None,
        tpu_device_id: Optional[int] = None,
        tpu_runtime_root: Optional[str] = None,
    ):
        """
        Initializes a TorchFunction instance.

        Parameters
        ----------
        func : tvm.tir.PrimFunc, optional
            The TileLang TIR function to compile and wrap.
        out_idx : Union[List[int], int], optional
            Index(es) of the output tensors to return (default: None).
        execution_backend : Literal["dlpack", "ctypes"], optional
            Execution backend to use for kernel execution (default: "dlpack").
        target : Union[str, Target], optional
            Compilation target, either as a string or a TVM Target object (default: "auto").
        target_host : Union[str, Target], optional
            Target host for cross-compilation (default: None).
        verbose : bool, optional
            Whether to enable verbose output (default: False).
        pass_configs : dict, optional
            Additional keyword arguments to pass to the Compiler PassContext.
            Available options:
                "tir.disable_vectorize": bool, default: False
                "tl.disable_tma_lower": bool, default: False
        from_database : bool, optional
            Whether to create a TorchFunction from a database.
        """
        self.execution_backend = execution_backend
        self.target = target
        self.target_host = target_host
        self.verbose = verbose

        if pass_configs is None:
            pass_configs = {}
        self.pass_configs = pass_configs
        self.mode = mode
        self.tpu_core_parallel = _normalize_tpu_core_parallel(tpu_core_parallel)

        if (
            self.tpu_core_parallel["enable"]
            and self.tpu_core_parallel["core_num"] == "auto"
        ):
            self.tpu_core_parallel["_resolved_core_num"] = _infer_tpu_core_num_from_tir(
                func,
                max_core_num=self.tpu_core_parallel["max_core_num"],
            )

        if tpu_device_id is None:
            tpu_device_id = int(os.environ.get("TL_TPU_DEVICE_ID", "0"))
        if tpu_device_id < 0:
            raise ValueError("tpu_device_id must be non-negative")
        self.tpu_device_id = int(tpu_device_id)
        self.tpu_runtime_root = tpu_runtime_root
        # If the target is specified as a string, validate it and convert it to a TVM Target.
        if isinstance(target, str):
            assert target in AVALIABLE_TARGETS, f"Invalid target: {target}"
            target = determine_target(target)

        # Ensure the target is always a TVM Target object.
        target = Target(target)

        # Validate the execution backend.
        assert execution_backend in [
            "dlpack",
            "ctypes",
            "cython",
        ], f"Invalid execution backend. {execution_backend}"

        assert mode in [
            "pcie",
            "cmodel",
        ], f"Invalid execution mode. {mode}"

        if execution_backend == "cython":
            from tilelang.contrib.cc import get_cplus_compiler

            assert (
                get_cplus_compiler() is not None
            ), "Cython backend requires a C++ compiler, please install or use other backends."

        if from_database:
            return

        # Compile the TileLang function and create a kernel adapter for execution.
        adapter = self._compile_and_create_adapter(func, out_idx)

        # The adapter's function is assigned as the callable function for this instance.
        self.adapter = adapter
        self.torch_function = adapter.func

    @classmethod
    def from_database(
        cls,
        func: PrimFunc,
        kernel_global_source: str,
        kernel_lib_path: str,
        params: List[KernelParam],
        target: Union[str, Target],
        target_host: Union[str, Target],
        out_idx: Union[List[int], int],
        execution_backend: Literal["dlpack", "ctypes", "cython"],
        pass_configs: Optional[Dict[str, Any]] = None,
        mode: Literal["pcie", "cmodel"] = "pcie",
        tpu_core_parallel: Optional[Union[bool, Dict[str, Any]]] = None,
        tpu_device_id: Optional[int] = None,
        tpu_runtime_root: Optional[str] = None,
    ):
        """
        Alternative constructor to create a TorchFunction directly from a database.
        """
        instance = cls(
            func=func,
            out_idx=out_idx,
            execution_backend=execution_backend,
            target=target,
            target_host=target_host,
            pass_configs=pass_configs,
            from_database=True,
            mode=mode,
            tpu_core_parallel=tpu_core_parallel,
            tpu_device_id=tpu_device_id,
            tpu_runtime_root=tpu_runtime_root,
        )

        instance.adapter = instance._create_adapter_from_database(
            func_or_mod=func,
            params=params,
            result_idx=out_idx,
            target=target,
            kernel_global_source=kernel_global_source,
            kernel_lib_path=kernel_lib_path,
        )
        instance.torch_function = instance.adapter.func
        return instance

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Invokes the compiled function with the given arguments.

        Parameters
        ----------
        *args : Any
            Positional arguments for the function.
        **kwds : Any
            Keyword arguments for the function.

        Returns
        -------
        Any
            The result of the function execution.
        """
        return self.torch_function(*args, **kwds)

    def _compile_and_create_adapter(self, tilelang_func: PrimFunc,
                                    out_idx: List[int]) -> BaseKernelAdapter:
        """
        Compiles the given TileLang PrimFunc using TVM and creates a kernel adapter.

        Parameters
        ----------
        tilelang_func : tvm.tir.PrimFunc
            The TileLang (TVM TIR) function to compile.

        Returns
        -------
        BaseKernelAdapter
            The compiled and ready-to-run kernel adapter.
        """
        verbose = self.verbose
        target = self.target
        target_host = self.target_host

        execution_backend = self.execution_backend
        pass_configs = self.pass_configs

        mode = self.mode

        # Compile the function with TVM, optimizing with shared memory lowering.
        enable_host_codegen = execution_backend == "dlpack"
        enable_device_compile = execution_backend == "dlpack"
        with _temporary_tpu_core_parallel_env(self.tpu_core_parallel):
            with tvm.transform.PassContext(opt_level=3, config=pass_configs):
                artifact = tilelang.lower(
                    tilelang_func,
                    target=target,
                    target_host=target_host,
                    enable_host_codegen=enable_host_codegen,
                    enable_device_compile=enable_device_compile)

        self.artifact = artifact

        # Create an adapter based on the specified execution backend.
        if execution_backend == "dlpack":
            # Use TorchDLPackKernelAdapter for interoperability with PyTorch via DLPack.
            # But we need to ensure that the runtime is enabled and the runtime module is not None.
            assert tvm.runtime.enabled("llvm"), "DLPack backend requires LLVM runtime."
            assert (artifact.rt_mod is not None), "DLPack backend requires a runtime module."
            adapter = TorchDLPackKernelAdapter(
                artifact.rt_mod, params=artifact.params, result_idx=out_idx)
        elif execution_backend == "ctypes":
            adapter = CtypesKernelAdapter(
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                kernel_global_source=artifact.kernel_source,
                verbose=verbose,
                pass_configs=pass_configs,
            )
        elif execution_backend == "cython":
            adapter = CythonKernelAdapter(
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                kernel_global_source=artifact.kernel_source,
                verbose=verbose,
                pass_configs=pass_configs,
                mode=mode,
                tpu_core_parallel=self.tpu_core_parallel,
                tpu_device_id=self.tpu_device_id,
                tpu_runtime_root=self.tpu_runtime_root,
            )
        else:
            # Handle invalid backend.
            raise ValueError(f"Invalid execution backend: {execution_backend}")

        return adapter

    def _create_adapter_from_database(
        self,
        params: List[KernelParam],
        result_idx: Union[List[int], int],
        target: Union[str, Target],
        func_or_mod: Union[PrimFunc, tvm.runtime.Module],
        kernel_global_source: str,
        kernel_lib_path: str,
    ) -> BaseKernelAdapter:
        target = self.target
        execution_backend = self.execution_backend

        # Create an adapter based on the specified execution backend.
        if execution_backend == "dlpack":
            raise ValueError("DLPack backend is not supported for TileLang JIT.")
        elif execution_backend == "ctypes":
            adapter = CtypesKernelAdapter.from_database(
                params=params,
                result_idx=result_idx,
                target=target,
                func_or_mod=func_or_mod,
                kernel_global_source=kernel_global_source,
                kernel_lib_path=kernel_lib_path,
            )
        elif execution_backend == "cython":
            adapter = CythonKernelAdapter.from_database(
                params=params,
                result_idx=result_idx,
                target=target,
                func_or_mod=func_or_mod,
                kernel_global_source=kernel_global_source,
                kernel_lib_path=kernel_lib_path,
            )
        else:
            # Handle invalid backend.
            raise ValueError(f"Invalid execution backend: {execution_backend}")

        return adapter

    @classmethod
    def from_tilelang_function(cls, tilelang_func: PrimFunc, **kwargs):
        """
        Alternative constructor to create a TorchFunction directly from a TileLang PrimFunc.

        Parameters
        ----------
        tilelang_func : tvm.tir.PrimFunc
            The TileLang (TVM TIR) function to compile.
        **kwargs : dict
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        TorchFunction
            An instance of TorchFunction wrapping the compiled function.
        """
        return cls(func=tilelang_func, **kwargs)

    def get_profiler(self,
                     tensor_supply_type: TensorSupplyType = TensorSupplyType.Auto) -> Profiler:
        """
        Creates a profiler to benchmark the compiled runtime module.

        Parameters
        ----------
        tensor_supply_type : TensorSupplyType, optional
            The type of input tensors to supply for profiling (default: TensorSupplyType.Auto).

        Returns
        -------
        Profiler
            A Profiler instance for benchmarking the runtime module.
        """
        return Profiler(self.params, self.out_idx,
                        tensor_supply_type).with_default_adapter(self.adapter)

    def get_kernel_source(self) -> str:
        """
        Returns the source code of the compiled kernel function.

        Returns
        -------
        str
            The source code of the compiled kernel function.
        """
        if self.execution_backend in {"ctypes", "cython"}:
            return self.adapter.get_kernel_source()
        return self.artifact.kernel_source

    def get_host_source(self) -> str:
        """
        Returns the source code of the host function.
        """
        return str(self.artifact.host_mod)

    def run_once(self, func: Optional[Callable] = None) -> None:
        return self.get_profiler().run_once(func)

    @property
    def out_idx(self) -> List[int]:
        return self.adapter.result_idx

    @property
    def params(self) -> List[KernelParam]:
        return self.artifact.params if self.artifact else self.adapter.params

    @property
    def kernel_source(self) -> str:
        return self.artifact.kernel_source if self.artifact else self.adapter.kernel_global_source

    @property
    def host_source(self) -> str:
        return str(self.artifact.host_mod) if self.artifact else ""

    def export_library(self, kernel_file: str) -> None:
        """
        Exports the compiled kernel function to a shared library file.

        Parameters
        ----------
        kernel_file : str
            The path to the shared library file to create.
        """
        # rt_module: tvm.runtime.Module = None
        # rt_params: dict = None
        # adapter: BaseKernelAdapter = None
        # torch_function: Callable = None
        # rt_module: use export_library to export
        # rt_params: use cloudpickle to serialize

        # Export the compiled kernel function to a shared library file.
        self.rt_module.export_library(kernel_file)
