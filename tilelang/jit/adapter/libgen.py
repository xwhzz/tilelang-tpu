# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from typing import Optional, Literal
from .utils import is_cuda_target, is_hip_target, is_cpu_target, is_tpu_target
from tilelang import tvm as tvm
from tilelang.contrib.nvcc import get_target_compute_version
from tvm.target import Target
import ctypes
import os
import tempfile
import subprocess
import logging
from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.jit.adapter.utils import get_tpu_template_dir

logger = logging.getLogger(__name__)


class LibraryGenerator(object):
    srcpath: Optional[str] = None
    libpath: Optional[str] = None
    lib_code: Optional[str] = None
    mode: Literal["pcie", "cmodel"] = "pcie"

    def __init__(self, target: Target, mode: Literal["pcie", "cmodel"] = "pcie"):
        self.target = target
        self.mode = mode

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    # Assume currently we only support CUDA compilation
    def load_lib(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self.libpath
        return ctypes.CDLL(lib_path)

    def compile_lib(self, timeout: float = None, with_tl: bool = True):
        target = self.target
        mode = self.mode
        if is_cuda_target(target):
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)
            compute_version = "".join(get_target_compute_version(target).split("."))
            if compute_version == "90":
                compute_version = "90a"
            libpath = src.name.replace(".cu", ".so")

            command = [
                "nvcc",
                "-std=c++17",
                "-w",  # Disable all warning messages
                "-Xcudafe",
                "--diag_suppress=177",
                "--compiler-options",
                "'-fPIC'",
                "-lineinfo",
                "--shared",
                src.name,
                "-lcuda",
                "-gencode",
                f"arch=compute_{compute_version},code=sm_{compute_version}",
            ]

        elif is_hip_target(target):
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
            libpath = src.name.replace(".cpp", ".so")

            command = [
                "hipcc",
                "-std=c++17",
                "-fPIC",
                "--shared",
                src.name,
            ]
        elif is_cpu_target(target):
            from tilelang.contrib.cc import get_cplus_compiler
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
            libpath = src.name.replace(".cpp", ".so")

            command = [get_cplus_compiler(), "-std=c++17", "-fPIC", "-shared", src.name]
            with_tl = False
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
            ]
        elif is_tpu_target(target):

            src = tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False)
            libpath = src.name.replace(".c", ".so")


            import os
            # 设置环境变量
            PPL_TOP = os.environ.get("PPL_PROJECT_ROOT", None)
            if not PPL_TOP:
                raise EnvironmentError("PPL_PROJECT_ROOT environment variable is not set.")
            CHIP = "bm1690"

            if mode=="pcie":
                self.tpu_compile_pcie(timeout=timeout, PPL_TOP=PPL_TOP, CHIP=CHIP)
            elif mode=="cmodel":
                self.tpu_compile_cmodel(timeout=timeout, PPL_TOP=PPL_TOP, CHIP=CHIP)
            else:
                raise ValueError(f"Unsupported compile mode: {mode}")
            self.srcpath = src.name
            self.libpath = f"{get_tpu_template_dir()}/main.so"
            return

        else:
            raise ValueError(f"Unsupported target: {target}")


        if with_tl:
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
                "-I" + CUTLASS_INCLUDE_DIR,
            ]
            command += ["-diag-suppress=20013"]
        command += ["-o", libpath]

        src.write(self.lib_code)
        src.flush()
        try:
            ret = subprocess.run(command, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret.returncode != 0:
            raise RuntimeError(f"Compilation Failed! {command}")

        self.srcpath = src.name
        self.libpath = libpath

    def remove_lib(self):
        if self.libpath:
            os.remove(self.libpath)
        self.libpath = None

    def get_source_path(self):
        return self.srcpath

    def get_lib_path(self):
        return self.libpath

    def set_lib_path(self, libpath):
        self.libpath = libpath

    def set_src_path(self, srcpath):
        self.srcpath = srcpath

    def tpu_compile_pcie(self, timeout, PPL_TOP, CHIP):
        TOOLCHAIN_DIR = f"{PPL_TOP}/third_party/toolchains_dir/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1"
        CROSS_COMPILE = f"{TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-"

        # 构建包含路径
        includes = [
            # "-I/lib/x86_64-linux-gnu/",
            "-I./build/include",
            f"-I{PPL_TOP}/runtime/{CHIP}/TPU1686/kernel/include",
            f"-I{PPL_TOP}/runtime/kernel",
            f"-I{PPL_TOP}/runtime/customize/include",
            f"-I{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/include"
        ]

        # 构建库路径
        lib_paths = [
            # "-L/lib/x86_64-linux-gnu/",
            f"-L{PPL_TOP}/runtime/{CHIP}/lib",
            "-L/opt/tpuv7/tpuv7-current/lib/",
            f"-L{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib"
        ]
        src_dir = get_tpu_template_dir()

        # 编译kernel.c
        cmd1 = [
            f"{CROSS_COMPILE}gcc",
            "-D__bm1690__",
            "-Dlibkernel_EXPORTS",
            *includes,
            "-Wl,--no-undefined",
            "-fPIC",
            "-c",
            f"{src_dir}/kernel.c",
            "-o",
            f"{src_dir}/kernel.o"
        ]

        # 编译ppl_helper.c
        cmd2 = [
            f"{CROSS_COMPILE}gcc", 
            "-D__bm1690__",
            "-Dlibkernel_EXPORTS",
            *includes,
            "-Wl,--no-undefined",
            "-fPIC",
            "-c",
            f"{PPL_TOP}/runtime/customize/src/ppl_helper.c",
            "-o",
            f"{src_dir}/ppl_helper.o"
        ]

        # 链接命令 - 创建共享库
        link_cmd = [
            f"{CROSS_COMPILE}gcc",
            "-fPIC",
            "-Wl,--no-undefined", 
            "-shared",
            "-Wl,-soname,libkernel.so",
            "-o", f"{src_dir}/libkernel.so",
            f"{src_dir}/kernel.o",
            f"{src_dir}/ppl_helper.o",
            *lib_paths,
            "-Wl,-rpath," + f"{PPL_TOP}/runtime/{CHIP}/lib:{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib",
            "-Wl,--whole-archive",
            "-Wl,-Bstatic",
            f"-l{CHIP}",
            "-Wl,-Bdynamic", 
            "-Wl,--no-whole-archive",
            "-lm"
        ]

        try:
            ret1 = subprocess.run(cmd1, timeout=timeout)
            ret2 = subprocess.run(cmd2, timeout=timeout)
            ret3 = subprocess.run(link_cmd, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret1.returncode != 0 or ret2.returncode != 0 or ret3.returncode != 0:
            raise RuntimeError(f"Compilation Failed! {link_cmd}")



        
        # 1. 编译main.cpp
        cmd4 = [
            "g++",
            f"-D__{CHIP}__",
            *includes,
            "-Wl,--no-undefined",
            "-std=c++11",
            "-fPIC",
            "-c",
            f"{src_dir}/kernel.cpp",
            "-o",
            f"{src_dir}/kernel_host.o"
        ]
        
        # 2. 编译main.cpp
        cmd5 = [
            "g++",
            f"-D__{CHIP}__", 
            *includes,
            "-Wl,--no-undefined",
            "-std=c++11",
            "-fPIC", 
            "-c",
            f"{src_dir}/main.cpp",
            "-o",
            f"{src_dir}/main.o"
        ]
        
        # 3. 生成动态库
        cmd_shared = [
            "g++",
            "-shared",
            "-fPIC",
            "-Wl,--no-undefined",
            "-o",
            f"{src_dir}/main.so",
            f"{src_dir}/kernel_host.o",
            f"{src_dir}/main.o",
            *lib_paths,
            "-Wl,-rpath," + f"{PPL_TOP}/runtime/{CHIP}/lib:{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib",
            "-ltpuv7_rt",
            "-lcdm_daemon_emulator", 
            "-lpthread"
        ]

        try:
            ret1 = subprocess.run(cmd4, timeout=timeout)
            ret2 = subprocess.run(cmd5, timeout=timeout)
            ret3 = subprocess.run(cmd_shared, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret1.returncode != 0 or ret2.returncode != 0 or ret3.returncode != 0:
            raise RuntimeError(f"Host Compilation Failed! {cmd_shared}")


    def tpu_compile_cmodel(self, PPL_TOP, CHIP, timeout):
        src_dir = get_tpu_template_dir()
        KERNEL_C = f"{src_dir}/kernel.c"
        KERNEL_CPP = f"{src_dir}/kernel.cpp"
        MAIN_CPP = f"{src_dir}/main.cpp"
        CHIP = "bm1690"
        OUTPUT_PATH = f"{src_dir}"

        def execute_command(cmd, task_name, timeout):
            """Execute a shell command and handle errors"""
            # fro debug
            # print(f"\n[{task_name}]")
            # print(f"Command: {cmd}")
            
            try:
                _ = subprocess.run(cmd, timeout= timeout, shell=True, check=True, text=True)
                print(f"{task_name} completed")
                return True
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Compile kernel failed because of {e}") from e

        print("=" * 60)
        print("PPL COMPILATION STARTING")
        print(f"Chip: {CHIP}")
        print(f"Output: {OUTPUT_PATH}")
        print("Mode: cmodel")
        print("=" * 60)
        
        # 1. Compile kernel-cpp (kernel_cpp)
        cmd1 = f"""/usr/bin/c++ -D__{CHIP}__ \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/kernel/include \
        -I{PPL_TOP}/runtime/customize/include \
        -I{PPL_TOP}/runtime/kernel \
        -I{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/include \
        -I{OUTPUT_PATH}/include \
        -Wl,--no-undefined -O3 -DNDEBUG -O3 -fPIC -std=c++11 \
        -c {KERNEL_CPP} \
        -o {OUTPUT_PATH}/kernel_cpp.o"""
        
        execute_command(cmd1, "Compile kernel cpp", timeout)
        
        # 2. Compile main-cpp (main_cpp)
        cmd2 = f"""/usr/bin/c++ -D__{CHIP}__ \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/kernel/include \
        -I{PPL_TOP}/runtime/customize/include \
        -I{PPL_TOP}/runtime/kernel \
        -I{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/include \
        -I{OUTPUT_PATH}/include \
        -Wl,--no-undefined -O3 -DNDEBUG -O3 -fPIC -std=c++11 \
        -c {MAIN_CPP} \
        -o {OUTPUT_PATH}/main_cpp.o"""
        
        execute_command(cmd2, "Compile main cpp", timeout)
        
        # 3. Compile kernel-c (kernel_c)
        cmd3 = f"""/usr/bin/cc -D__{CHIP}__ -Dkernel_EXPORTS \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/kernel/include \
        -I{PPL_TOP}/runtime/customize/include \
        -I{PPL_TOP}/runtime/kernel \
        -I{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/include \
        -I{OUTPUT_PATH}/include \
        -I{OUTPUT_PATH}/include \
        -I{PPL_TOP}/include \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/common/include \
        -Wl,--no-undefined -O3 -DNDEBUG -fPIC -O3 \
        -c {KERNEL_C} \
        -o {OUTPUT_PATH}/kernel_c.o"""
        
        execute_command(cmd3, "Compile C kernel", timeout)
        
        # 4. Compile ppl_helper.c
        PPL_HELPER = f"{PPL_TOP}/runtime/customize/src/ppl_helper.c"
        cmd4 = f"""/usr/bin/cc -D__{CHIP}__ -Dkernel_EXPORTS \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/kernel/include \
        -I{PPL_TOP}/runtime/customize/include \
        -I{PPL_TOP}/runtime/kernel \
        -I{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/include \
        -I{OUTPUT_PATH}/include \
        -I{OUTPUT_PATH}/include \
        -I{PPL_TOP}/include \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/common/include \
        -Wl,--no-undefined -O3 -DNDEBUG -fPIC -O3 \
        -c {PPL_HELPER} \
        -o {OUTPUT_PATH}/ppl_helper_c.o"""
        
        execute_command(cmd4, "Compile PPL helper", timeout)
        
        # 5. Link libkernel.so
        cmd5 = f"""/usr/bin/cc -fPIC -Wl,--no-undefined -O3 -DNDEBUG -shared -Wl,-soname,libkernel.so \
        -o {OUTPUT_PATH}/libkernel.so \
        {OUTPUT_PATH}/kernel_c.o \
        {OUTPUT_PATH}/ppl_helper_c.o \
        -L{PPL_TOP}/runtime/{CHIP}/lib \
        -L{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib \
        -Wl,-rpath,{PPL_TOP}/runtime/{CHIP}/lib:{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib \
        {PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib/libtpuv7_emulator.so -lm"""
        
        execute_command(cmd5, "Link libkernel.so", timeout)
        
        # 6. Link executable
        cmd6 = f"""/usr/bin/c++ -O3 -DNDEBUG -fPIC -shared \
        {OUTPUT_PATH}/kernel_cpp.o \
        {OUTPUT_PATH}/main_cpp.o \
        -o {OUTPUT_PATH}/main.so \
        -L{PPL_TOP}/runtime/{CHIP}/lib \
        -L{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib \
        -Wl,-rpath,{PPL_TOP}/runtime/{CHIP}/lib:{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib \
        -ltpuv7_rt -lcdm_daemon_emulator -lpthread"""
                
        execute_command(cmd6, "Link main.so lib", timeout)

