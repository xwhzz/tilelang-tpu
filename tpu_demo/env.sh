export SRC_DIR="../src/tl_templates/tpu"
export LD_LIBRARY_PATH="/opt/tpuv7/tpuv7-current/lib:${PPL_PROJECT_ROOT}/runtime/bm1690/tpuv7-runtime-emulator/lib:${LD_LIBRARY_PATH}"
export TPU_KERNEL_PATH="${SRC_DIR}"
export PPL_KERNEL_PATH="${SRC_DIR}/libkernel.so"