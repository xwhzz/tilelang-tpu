# TPU Template Build Artifacts

`src/tl_templates/tpu` produces a set of temporary build artifacts during compilation. The current implementation writes these generated files directly into the template directory instead of a separate build output directory.

Output location:

- `src/tl_templates/tpu/`

The following generated files are intentionally ignored:

- `src/tl_templates/tpu/kernel.c`
- `src/tl_templates/tpu/kernel.cpp`
- `src/tl_templates/tpu/kernel.h`
- `src/tl_templates/tpu/libkernel.so`
- `src/tl_templates/tpu/main.cpp`
- `src/tl_templates/tpu/main.so`

These files are generated as part of code emission, interface generation, and shared library linking, and should not be committed to Git.
