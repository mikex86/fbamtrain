# Adding a Triton Kernel Variant (New Dtype Combo)

This repository bakes Triton kernels ahead-of-time. To add a new data type combination (e.g., a new input/output dtype mix) follow the steps below. Paths are relative to `kernel_gen/kernel_gen/`.

## 1) Locate the kernel and its compiler script
- Find the Triton source under `kernels/` (e.g., `kernels/addmm_act.py`).
- Find the corresponding `compile_*.sh` script that already builds variants for this kernel (e.g., `compile_addmm.sh`).

## 2) Define the new signature
- Inside the script, signatures encode pointer dtypes/alignments plus scalar params. Example:
  ```
  MATMUL_FP16_SIGNATURE="*fp16:16, *fp16:16, *fp16:16, *fp16:16, i32, i32, ..."
  ```
- Create a new signature string that matches your desired input/output/accumulator dtypes. Use the Triton parser tokens (`fp16`, `bf16`, `fp32`, etc.) and keep the alignment suffix (`:16`) consistent with existing entries.
- Prefer deriving kernel behavior from the signature (pointer element dtypes) rather than environment variables or global module state. Kernels should read/write in the dtype their pointers encode and only cast when an output dtype differs from the compute dtype.

## 3) Add a compiler invocation
- Append a new `pysrc/triton_aot_compiler.py` call using the new signature. Mirror an existing call, changing:
  - `--kernel-name` (usually unchanged)
  - `--signature` (your new string)
  - `--out-name` (include dtype + arch, e.g., `output/matmul_fp16_out_fp32_sm89`)
  - Targets: add one line per architecture you need (e.g., `--target cuda:89:32`, `--target cuda:100:32`, `--target hip:gfx942:64`).
- Keep num-warp/stage settings aligned with the kernel’s tuning unless you intend to retune.

### Tuning reminder
- Tune kernels by running the Triton kernel in *eager mode* with an `@triton.autotune` annotation active and `TRITON_PRINT_AUTOTUNING=1` set in the environment. The console output will print candidate configs and the chosen winner.
- Take the printed best config (block sizes, num_warps, num_stages, etc.) and copy those parameters into the compile script invocation so the AOT build uses the tuned values.

## 4) Run the script
- Ensure `PYTHON_BIN` points to a Triton-enabled interpreter and `CUDA_HOME` (or ROCm) is set.
- From `kernel_gen/kernel_gen/` run:
  ```
  bash compile_addmm.sh   # or the relevant script
  ```
- Artifacts appear in `output/` (cubin/hsaco + meta sidecars).

## 5) Embed the new binary
- The assembler embeds *whatever is present* in `output/`. Rebuild C++ to re-run `kernel_gen/src/embed_kernels_{nv,amd}.asm` with the new files.
- If you added a new filename pattern, update the architecture header (`kernel_gen/include/nv/kernels_nv.h` or `.../amd/...`) to declare it and add an alias, then expose it through `src/passes/kernel_binaries.cpp`.

## 6) Wire into a pass
- Select the new `k<name>` symbol in the relevant pass (e.g., matmul or cast), and extend the dtype-selection logic to return it.
- Update argument validation if the new combo changes layout or accumulator expectations.

## 7) Validate
- Add or extend a test in `tests/` that exercises the new dtype path.
- Optionally run a quick perf sanity check under `benchmarks/` for regressions.
