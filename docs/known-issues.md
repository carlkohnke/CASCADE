# Known issues and limitations

Run `cascade doctor` and `cascade self-test` when diagnosing installation
errors or solver failures. 

## Platform support

- Supported desktop environments are native Windows 10/11, Linux, and WSL2 on
  x86-64 with Python 3.12. Native Windows and WSL use separate documented
  launcher paths; macOS is not currently supported.
- GPU execution requires an NVIDIA driver compatible with the selected CuPy
  CUDA package. `gpu-cu13` needs a driver reporting CUDA 13.0 or later (NVIDIA
  580 or newer); on older drivers it fails with `cudaErrorInsufficientDriver`.
  CUDA 13 also dropped Maxwell, Pascal and Volta GPUs (sm_50 to sm_72, such as
  P100 and V100); use `gpu-cu12` for them, or `gpu-cu11` for Kepler (sm_35/37).
- On GPUs in Exclusive_Process compute mode, common on HPC clusters, only one
  process can hold a device. CASCADE runs its GPU check inside the run process
  in that mode, but a run still fails with "GPU is busy or unavailable" while
  another process, such as an interactive Python session with CuPy, holds the
  GPU.
  Use `gpu-cu12` for drivers reporting CUDA 12.x and `gpu-cu11` for 11.2 to
  11.8. `cascade doctor` reports the driver's CUDA version and flags a
  mismatched CuPy package. Use `cascade doctor --require-gpu` before
  GPU-accelerated production runs.
- `gpu-cu11` is limited to CuPy 13, the last CuPy release for CUDA 11. CuPy 13
  keeps NumPy's pre-2.0 scalar type promotion, so a few mixed-precision
  intermediate results can differ in rounding from CuPy 14 runs. CUDA 13 is the
  qualified GPU configuration; the CUDA 11 and 12 families are not yet listed in
  the tested environments.

## Flow and transport models

- The configurable tissue oxygen-consumption model is Michaelis-Menten
  (Vmax/Km). Studio's custom wall-exchange law is visible but disabled because
  there is not yet a reproducible user-defined-law interface.
