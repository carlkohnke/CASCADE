# Known issues and limitations

Run `cascade doctor` and `cascade self-test` when diagnosing installation
errors or solver failures. 

## Platform support

- Supported desktop environments are native Windows 10/11, Linux, and WSL2 on
  x86-64 with Python 3.12. Native Windows and WSL use separate documented
  launcher paths; macOS is not currently supported.
- GPU execution requires an NVIDIA driver compatible with the selected CuPy
  CUDA package. Use `cascade doctor --require-gpu` before GPU-accelerated production runs.

## Flow and transport models

- The configurable tissue oxygen-consumption model is Michaelis-Menten
  (Vmax/Km). Studio's custom wall-exchange law is visible but disabled because
  there is not yet a reproducible user-defined-law interface.
