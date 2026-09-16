# Known issues and limitations

Run `cascade doctor` and `cascade self-test` when diagnosing installation
errors or solver failures. 

Sometimes I write lazy code; feel free to submit issues in the GitHub

## Platform support

- The supported desktop environments are Linux and WSL2 on x86-64 with Python
  3.9. The bundled Windows launchers require WSL and a distribution named
  `Ubuntu`; future releases may support Windows and/or macOS.
- GPU execution requires an NVIDIA driver compatible with the selected CuPy
  CUDA package. Use `cascade doctor --require-gpu` before GPU-accelerated production runs.

## Flow and transport models

- The configurable tissue oxygen-consumption model is Michaelis-Menten
  (Vmax/Km). Studio's custom wall-exchange law is visible but disabled because
  there is not yet a reproducible user-defined-law interface.
