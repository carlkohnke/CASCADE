# Known issues and limitations

Run `cascade doctor` and `cascade self-test` when diagnosing installation
errors or solver failures.

## Platform support

- The supported desktop environments are Linux and WSL2 on x86-64 with Python
  3.9. The bundled Windows launchers require WSL and a distribution named
  `Ubuntu`; future releases may support Windows and/or macOS.
- GPU execution requires an NVIDIA driver compatible with the selected CuPy
  CUDA package. Use `cascade doctor --require-gpu` before GPU-accelerated production runs.

## Flow and transport models

- Pressure-pressure flow is supported for steady linear resistance networks.
  With hematocrit-dependent viscosity, CASCADE still performs the configured
  hematocrit/viscosity fixed-point iterations, but it does not search over inlet
  flow: each iteration imposes the requested pressure drop algebraically.
- The configurable tissue oxygen-consumption model is Michaelis-Menten
  (Vmax/Km). Studio's custom wall-exchange law is visible but disabled because
  there is not yet a reproducible user-defined-law interface.

## First-run latency

- A new process must import the scientific stack, construct or load geometry,
  initialize CUDA, and compile first-use kernels. For related cases, use Studio,
  `cascade batch`, or `cascade sweep` so compatible geometry and accelerator state
  can remain warm. This improves subsequent simulations' runtimes without affecting
  numerical results.
