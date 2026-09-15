# Known issues and limitations

Run `cascade doctor` and `cascade self-test` when diagnosing installation
errors or solver failures.

## Platform support

- Native Windows 10/11 and Linux on x86-64 with CPython 3.12 are supported.
  Native Windows installation uses the released wheel and does not require WSL.
  Windows on ARM, 32-bit Python, and macOS are not currently qualified.
- GPU execution requires an NVIDIA driver compatible with the selected CuPy
  CUDA package. CUDA 13 is the qualified native Windows GPU option. Use
  `cascade doctor --require-gpu` before GPU-accelerated production runs.

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
