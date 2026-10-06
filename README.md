# CASCADE

CASCADE is a high-performance framework for simulating vessel-resolved oxygen
transport in vascularized tissues. In seconds to minutes, this method predicts
tissue oxygenation across scales, from small tissue voxels to whole organs.

Users provide tissue geometry, vessel networks, flow boundary conditions,
tissue and perfusate properties, cellular uptake coefficients (Michaelis-Menten).
CASCADE simulations return the 3D quasisteady oxygen distributions within and
surrounding vascular networks containing up to hundreds of millions of discrete
blood vessels.

## Requirements

- Windows 10/11 or Linux on x86-64.
- Python 3.12.
- A compatible NVIDIA driver for GPU acceleration.

The direct Python dependencies and supported version ranges are declared in [pyproject.toml](pyproject.toml), including the optional GUI and CUDA extras. See [dependency files](requirements/README.md) for platform-specific locks.

## Installation

Choose the guide for your platform:

- [Windows](docs/windows.md)
- [Linux](docs/linux.md)
- [WSL](docs/wsl.md)

A fresh installation should only take a few minutes.

## Graphical user interface (GUI)

This project has an interactive GUI for running and analyzing simulations,
though .NPZ files may also be exported for analysis in ParaView or other viewer.

See [GUI](docs/gui.md)


## Quick start

See the [Studio (GUI) guide](docs/gui.md) or [CLI guide](docs/cli.md) for the complete
installation procedure and workflow.

Once installed, launch the graphical interface with:

```text
cascade-gui
```
(or use the launcher application if on Windows!)

Or create and run a starter configuration from the command line:

```text
cascade init-settings case.json
cascade run --settings case.json
```

See the [settings reference](docs/settings.md) for every available option and
the [example simulations](docs/examples.md) for complete configurations.

## Demo and tested environments

The [publication demo](docs/publication-demo.md) uses a small simulated vascular
network to demonstrate blood flow, vessel and tissue oxygen transport, and
result export. The guide includes installation and run instructions, expected
outputs, and measured timings.

See [tested versions and hardware](docs/tested-environments.md) for the separate
native Windows and WSL test systems, dependency versions, and validation scope.

## Simulation workflow

CASCADE builds or loads the tissue domain and vascular network, solves
[vascular flow](src/cascade/flow/api.py), then evaluates
[vessel oxygen](src/cascade/concentration/vessel/api.py) and
[tissue oxygen](src/cascade/concentration/tissue/api.py). The
[simulation engine](src/cascade/simulation/engine.py) coordinates these steps
and exports the results.

Users may pick between a direct solution (using pairwise interactions directly) or
an alternate Fourier-space (FFT) implementation.

## Documentation

- [CASCADE Studio GUI](docs/gui.md)
- [Command-line interface](docs/cli.md)
- [Example simulations](docs/examples.md)
- [Main-text simulations](docs/publication-reproduction.md)
- [Tissue domains](docs/domains.md) -> importing your own tissues
- [Custom vascular geometry](docs/custom-geometry.md) -> importing your own vascular data
- [Outputs and visualization](docs/outputs.md)
- [Known issues and limitations](docs/known-issues.md)
- [Development setup](docs/development.md)
- [Dependency and lock files](requirements/README.md)
- [svVascularize compatibility](docs/svv-compatibility.md)

## License

CASCADE is provided under Stanford's academic, non-commercial license. Review
the complete terms in [LICENSE](LICENSE) before accessing or using the software.

The license permits internal academic, non-commercial use and restricts
redistribution of the software and derivatives without prior written Stanford
permission.
