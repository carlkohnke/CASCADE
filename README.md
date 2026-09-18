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

## Installation

Choose the guide for your platform:

- [Windows](docs/windows.md)
- [Linux](docs/linux.md)
- [WSL](docs/wsl.md)

## Graphical user interface (GUI)

This project has an interactive GUI for running and analyzing simulations,
though .NPZ files may also be exported for analysis in ParaView or other viewer.

See [GUI](docs/gui.md)


## Quick start

See the [Studio guide](docs/gui.md) or [CLI guide](docs/cli.md) for the complete
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

## Documentation

- [CASCADE Studio](docs/gui.md)
- [Command-line interface](docs/cli.md)
- [Example simulations](docs/examples.md)
- [Domains](docs/domains.md)
- [Custom vascular geometry](docs/custom-geometry.md)
- [Outputs and visualization](docs/outputs.md)
- [Known issues and limitations](docs/known-issues.md)
- [Development setup](docs/development.md)
- [Dependency and lock files](requirements/README.md)
- [Public svVascularize compatibility](docs/svv-compatibility.md)

## License

CASCADE is provided under Stanford's academic, non-commercial license. Review
the complete terms in [LICENSE](LICENSE) before accessing or using the software.
