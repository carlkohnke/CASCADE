# CASCADE

CASCADE is a high-performance framework for vessel-resolved oxygen transport
in vascularized tissues. It simulates blood flow, oxygen delivery, and tissue
oxygenation from engineered constructs to whole organs.

## Requirements

- Windows 10/11 or Linux on x86-64.
- Python 3.12.
- A compatible NVIDIA driver for GPU acceleration.

## Install

Choose the guide for your platform:

- [Native Windows](docs/windows.md)
- [Linux](docs/linux.md)
- [WSL](docs/wsl.md)

Each guide covers CPU and NVIDIA GPU setup, verification, and launching
CASCADE Studio.

## Quick start

Launch the graphical interface with:

```text
cascade-gui
```

Or create and run a starter configuration from the command line:

```text
cascade init-settings case.json
cascade run --settings case.json
```

See the [Studio guide](docs/gui.md) or [CLI guide](docs/cli.md) for the complete
workflow.

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
