# Tested versions and hardware

CASCADE `0.1.0rc5` was tested on native Windows and WSL 2 on the same laptop:
Intel Core i9-11950H (8 cores, 16 threads, 2.60 GHz), approximately 64 GB RAM,
and an NVIDIA GeForce RTX 3080 Laptop GPU with 16 GiB VRAM.

## Native Windows

| Component | Tested version |
| --- | --- |
| OS | Windows 11 x86-64, build 22621 |
| Python | 3.12.14 |
| CUDA / CuPy | CUDA 13 / CuPy 14.2.0 |
| NVIDIA driver | 616.92 |

CPU, CUDA, CLI, and Studio checks passed. Tested source revision:
`0d97401c7e2203f32efe5135efb254ec4d84c35b`.

Dependency versions: [CPU lock](../requirements/locks/py312-win-amd64-cpu.txt),
[CUDA lock](../requirements/locks/py312-win-amd64-cu13.txt), and
[shared dependencies](../requirements/locks/py312-win-amd64-common.txt).

## Ubuntu under WSL 2

| Component | Tested version |
| --- | --- |
| Guest OS | Ubuntu 24.04.3 LTS, x86-64 |
| Host OS | Windows 11 Pro, build 22621.4317 |
| WSL / kernel | 3.0.1.0 / Linux 6.18.40.1-1 |
| Python | 3.12.3 |
| Memory available to WSL | 52 GiB |
| Demo backend | CPU |

Installation, dependency checks, CPU self-test, and the
[publication demo](publication-demo.md) passed on 5 October 2026. Tested source
revision: `adc0bc490203f5a9baf96650088cbc9c5f1f8df3`, with the accompanying demo
configuration. These demo checks did not exercise CUDA or interactive Studio.

Dependency versions: [Linux/WSL CPU lock](../requirements/locks/py312-linux-x86_64-publication-cpu.txt).

## Hardware requirements

The CPU demo needs no non-standard hardware. GPU acceleration requires an
NVIDIA GPU and a compatible driver. Memory needs depend on simulation size;
the hardware above describes the test system, not minimum requirements.
