# Temporary package verification

These artifacts are verification builds under ignored `validation/tmp/`, not release artifacts and not replacements for the tagged `dist/` files.

| Artifact | SHA-256 | Check |
| --- | --- | --- |
| `cascade_vascular-0.1.0rc1-py3-none-any.whl` | `b1682bd582cf7ed2b422faa64efddad4f2e1f82bbd03f6d31a078bc93f40624b` | `twine check` passed |
| `cascade_vascular-0.1.0rc1.tar.gz` | `6d14b1c8cbff1f6c30c1357e2848137deafec686c09d477a3435f67f0a0252f1` | `twine check` passed |

Content audit confirmed:

- `cascade/assets/domains/bivent3.stl` is present in wheel and sdist.
- Root legacy script copies and obsolete same-process comparison helpers are absent.
- `docs/internal/`, validation results, and the workstation-specific historical audit are absent.
- A no-dependencies wheel install outside the checkout resolved the packaged STL at 652,584 bytes with SHA-256 `10fd497650eb881b37390861cb960db88e0c7e6f0b18267ffbb35d564c24a060`.
