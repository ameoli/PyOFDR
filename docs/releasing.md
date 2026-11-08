# Publishing PyOFDR

The [release workflow](../.github/workflows/publish.yml) builds a source distribution, builds the wheel from it and runs the installed CLI. Pull requests exercise the build and installation steps.

## First upload

Use the PyPI account that will own the project. In [PyPI publishing settings](https://pypi.org/manage/account/publishing/), register a pending GitHub publisher with these values.

| Field | Value |
| --- | --- |
| PyPI project | `pyofdr` |
| GitHub owner | `ameoli` |
| Repository | `PyOFDR` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

The publish job uses this identity to request a short-lived token. The first successful upload creates the PyPI project. See the [PyPI guide to pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Tag a release

Set the version in `pyproject.toml` and `src/pyofdr/__init__.py`. Once the release commit is on `main` and its checks pass, tag that commit. For version 1.0.0 the commands are:

```bash
git switch main
git pull --ff-only
git tag -a v1.0.0 -m "PyOFDR 1.0.0"
git push origin v1.0.0
```

The workflow compares the tag with the version in the built wheel before upload. The last job installs that exact version from the public index and runs `pyofdr validate`. If only that job fails, rerun the failed job and check its output.
