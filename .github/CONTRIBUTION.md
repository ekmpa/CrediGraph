## Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment. If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip` installed globally you can just invoke:

```sh
pip install uv
```

### Using uv

#### Background

_uv_ manages a lockfile which maintains a consistent and fixed dependency graph for all _CrediGraph_ dependencies. These dependencies are bundled in a python virtual environment stored in a hidden `.venv` folder in the root project directory. The virtual environment is generated on the fly based on the lockfile and is created upon calling `uv sync`.

#### Running commands

_uv_ bundles all dependencies (including Python itself) into the virtual environment. Most commands can be run by simply prepending `uv run`
to the respective command.

For example:

- Instead of running `python <command>`, you will run `uv run python <command>`
- Instead of running `pytest`, you will run `uv run pytest`

#### Adding and Removing Dependencies

_uv_ has a package resolver which will automatically resolve all packages to their most recent version at the time of installation while respecting the current dependencies.

To add or remove a _core_ dependency, issue `uv add <package>` and `uv remove <package>`, respectively. For instance, to add `numpy` as a core dependency, we would issue:

```sh
uv add numpy
```

**Note**: this will automatically update the [pyproject.toml](../pyproject.toml) and [uv lock file](../uv.lock) with the new package which will be reflected in version control.

In order to facilitate modularity and avoid burdening users with dependencies they don't need, it's recommended to minimize core dependencies to those that **all** users will require for **every** release. To support this, _uv_ has the notion of _dependency groups_, which facilitate auxilary dependencies. For instance, the _dev_ group is the set of dependencies required for _CrediGraph_ development, but is not necessarily shipped to end-users of the library.

To add or remove a package from a dependency group, issue `uv add --<group> <package>` and `uv remove --<group> <package>`, respectively. For instance, to add `hypothesis` as a `dev` dependancy, we would issue:

```sh
uv add --dev hypothesis
```

Note, that auxilary dependency groups can be synced by running `uv sync --group <group name>`.

In general, any wheels published on [pypi](https://pypi.org/) can be directly added, making _uv_ a drop-in replacement for _pip_. For more complex use cases such as non-python dependencies, or installing specific package versions, consult the [uv documentation](https://docs.astral.sh/uv/).

#### Activating the virtual environment

Sometimes you will want to activate the virtual environment manually in order to access the dependencies explicitly (for instance, for integration with a language server for code completion). The virtual environment can be activated by invoking the appropriate activation file, dependencing on your shell. For instance, for _bash_, you can issue:

```sh
. .venv/bin/activate
```

to activate the environment.

**Note**: after doing so, you will have direct access to all executables (e.g. Python) as usual.

## Dev Installation

#### From Source

```sh
# Clone the repo

# Install core dependencies into an isolated environment
uv sync
```

### Install pre-commit hooks:

CrediGraph ships with a set of [pre-commit hooks](../.pre-commit-config.yaml) that automatically apply code formatting, linting, static type analysis, and more.

The hooks can be installed by issuing:

```sh
uv run pre-commit install
```

It is recommended to use these hooks when commiting code remotely but they can also be skipped by commiting with the `--no-verify` flag.

## Unit Testing

The CrediGraph test suite is located under `test/`.
Run the entire (unit) test suite with

```sh
uv run pytest test/
```

## Continuous Integration

In process.
