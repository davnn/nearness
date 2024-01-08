# Contributing to `nearness`

Thanks for contributing! Before implementing features and changes, please submit an issue to discuss the proposed
changes.

## How to submit a pull request

1. [Fork this repository](https://github.com/davnn/nearness/fork).
2. Clone the forked repository and add a new branch with the feature name.

Before submitting your code as a pull request please do the following steps:

1. Add any changes you want
2. Add tests for the new changes
3. Edit documentation if you have changed something significant
4. Run `task format` to format your changes.
5. Run `task lint` to ensure code style checks.
6. Run `task typing` to ensure type checks.
7. Run `task safety` to ensure safety checks.
8. Run `task test` to verify that all tests are passing.

Conveniently, you can run `task check`, to combine the mentioned commands before contributing a commit.
We use [gitmoji](https://gitmoji.dev/) to categorize different kinds of commits.

## Contributing without `task`

We use [task](https://taskfile.dev/) to provide pre-configured CLI commands for the project, but `task` is
not required, you can also run the commands directly from the CLI. Have a look at `Taskfile.yml` for a reference of
commands.

## Set up a poetry environment

We use [micromamba](https://github.com/mamba-org/mamba) to set up a Python environment
for [poetry](https://python-poetry.org/). Make sure that ``micromamba`` is installed
and available. Run `task env-create` to create an empty Python 3.10 environment named `nearness`. After you have
successfully created the environment activate it with `micromamba activate nearness`. If `poetry` is not already
installed, run `task poetry-install`. Using the activated `nearness` environment check if poetry is using the right
environment with `poetry env info`. Once the poetry setup is complete, you are ready to install the dependencies.

Note that Conda, Mamba or Micromamba is not necessary, you can use a tool of your choice to manage your Python
environment. One benefit of using Conda is that we can override packages that are not easy to install with `pip`,
for example, dependencies with GPU-support.

## Install dependencies

We use [`poetry`](https://github.com/python-poetry/poetry) to manage the dependencies. With an active poetry env,
run `task install` to install all dependencies into environment. After the dependencies are installed, run
`task pre-commit-install` to add the [pre-commit](https://pre-commit.com/) hooks.

## Other help

You can contribute by spreading a word about this library. It would also be a huge contribution to write a short article
on how you are using this project. You can also share your best practices with us.
