# CLAUDE.md

DifField is a PyTorch library for differentiable field calculus: aggregate programs
written with `iterate`, `scatter`, `gather`, `branch` and `mux` run as message passing
on a graph, and gradients flow through every round. The package lives in
`src/diffield/` and its tests in `tests/`:

- `core`: the round context, the state store, alignment and the execution mode;
- `functional`: the tensor operators (scatter aggregations, masks, folds);
- `layers`: the `nn.Module` form of the primitives;
- `dsl`: the user-facing primitives and the building blocks written on top of them;
- `sim`: scenarios, the simulation engine, events, recording and physics helpers;
- `decentralized`: the same programs, one device at a time, on Mesa.

`examples/` holds the experiments (gradients, channels, boids, SEAMS, VMAS). They
are a client of the library, and will move to a repository of their own: they use
only the public API of `diffield`, and nothing in `src/` or `tests/` imports them.
The documentation is plain Markdown in `docs/`, with its figures in `docs/assets/`.

## Commands

```bash
uv sync --extra cpu --extra decentralized --extra vmas   # set up (Python 3.12+)
uv run pytest tests                     # the library tests, as in CI
uv run pytest tests/aggregate/test_election.py::test_name   # a single test
uv run pytest examples                  # the tests of the experiments
uv run ruff check .                     # lint and style, line length 100
uv run pyright                          # types, of src/ and tests/
uvx pre-commit install                  # run all of the above before each commit
```

A change is done only when `pytest`, `ruff` and `pyright` all pass: CI and the
pre-commit hooks run exactly these. When a change touches the public API, also
update the README or the pages in `docs/`.

## Workflow: test-driven development

Every feature and every bug fix follows red → green → refactor. Do not write
production code that no failing test asks for.

1. **Understand.** Restate the requested behaviour as a short list of observable
   cases (normal case, edge cases, error cases). If a requirement is ambiguous,
   ask before writing code.
2. **Red.** Write *one* test for the next case and run it. It must fail, and fail
   for the expected reason (an assertion or a missing name, not a typo or an
   import error elsewhere). Do not write the implementation in the same step.
3. **Green.** Write the minimum code that makes that test pass. Run the whole
   suite, not only the new test.
4. **Refactor.** With the suite green, remove duplication and improve names and
   structure, in both production and test code. Run the suite after each change.
5. **Commit** (see below), then repeat from step 2 for the next case.

Rules:

- Never change or delete an existing test just to make it pass. If a test is
  wrong, say so, explain why, and fix it in its own commit.
- Never skip, `xfail` or weaken an assertion to get to green.
- For a bug, first write a test that reproduces it; the fix comes after.
- Test behaviour through the public API, not implementation details, so the
  refactor step stays free.
- Tests are grouped by the layer they exercise (`tests/aggregate/`,
  `tests/decentralized/`, …, see `TEST_STRUCTURE.md`). Reuse the fixtures of
  `conftest.py` and the `support.py` helpers instead of building new ones.
- Name tests after the behaviour: `test_<unit>_<does_what>_<when>`, and lay them
  out as Arrange / Act / Assert.
- Keep tests fast and deterministic: fix random seeds, keep graphs small.
- Differentiability is behaviour: a new primitive gets a test that gradients
  reach its parameters and stay finite (`assert_finite_gradients`).

## Commits

Commit small and often: one logical change per commit, each one leaving the
suite green. A typical feature is a series of commits, e.g. a failing test plus
its implementation, then a refactor.

- Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
  `<type>(<optional scope>): <description>`.
  - `feat:` a new capability, `fix:` a bug fix, `feat!:` or a
    `BREAKING CHANGE:` footer for a change of the public API;
  - `test:`, `refactor:`, `docs:`, `style:`, `perf:`, `build:`, `ci:`, `chore:`
    for the rest. Scopes are the subpackages or `examples`, e.g. `fix(dsl):`.
- Subject: imperative mood, lowercase, no trailing period, ≤ 72 characters.
- Body (when useful): wrap at 72 columns, explain *why* and what the previous
  behaviour was, not a line-by-line list of the diff. Follow the style of the
  existing history (`git log`).
- Stage files explicitly (`git add <path>`), never `git add -A` / `git add .`.
- Never use `--no-verify`; if a pre-commit hook fails, fix the cause.
- Never amend or rewrite commits that are already pushed, and do not push
  unless asked.
- **No AI attribution.** Do not add `Co-Authored-By: Claude …`, "Generated
  with Claude Code", emoji or any other reference to AI tools in commit
  messages, PR titles or PR descriptions. Commits are authored by the git user
  only.

## Python style

Follow [PEP 8](https://peps.python.org/pep-0008/),
[PEP 257](https://peps.python.org/pep-0257/) and the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html);
ruff and pyright enforce the mechanical part. Match the surrounding code first.

### Design and modularity

- One module, one responsibility. Put new code in the subpackage it belongs to
  (`dsl` for primitives and building blocks, `functional` for tensor operators,
  `sim` for simulation concerns, `decentralized` for device-by-device
  execution). `dsl`, `sim` and `decentralized` depend on `core`, never the other
  way round; the few imports inside functions break the remaining cycles
  between `core`/`functional` and `layers`/`dsl`, do not add new ones.
- A primitive works on fields (`Tensor` of shape `[N, ...]`) and link fields
  (`LinkField`), in both execution modes: `hard` must stay exact, `soft` must
  stay differentiable.
- Small, focused functions; a function longer than ~40 lines or with deep
  nesting is a signal to split it. Prefer early returns over nested `if`s.
- Prefer composition and plain functions over class hierarchies; use
  `dataclass` for data carriers.
- Keep side effects at the edges and pure logic in the middle, so it is easy
  to test.
- No mutable module-level state; if it is unavoidable, make it private
  (`_name`) and expose it through functions.
- Do not duplicate: look for an existing helper before writing a new one.
- Do not add a dependency without asking; experiment-only dependencies go in an
  extra (`decentralized`, `vmas`), not in the core.

### Code

- Type-annotate every public function, method and attribute. Use modern
  syntax: `list[int]`, `X | None`, `collections.abc` abstract types for
  parameters (`Sequence`, `Mapping`, `Callable`).
- Naming: `lower_with_under` for modules, functions and variables, `CapWords`
  for classes, `CAPS_WITH_UNDER` for constants, a leading `_` for internals.
  Descriptive names, no type in the name, no single letters outside short
  loops or maths. Do not shadow builtins (`max`, `min`, `id`, `list`, …).
- Imports: from the subpackage that owns the name
  (`from diffield.dsl import gradient`, `from diffield.sim import GridScenario`),
  grouped as standard library / third party / local, no wildcard imports.
- Never use mutable default arguments; default to `None` and build the value
  inside.
- Exceptions: raise specific built-in ones (`ValueError`, `TypeError`) or a
  custom one ending in `Error`; no bare `except:`, no broad `except Exception`
  unless re-raising; keep `try` blocks small. Do not use `assert` for runtime
  checks in library code (it is fine in tests).
- Comprehensions only when they stay simple; otherwise use a loop.
- Use f-strings, `pathlib`, context managers (`with`) for resources, and
  `logging` instead of `print` in library code.
- Executable scripts put their logic in `main()` behind
  `if __name__ == "__main__":`.

### Docstrings and comments

- Every public module, class and function has a docstring in Google style: a
  one-line summary ending with a period, then `Args:`, `Returns:` (or
  `Yields:`) and `Raises:` when relevant. Describe behaviour and contract, not
  implementation.
- Comments explain *why*, not *what*. Do not leave commented-out code, TODOs
  without context, or comments describing the change you just made.
