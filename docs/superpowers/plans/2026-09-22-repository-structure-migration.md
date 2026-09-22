# EvoMath Harness Repository Structure Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the runnable V1 LangGraph Workflow Agent into an independent legacy application while preserving the `adaptive_math` V2/V3 package and all current Harness Evolution work.

**Architecture:** The root project remains the only home of `adaptive_math` and its V2/V3 configs, scripts, tests, and training stack. V1 moves into `legacy/langgraph-agent` as a separately installable `langgraph_agent` package with its own metadata, CLI, tests, examples, configuration, and lock file; neither package may import the other.

**Tech Stack:** Python 3.12, uv, Hatchling, Pydantic, Pydantic Settings, LangGraph-compatible workflow prototype, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-22-repository-structure-design.md`

## Global Constraints

- V1 is named **LangGraph Workflow Agent** and remains independently installable, runnable, and testable.
- V2/V3 retain every existing `adaptive_math.*` import path.
- V3 `adaptive_math.harness` and `adaptive_math.evolution` remain inside the root package.
- V1 and V2/V3 must not import each other.
- Existing uncommitted Harness/Evolution files and tests must not be moved, reformatted, staged, or overwritten.
- Migration changes file organization and engineering entry points only; it does not redesign V1 algorithms.
- Root and V1 projects each own a `pyproject.toml` and `uv.lock`.
- File moves and behavioral fixes remain reviewable; no unrelated formatting or refactoring.

## Review Focus

- Importing `langgraph_agent` from its installed package must not depend on the repository root being on `sys.path`; Task 1 adds an isolated import test.
- Running the V1 CLI without `DASHSCOPE_API_KEY` must still reach `--help` without importing settings or failing validation; Task 2 adds a subprocess CLI test.
- Relative training and example data paths must resolve from the V1 project rather than the caller's working directory; Task 3 adds a non-project-CWD smoke test.
- Root quality commands must not discover or silently exclude V1 source; Task 4 adds explicit package-boundary tests and removes the old exclusion list.
- Historical design links must keep pointing to the moved V1 evidence; Task 5 adds a repository-relative Markdown link/path scan.

---

## Target File Map

### Root project retained

- `src/adaptive_math/**` — V2 Agentic RL and V3 Harness Evolution implementation.
- `configs/**`, `scripts/**`, `tests/{unit,integration,contract}/**` — V2/V3 operations and verification.
- `pyproject.toml`, `uv.lock` — root package metadata and lock.

### V1 project created

- `legacy/langgraph-agent/pyproject.toml` — V1 package, dependency, CLI, pytest, and Ruff configuration.
- `legacy/langgraph-agent/src/langgraph_agent/cli.py` — `AdaptiveSolver` and CLI entry.
- `legacy/langgraph-agent/src/langgraph_agent/{config,router,workflow,llm,rl}/**` — moved V1 modules.
- `legacy/langgraph-agent/tests/**` — moved and updated characterization tests.
- `legacy/langgraph-agent/examples/**` — moved V1 examples with package imports.
- `legacy/langgraph-agent/scripts/train.py` — moved legacy training entry.
- `legacy/langgraph-agent/configs/training.json` — moved legacy training config.
- `legacy/langgraph-agent/examples/data/demo_trajectories.json` — moved demo output.
- `legacy/langgraph-agent/README.md` — independent setup, run, test, and scope documentation.
- `legacy/langgraph-agent/uv.lock` — independent resolved environment.

## Task 1: Move V1 source into an importable `langgraph_agent` package

**Files:**
- Create: `legacy/langgraph-agent/tests/test_package_boundary.py`
- Move: `src/main.py` → `legacy/langgraph-agent/src/langgraph_agent/cli.py`
- Move: `src/config/**` → `legacy/langgraph-agent/src/langgraph_agent/config/**`
- Move: `src/router/**` → `legacy/langgraph-agent/src/langgraph_agent/router/**`
- Move: `src/graph/**` → `legacy/langgraph-agent/src/langgraph_agent/workflow/**`
- Move: `src/llm/**` → `legacy/langgraph-agent/src/langgraph_agent/llm/**`
- Move: `src/rl/**` → `legacy/langgraph-agent/src/langgraph_agent/rl/**`
- Create: `legacy/langgraph-agent/src/langgraph_agent/__init__.py`
- Modify: all moved Python files containing `src.*` imports.

**Interfaces:**
- Consumes: Existing V1 `AdaptiveSolver`, `Router`, `Workflow`, and RL public exports.
- Produces: `langgraph_agent.AdaptiveSolver`; importable `langgraph_agent.{config,router,workflow,llm,rl}` packages.

- [ ] **Step 1: Record the existing V1 characterization baseline**

Run:

```bash
uv run pytest tests/test_router.py tests/test_graph.py tests/test_rl.py -q
```

Expected: all current V1 tests pass before any move. Save the test count in the task notes.

- [ ] **Step 2: Write the isolated package-boundary test**

Create `legacy/langgraph-agent/tests/test_package_boundary.py`:

```python
from __future__ import annotations

import ast
from pathlib import Path

import langgraph_agent
from langgraph_agent.cli import AdaptiveSolver


def test_public_package_exports_solver() -> None:
    assert langgraph_agent.AdaptiveSolver is AdaptiveSolver


def test_v1_does_not_import_adaptive_math_or_legacy_src_package() -> None:
    package_root = Path(langgraph_agent.__file__).parent
    forbidden: list[tuple[Path, str]] = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name == "adaptive_math" or name.startswith("adaptive_math."):
                    forbidden.append((path, name))
                if name == "src" or name.startswith("src."):
                    forbidden.append((path, name))
    assert forbidden == []
```

- [ ] **Step 3: Run the boundary test to verify it fails before packaging**

Run:

```bash
PYTHONPATH=legacy/langgraph-agent/src uv run pytest legacy/langgraph-agent/tests/test_package_boundary.py -q
```

Expected: FAIL because `langgraph_agent` does not exist yet.

- [ ] **Step 4: Move source files and normalize package imports**

Use `git mv` for tracked files. Apply these exact import rules:

```python
# legacy/langgraph-agent/src/langgraph_agent/cli.py
from langgraph_agent.config import get_settings, validate_settings
from langgraph_agent.llm import create_llm_client
from langgraph_agent.router import RouteDecision, Router
from langgraph_agent.workflow import Workflow
```

Remove the old repository-root `sys.path.insert` block from `cli.py`. Relative imports already used inside each moved subpackage remain relative. Rename the physical `graph` directory to `workflow`, while preserving the public class name `Workflow`.

Create `legacy/langgraph-agent/src/langgraph_agent/__init__.py`:

```python
"""Runnable V1 LangGraph workflow-agent baseline."""

from langgraph_agent.cli import AdaptiveSolver

__all__ = ["AdaptiveSolver"]
```

- [ ] **Step 5: Run the isolated boundary and moved characterization tests**

Run:

```bash
PYTHONPATH=legacy/langgraph-agent/src uv run pytest \
  legacy/langgraph-agent/tests/test_package_boundary.py \
  legacy/langgraph-agent/tests/test_router.py \
  legacy/langgraph-agent/tests/test_workflow.py \
  legacy/langgraph-agent/tests/test_rl.py -q
```

Expected: PASS with the same Router/Workflow/RL behavior as the Step 1 baseline.

- [ ] **Step 6: Commit the source boundary**

```bash
git add src legacy/langgraph-agent/src legacy/langgraph-agent/tests
git commit -m "refactor: isolate V1 langgraph agent package"
```

## Task 2: Add independent V1 packaging and CLI

**Files:**
- Create: `legacy/langgraph-agent/pyproject.toml`
- Modify: `legacy/langgraph-agent/src/langgraph_agent/cli.py`
- Modify: `legacy/langgraph-agent/src/langgraph_agent/config/settings.py`
- Create: `legacy/langgraph-agent/tests/test_cli.py`
- Create: `legacy/langgraph-agent/README.md`
- Generate: `legacy/langgraph-agent/uv.lock`

**Interfaces:**
- Consumes: `langgraph_agent.cli:main` and the package created in Task 1.
- Produces: `langgraph-agent` console command and an independent uv environment.

- [ ] **Step 1: Write CLI tests before adding the entry point**

Create `legacy/langgraph-agent/tests/test_cli.py`:

```python
from __future__ import annotations

import os
import subprocess
import sys


def test_cli_help_does_not_require_api_key() -> None:
    env = dict(os.environ)
    env.pop("DASHSCOPE_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-m", "langgraph_agent.cli", "--help"],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    assert result.returncode == 0
    assert "LangGraph Workflow Agent" in result.stdout
```

- [ ] **Step 2: Run the CLI test to verify the missing packaging behavior**

Run from `legacy/langgraph-agent`:

```bash
PYTHONPATH=src uv run pytest tests/test_cli.py -q
```

Expected: FAIL because importing `cli.py` eagerly constructs settings or because the final CLI description is not present.

- [ ] **Step 3: Add package metadata and deferred settings construction**

Create `legacy/langgraph-agent/pyproject.toml` with these sections:

```toml
[project]
name = "langgraph-workflow-agent"
version = "0.1.0"
description = "Runnable V1 LangGraph workflow-agent baseline for EvoMath Harness"
requires-python = ">=3.12,<3.13"
dependencies = [
  "langgraph>=0.2,<1",
  "numpy>=1.26,<3",
  "pydantic>=2.9,<3",
  "pydantic-settings>=2.5,<3",
  "typing-extensions>=4.12,<5",
]

[project.optional-dependencies]
training = [
  "datasets>=3,<4",
  "matplotlib>=3.9,<4",
  "torch>=2.6,<3",
]

[project.scripts]
langgraph-agent = "langgraph_agent.cli:main"

[dependency-groups]
dev = ["pytest>=8,<9", "ruff>=0.8,<1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/langgraph_agent"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

Replace the module-level `settings = Settings()` singleton with a cached function:

```python
from functools import lru_cache


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def validate_settings(settings: Settings | None = None) -> bool:
    current = settings or get_settings()
    return bool(current.DASHSCOPE_API_KEY)
```

Update CLI parser description to `LangGraph Workflow Agent` and add:

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Resolve the independent environment**

Run from `legacy/langgraph-agent`:

```bash
uv lock
uv sync --dev
```

Expected: `uv.lock` is created and the V1 package installs without the root project.

- [ ] **Step 5: Verify the installed CLI and V1 suite**

Run from `legacy/langgraph-agent`:

```bash
uv run langgraph-agent --help
uv run pytest -q
uv run ruff check src tests
```

Expected: help exits 0 without an API key; tests and Ruff pass.

- [ ] **Step 6: Document and commit the standalone application**

Write `legacy/langgraph-agent/README.md` with: historical role, architecture, installation, `uv run langgraph-agent --query ...`, interactive mode, examples, tests, training-extra warning, and explicit statement that direct Qwen responses and early GRPO metrics are prototype behavior.

```bash
git add legacy/langgraph-agent
git commit -m "build: add standalone V1 langgraph application"
```

## Task 3: Move V1 examples, assets, and training entry points

**Files:**
- Move: `examples/**` → `legacy/langgraph-agent/examples/**`
- Move: `autodl_train.py` → `legacy/langgraph-agent/scripts/train.py`
- Move: `training_config.json` → `legacy/langgraph-agent/configs/training.json`
- Move: `demo_trajectories.json` → `legacy/langgraph-agent/examples/data/demo_trajectories.json`
- Create: `legacy/langgraph-agent/tests/test_examples.py`
- Modify: moved examples and training script imports/paths.

**Interfaces:**
- Consumes: Installed `langgraph_agent` package and V1 project root.
- Produces: Examples and training entry points that run without root-project path injection.

- [ ] **Step 1: Write a non-project-CWD example smoke test**

Create `legacy/langgraph-agent/tests/test_examples.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).parents[1]


def test_router_demo_runs_outside_project_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT / "examples" / "router_demo.py")],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the smoke test before moving examples**

Run from `legacy/langgraph-agent`:

```bash
uv run pytest tests/test_examples.py -q
```

Expected: FAIL because `examples/router_demo.py` has not moved yet.

- [ ] **Step 3: Move files and replace repository-root path injection**

Use `git mv` for tracked files. Replace imports with installed-package imports:

```python
from langgraph_agent.cli import AdaptiveSolver
from langgraph_agent.router import analyze_complexity, route_query
```

For RL examples use `from langgraph_agent.rl import ...`. In `scripts/train.py`, replace every `src.rl` import with `langgraph_agent.rl`; specifically correct the existing invalid import to:

```python
from langgraph_agent.rl.trajectory_collector import TrajectoryCollector
```

Resolve assets relative to the V1 project:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training.json"
DEFAULT_TRAJECTORIES = PROJECT_ROOT / "examples" / "data" / "demo_trajectories.json"
```

- [ ] **Step 4: Verify examples and the training entry help/import path**

Run from `legacy/langgraph-agent`:

```bash
uv run pytest tests/test_examples.py -q
uv run python examples/simple_task.py
uv run python examples/router_demo.py
uv run python -c "import runpy; runpy.run_path('scripts/train.py', run_name='not_main')"
```

Expected: all commands exit 0; importing the training entry does not start training.

- [ ] **Step 5: Commit V1 operational assets**

```bash
git add examples autodl_train.py training_config.json demo_trajectories.json legacy/langgraph-agent
git commit -m "refactor: move V1 examples and training assets"
```

## Task 4: Clean the root project and enforce package separation

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/contract/test_legacy_package_boundary.py`
- Modify: `.gitignore`
- Verify: `uv.lock` remains unchanged unless `uv lock --check` proves metadata requires regeneration.

**Interfaces:**
- Consumes: Final V1 and root source locations from Tasks 1–3.
- Produces: Root tooling that covers only `adaptive_math` and a contract test preventing cross-imports.

- [ ] **Step 1: Write the cross-package contract test**

Create `tests/contract/test_legacy_package_boundary.py`:

```python
from __future__ import annotations

import ast
from pathlib import Path


ROOT_PACKAGE = Path("src/adaptive_math")
LEGACY_PACKAGE = Path("legacy/langgraph-agent/src/langgraph_agent")


def _imports(package: Path) -> set[str]:
    names: set[str] = set()
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
    return names


def test_v1_and_v2_v3_do_not_cross_import() -> None:
    root_imports = _imports(ROOT_PACKAGE)
    legacy_imports = _imports(LEGACY_PACKAGE)
    assert not any(name == "langgraph_agent" or name.startswith("langgraph_agent.") for name in root_imports)
    assert not any(name == "adaptive_math" or name.startswith("adaptive_math.") for name in legacy_imports)
```

- [ ] **Step 2: Run the contract test before root cleanup**

Run:

```bash
uv run pytest tests/contract/test_legacy_package_boundary.py -q
```

Expected: PASS after Tasks 1–3; this pins the boundary before tooling cleanup.

- [ ] **Step 3: Remove obsolete root exclusions**

Delete these Ruff exclusions from root `pyproject.toml`:

```toml
"src/main.py",
"src/__init__.py",
"src/config",
"src/graph",
"src/llm",
"src/rl",
"src/router",
"tests/__init__.py",
"tests/test_graph.py",
"tests/test_rl.py",
"tests/test_router.py",
"examples",
"autodl_train.py",
```

Remove the migration comment above the list. Keep only exclusions that still exist and are intentionally generated; if none remain, remove the whole `exclude` key.

- [ ] **Step 4: Add explicit environment ignores without hiding tracked evidence**

Append to `.gitignore`:

```gitignore
# standalone legacy application environment
legacy/langgraph-agent/.venv/
legacy/langgraph-agent/artifacts/
```

Do not ignore `legacy/langgraph-agent/uv.lock`, configs, tests, or examples.

- [ ] **Step 5: Verify both project boundaries**

Run from the repository root:

```bash
uv lock --check
uv run pytest -q
uv run ruff check src tests scripts
uv run mypy
```

Run from `legacy/langgraph-agent`:

```bash
uv lock --check
uv run pytest -q
uv run ruff check src tests
```

Expected: both projects pass independently; root tools do not scan the V1 package.

- [ ] **Step 6: Commit root cleanup and boundary enforcement**

```bash
git add pyproject.toml .gitignore tests/contract/test_legacy_package_boundary.py uv.lock
git commit -m "chore: enforce V1 and adaptive-math boundaries"
```

## Task 5: Update documentation and validate repository paths

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/superpowers/specs/2026-09-09-adaptive-math-rl-product-design.md`
- Modify: any tracked Markdown file returned by the old-path scan.
- Create: `tests/contract/test_documentation_paths.py`

**Interfaces:**
- Consumes: Final paths and commands from Tasks 1–4.
- Produces: Accurate V1/V2/V3 navigation and a regression check for moved paths.

- [ ] **Step 1: Write a documentation path regression test**

Create `tests/contract/test_documentation_paths.py`:

```python
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[2]
MOVED_PATHS = (
    "src/main.py",
    "src/config/",
    "src/router/",
    "src/graph/",
    "src/llm/",
    "src/rl/",
    "tests/test_router.py",
    "tests/test_graph.py",
    "tests/test_rl.py",
    "autodl_train.py",
    "training_config.json",
    "demo_trajectories.json",
)


def test_active_docs_do_not_reference_removed_v1_paths() -> None:
    offenders: list[tuple[str, str]] = []
    for path in (ROOT / "README.md", ROOT / "docs" / "architecture.md"):
        text = path.read_text(encoding="utf-8")
        for moved in MOVED_PATHS:
            if moved in text:
                offenders.append((str(path.relative_to(ROOT)), moved))
    assert offenders == []


def test_readme_relative_markdown_links_exist() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\((?!https?://|#)([^)#]+)(?:#[^)]*)?\)", text)
    missing = [target for target in targets if not (ROOT / target).exists()]
    assert missing == []
```

- [ ] **Step 2: Run the documentation test to verify old references fail**

Run:

```bash
uv run pytest tests/contract/test_documentation_paths.py -q
```

Expected: FAIL and list current V1 paths in `README.md` and `docs/architecture.md`.

- [ ] **Step 3: Update README and current architecture documentation**

Make these exact narrative changes:

- V1 code location becomes `legacy/langgraph-agent`.
- V1 test command becomes `cd legacy/langgraph-agent && uv run pytest`.
- Root directory tree contains only `src/adaptive_math` under active source.
- V1/V2 diagrams and architecture trade-offs remain intact.
- `docs/architecture.md` says V1 is an independently runnable historical baseline, not scheduled for deletion.
- Historical product-design evidence links point to the new V1 paths while preserving the original analysis.

Run the full old-path scan:

```bash
rg -n 'src/(main|config|router|graph|llm|rl)|tests/test_(router|graph|rl)|autodl_train|training_config|demo_trajectories' README.md docs pyproject.toml
```

For dated plans whose commands intentionally describe historical repository state, add a one-line archival note linking to `legacy/langgraph-agent` rather than rewriting the historical plan body.

- [ ] **Step 4: Verify documentation and both projects**

Run:

```bash
uv run pytest tests/contract/test_documentation_paths.py -q
git diff --check
```

Then run from root:

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run mypy
```

Then run from `legacy/langgraph-agent`:

```bash
uv run pytest -q
uv run ruff check src tests
uv run langgraph-agent --help
```

Expected: all checks pass and the CLI prints `LangGraph Workflow Agent`.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs tests/contract/test_documentation_paths.py
git commit -m "docs: document independent V1 architecture"
```

## Task 6: Final verification and GitHub synchronization

**Files:**
- Verify only; modify files only to correct failures attributable to Tasks 1–5.

**Interfaces:**
- Consumes: All migration commits.
- Produces: One verified main branch synchronized to `origin/main` without unrelated user changes.

- [ ] **Step 1: Audit the exact changed-file scope**

Run:

```bash
git status --short
git diff --name-status origin/main...HEAD
```

Expected: migration commits include only V1 moves, root boundary configuration, contract tests, README, and documentation. Existing user-owned V3 modifications remain unstaged unless they were committed separately before execution.

- [ ] **Step 2: Run final root verification**

```bash
uv lock --check
uv run pytest -q
uv run ruff check src tests scripts
uv run mypy
```

Expected: all commands exit 0.

- [ ] **Step 3: Run final V1 verification**

```bash
cd legacy/langgraph-agent
uv lock --check
uv run pytest -q
uv run ruff check src tests
uv run langgraph-agent --help
```

Expected: all commands exit 0 without requiring the root package.

- [ ] **Step 4: Verify the remote before pushing**

```bash
git fetch origin main
git log --oneline --left-right origin/main...HEAD
```

Expected: no unexpected remote-only commits. If remote has moved, stop and reconcile without force-pushing.

- [ ] **Step 5: Push migration commits**

```bash
git push origin main
git ls-remote --heads origin main
```

Expected: remote `main` points to the final local commit.
