"""
Run `ty` over the package and print a per-file diagnostic-count table.

Invoked by `make type-check`, which supplies `ty` via `uv run --with ty`.
"""

import json
import os
import subprocess
import sys


def markdown(columns: list[str], rows: list[list], width: int = 0) -> str:
    """
    Render rows as a padded GitHub-flavored markdown table.
    Each column is widened to its longest cell so the raw text lines up;
    cells longer than `width` (when nonzero) are truncated with an ellipsis.
    """

    def clip(value) -> str:
        text = str(value)
        # markdown cells can't hold newlines, so cap length rather than wrap
        return text[: width - 1] + "…" if width and len(text) > width else text

    cells = [[clip(value) for value in row] for row in rows]
    widths = [max([len(c), *(len(r[i]) for r in cells)]) for i, c in enumerate(columns)]

    def line(row):
        return "| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |"

    divider = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(columns), divider, *map(line, cells)])


def run_ty(target: str = "trimesh") -> list[dict]:
    """
    Run `ty` on `target` and return its diagnostics — ty exits non-zero
    whenever it finds anything so we parse output rather than trust the
    return code. the `gitlab` format is a JSON array of diagnostics, each
    carrying `location.path` and `location.positions`.
    """
    completed = subprocess.run(
        ["ty", "check", "--output-format", "gitlab", target],
        capture_output=True,
        text=True,
    )

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        # ty failed before emitting JSON — surface its error rather than
        # an empty table (e.g. `ty` missing or the target failed to load)
        sys.stderr.write(completed.stderr)
        sys.exit(completed.returncode or 1)


def main() -> int:
    diagnostics = run_ty()
    if not diagnostics:
        print("no diagnostics — clean")
        return 0

    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        path = diagnostic["location"]["path"]
        counts[path] = counts.get(path, 0) + 1

    # worst file first — that's the actionable view
    rows = [[f"`{path}`", n] for path, n in counts.items()]
    rows.sort(key=lambda row: row[1], reverse=True)
    rows.append(["total", sum(row[1] for row in rows)])

    print(markdown(["File", "Diagnostics"], rows))

    # `TYPE_ALL=1 make type-check` (or `--all`): also list every
    # diagnostic, in source order by location
    if os.environ.get("TYPE_ALL") or "-all" in " ".join(sys.argv[1:]):
        ordered = sorted(
            diagnostics,
            key=lambda d: (
                d["location"]["path"],
                d["location"]["positions"]["begin"]["line"],
            ),
        )
        detail = [
            [
                f"`{d['location']['path']}:{d['location']['positions']['begin']['line']}`",
                d["description"],
            ]
            for d in ordered
        ]
        print()
        print(markdown(["Location", "Error"], detail, width=120))

    return 0


if __name__ == "__main__":
    sys.exit(main())
