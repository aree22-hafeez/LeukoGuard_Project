"""Minimal notebook executor (no Jupyter in this sandbox).
Reads a percent-format .py file (# %% / # %% [markdown]), executes the code cells
in one namespace, captures stdout, display() output and matplotlib figures, and
writes a standard .ipynb (nbformat 4) with outputs embedded."""
import ast, base64, contextlib, io, json, sys, traceback
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def split_cells(text):
    cells, cur, kind = [], [], None
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None: cells.append((kind, "\n".join(cur).strip("\n")))
            kind = "markdown" if "[markdown]" in line else "code"; cur = []
        else:
            cur.append(line)
    if kind is not None: cells.append((kind, "\n".join(cur).strip("\n")))
    out = []
    for k, src in cells:
        if k == "markdown":
            src = "\n".join(l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in src.splitlines())
        out.append((k, src))
    return out

def run(src_path, out_path, cwd=None):
    import os
    if cwd: os.chdir(cwd)
    cells = split_cells(Path(src_path).read_text())
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    ns = {"__name__": "__main__"}
    count = 0
    for i, (kind, src) in enumerate(cells):
        lines = [l + "\n" for l in src.split("\n")]; lines[-1] = lines[-1].rstrip("\n")
        if kind == "markdown":
            nb["cells"].append({"cell_type": "markdown", "id": f"c{i}", "metadata": {}, "source": lines}); continue
        count += 1; outputs = []
        buf = io.StringIO()
        def flush():
            t = buf.getvalue()
            if t: outputs.append({"output_type": "stream", "name": "stdout", "text": t})
            buf.seek(0); buf.truncate()
        def display(obj):
            flush()
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                o = obj.to_frame() if isinstance(obj, pd.Series) else obj
                outputs.append({"output_type": "display_data", "metadata": {},
                    "data": {"text/html": o.to_html(max_rows=80), "text/plain": o.to_string(max_rows=80)}})
            else:
                outputs.append({"output_type": "display_data", "metadata": {}, "data": {"text/plain": repr(obj)}})
        def show(*a, **k):
            flush()
            for num in plt.get_fignums():
                fig = plt.figure(num); b = io.BytesIO()
                fig.savefig(b, format="png", dpi=90, bbox_inches="tight")
                outputs.append({"output_type": "display_data", "metadata": {},
                                "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}})
            plt.close("all")
        ns["display"] = display; plt.show = show
        tree = ast.parse(src); last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, f"<cell {count}>", "exec"), ns)
                val = eval(compile(last, f"<cell {count}>", "eval"), ns) if last else None
            if val is not None: display(val)
            show()
        except Exception:
            flush()
            tb = traceback.format_exc()
            outputs.append({"output_type": "error", "ename": "Error", "evalue": "", "traceback": tb.splitlines()})
            print(f"ERROR in cell {count}:\n{src[:300]}\n{tb}", file=sys.stderr)
            nb["cells"].append({"cell_type": "code", "id": f"c{i}", "execution_count": count, "metadata": {}, "outputs": outputs, "source": lines})
            Path(out_path).write_text(json.dumps(nb, indent=1)); sys.exit(1)
        flush()
        nb["cells"].append({"cell_type": "code", "id": f"c{i}", "execution_count": count, "metadata": {}, "outputs": outputs, "source": lines})
        print(f"cell {count} ok", file=sys.stderr)
    Path(out_path).write_text(json.dumps(nb, indent=1))

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
