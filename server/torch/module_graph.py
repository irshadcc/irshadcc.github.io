"""Lowers a torch.fx-traced module to the JSON spec that ModuleGraph draws.

ModuleGraph (src/utils/distributed/graph/ModuleGraph.astro) takes a nested module spec, described
in src/utils/distributed/graph/moduleGraph.ts: every module with operations is a scope with
`inputs`, `operations` (ops and submodules) and `outputs`; references name a tensor in the current
scope ("x" for an input of the scope, "relu.out" for an output of a sibling operation).

The lowering:
  1. runs the traced graph on example inputs, recording every node's value (an Interpreter);
  2. places each node in a scope: an op (call_function, call_method, get_attr) in the module it
     runs in (the innermost entry of node.meta["nn_module_stack"]), a leaf module (call_module) in
     its parent, both by qualified name, so "experts.0.w1" sits in "experts" > "0";
  3. turns each node's tensor arguments into references, routing a tensor that crosses module
     boundaries out through the outputs of the modules it leaves and in through the inputs of the
     modules it enters;
  4. attaches the values (shape, and the matrix shown on hover when small enough) and the
     parameters of each module as `weights` (raw values when small, singular values and a
     histogram when not).

Usage:
    gm = torch.fx.symbolic_trace(model)
    spec = FxModuleGraphLowering(gm).lower(x)
    json.dump(spec, open("graph.json", "w"))

or `module_graph(model, x)`; `python module_graph.py out.json` writes an example.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx


@dataclass
class Scope:
    """A module with operations: the spec being built and the bookkeeping for its references."""

    path: tuple[str, ...]
    spec: dict
    # Node name -> the reference to its value in this scope, once routed here.
    refs: dict[str, str] = field(default_factory=dict)
    children: dict[str, "Scope"] = field(default_factory=dict)


def _module_name(target: Any) -> str:
    """A short name for a call_function target: "torch.relu", "F.gelu", "operator.add"."""
    name = getattr(target, "__name__", str(target))
    module = getattr(target, "__module__", None) or ""
    if module == "_operator":
        module = "operator"
    elif module.startswith("torch.nn.functional"):
        module = "F"
    elif module.startswith("torch._C._nn") or module.startswith("torch._C._VariableFunctions"):
        module = "torch"
    elif not module and isinstance(target, type(torch.relu)):  # builtin torch functions
        module = "torch"
    return f"{module}.{name}" if module else name


class FxModuleGraphLowering:
    """
    Lowers a torch.fx.GraphModule (from torch.fx.symbolic_trace) to ModuleGraph's JSON spec.

    max_rows / max_cols: tensors whose matrix view is at most this size carry their values (the
    hover matrix); larger ones carry their shape only. max_weight_values: parameters with at most
    this many entries carry their values; larger ones carry singular values and a histogram.
    vector_weights: also list 1-D parameters (biases, norm scales), whose rank says nothing.
    symbols: the meaning of names used in symbolic shapes, passed through to the spec.
    root: the module that was traced. symbolic_trace replaces modules that are not called
    directly (a ModuleList, a module only some of whose children are called) with plain
    nn.Module placeholders; with `root`, boxes get their real class and parameters. Without it,
    classes come from node.meta["nn_module_stack"] where FX recorded them.
    """

    def __init__(
        self,
        fx_module_graph: torch.fx.GraphModule,
        *,
        root: torch.nn.Module | None = None,
        max_rows: int = 16,
        max_cols: int = 16,
        max_weight_values: int = 4096,
        vector_weights: bool = False,
        decimals: int = 2,
        symbols: dict[str, str] | None = None,
    ):
        self.fx_module_graph = fx_module_graph
        self.root_module = root if root is not None else fx_module_graph
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.max_weight_values = max_weight_values
        self.vector_weights = vector_weights
        self.decimals = decimals
        self.symbols = symbols
        self.module_graph: dict | None = None

    # Values -----------------------------------------------------------------------------------

    def _cell(self, v: Any) -> str:
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{v:.{self.decimals}f}"
        return str(v)

    def _tensor_value(self, name: str, value: Any) -> dict:
        """A TensorValue: shape, and values as a matrix when small enough."""
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu()
            out: dict = {"name": name, "shape": list(t.shape)}
            if t.is_floating_point() or t.dtype in (torch.int32, torch.int64, torch.bool):
                m = t.reshape(1, -1) if t.dim() <= 1 else t.reshape(-1, t.shape[-1])
                if m.shape[0] <= self.max_rows and m.shape[1] <= self.max_cols and m.numel():
                    out["values"] = [[self._cell(v) for v in row] for row in m.tolist()]
                    if t.dim() > 2:
                        out["axes"] = [f"({', '.join(f'dim {i}' for i in range(t.dim() - 1))})", f"dim {t.dim() - 1}"]
            out["note"] = f"{str(t.dtype).replace('torch.', '')}"
            return out
        if isinstance(value, (tuple, list)):
            return {"name": name, "shape": [len(value)], "note": f"{type(value).__name__} of {len(value)}"}
        if isinstance(value, (int, float, bool)):
            return {"name": name, "shape": [], "values": [[self._cell(value)]]}
        return {"name": name, "shape": [], "note": type(value).__name__}

    def _weight(self, p: torch.Tensor) -> dict | None:
        """A WeightSpec for a parameter, viewed as (shape[0], rest); None for skipped vectors."""
        t = p.detach().float().cpu()
        if t.dim() < 2 and not self.vector_weights:
            return None
        m = t.reshape(1, -1) if t.dim() < 2 else t.reshape(t.shape[0], -1)
        spec: dict = {"shape": list(t.shape)}
        if t.numel() <= self.max_weight_values:
            spec["values"] = [[round(v, 4) for v in row] for row in m.tolist()]
            return spec
        spec["singular_values"] = [round(v, 6) for v in torch.linalg.svdvals(m).tolist()]
        lo, hi = float(t.min()), float(t.max())
        bins = 24 if hi > lo else 1
        counts = torch.histc(t, bins=bins, min=lo, max=hi if hi > lo else lo + 1)
        width = (hi - lo) / bins if hi > lo else 1.0
        spec["histogram"] = {
            "edges": [round(lo + i * width, 6) for i in range(bins + 1)],
            "counts": [int(c) for c in counts.tolist()],
        }
        spec["note"] = f"{t.numel()} values: singular values and histogram computed in torch."
        return spec

    def _weights(self, module: torch.nn.Module, recurse: bool) -> dict | None:
        ws = {}
        for k, p in module.named_parameters(recurse=recurse):
            w = self._weight(p)
            if w is not None:
                ws[k] = w
        return ws or None

    # Scopes -----------------------------------------------------------------------------------

    def _scope(self, path: tuple[str, ...]) -> Scope:
        """The scope for a module path, creating it (and its parents) as a submodule box."""
        s = self.root
        for i, key in enumerate(path):
            if key not in s.children:
                qualified = ".".join(path[: i + 1])
                sub = self.root_module.get_submodule(qualified)
                cls = type(sub).__name__
                if type(sub) is torch.nn.Module and qualified in self.classes:
                    cls = self.classes[qualified]
                spec = {"type": "module", "class": cls, "inputs": {}, "operations": {}, "outputs": {}}
                weights = self._weights(sub, recurse=False)
                if weights:
                    spec["weights"] = weights
                s.spec["operations"][key] = spec
                s.children[key] = Scope(path[: i + 1], spec)
            s = s.children[key]
        return s

    def _place(self, node: torch.fx.Node) -> tuple[tuple[str, ...], str | None]:
        """The scope path of a node, and for a leaf module its attribute name there."""
        if node.op == "call_module":
            parts = tuple(str(node.target).split("."))
            return parts[:-1], parts[-1]
        stack = node.meta.get("nn_module_stack") or {}
        if not stack:
            return (), None
        qualified = list(stack.values())[-1][0]
        return tuple(qualified.split(".")) if qualified else (), None

    def _key(self, scope: Scope, wanted: str) -> str:
        """An operation key in a scope: no dots, not taken by another operation or submodule."""
        base = wanted.replace(".", "_") or "op"
        key, i = base, 1
        while key in scope.spec["operations"]:
            key, i = f"{base}_{i}", i + 1
        return key

    def _ref(self, node: torch.fx.Node, path: tuple[str, ...]) -> str:
        """
        A reference to node's value in the scope at `path`: exported through the outputs of the
        modules between its own scope and the common ancestor, imported through the inputs of the
        modules from there down to `path`.
        """
        target = self._scope(path)
        if node.name in target.refs:
            return target.refs[node.name]
        home, ref = self.home[node.name]
        common = 0
        while common < min(len(home), len(path)) and home[common] == path[common]:
            common += 1
        # Up: out of each module from the producer's scope to the common ancestor.
        for depth in range(len(home), common, -1):
            inner = self._scope(home[:depth])
            outer = self._scope(home[: depth - 1])
            if node.name not in outer.refs:
                inner.spec["outputs"][node.name] = ref
                outer.refs[node.name] = f"{home[depth - 1]}.{node.name}"
            ref = outer.refs[node.name]
        # Down: into each module from the common ancestor to the consumer's scope.
        for depth in range(common + 1, len(path) + 1):
            inner = self._scope(path[:depth])
            if node.name not in inner.refs:
                inner.spec["inputs"][node.name] = self._scope(path[: depth - 1]).refs[node.name]
                inner.refs[node.name] = node.name
        return target.refs[node.name]

    def _inputs(self, node: torch.fx.Node, path: tuple[str, ...]) -> dict[str, str]:
        """Input ports: positional arguments as arg0, arg1 (list items as arg0_1), keywords by name."""
        ports: dict[str, str] = {}

        def visit(name: str, a: Any) -> None:
            if isinstance(a, torch.fx.Node):
                ports[name] = self._ref(a, path)
            elif isinstance(a, (tuple, list)):
                for j, item in enumerate(a):
                    visit(f"{name}_{j}", item)
            elif isinstance(a, dict):
                for k, item in a.items():
                    visit(f"{name}_{k}", item)

        for i, a in enumerate(node.args):
            visit(f"arg{i}", a)
        for k, a in node.kwargs.items():
            visit(k, a)
        return ports

    # Lowering ---------------------------------------------------------------------------------

    def lower(self, *example_inputs: Any) -> dict:
        """
        Lower the FX graph to a ModuleGraph spec, running it on `example_inputs` for the values.
        """
        gm = self.fx_module_graph
        values: dict[str, Any] = {}

        class Recorder(torch.fx.Interpreter):
            def run_node(self, n: torch.fx.Node) -> Any:
                out = super().run_node(n)
                values[n.name] = out
                return out

        with torch.no_grad():
            Recorder(gm).run(*example_inputs)

        # Classes FX recorded for the modules it entered, by qualified name.
        self.classes: dict[str, str] = {
            path: getattr(cls, "__name__", str(cls))
            for n in gm.graph.nodes
            for path, cls in (n.meta.get("nn_module_stack") or {}).values()
        }
        spec: dict = {"name": "", "type": "module", "class": type(self.root_module).__name__}
        if self.symbols:
            spec["symbols"] = self.symbols
        spec.update({"inputs": {}, "operations": {}, "outputs": {}})
        weights = self._weights(self.root_module, recurse=False)
        if weights:
            spec["weights"] = weights
        self.root = Scope((), spec)
        # Node name -> (scope path, reference to its value in that scope).
        self.home: dict[str, tuple[tuple[str, ...], str]] = {}

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                spec["inputs"][node.name] = self._tensor_value(node.name, values.get(node.name))
                self.root.refs[node.name] = node.name
                self.home[node.name] = ((), node.name)
                continue
            if node.op == "output":
                outs = node.args[0]
                items = (
                    outs.items() if isinstance(outs, dict)
                    else enumerate(outs) if isinstance(outs, (tuple, list))
                    else [("", outs)]
                )
                for k, a in items:
                    if isinstance(a, torch.fx.Node):
                        name = f"out{k}" if k != "" and not isinstance(k, str) else (k or "out")
                        spec["outputs"][name] = self._ref(a, ())
                continue

            path, attr = self._place(node)
            scope = self._scope(path)
            key = self._key(scope, attr if attr is not None else node.name)
            out = {"out": self._tensor_value(node.name, values.get(node.name))}
            inputs = self._inputs(node, path)
            if node.op == "call_module":
                sub = self.root_module.get_submodule(str(node.target))
                op: dict = {"type": "module", "class": type(sub).__name__, "inputs": inputs, "outputs": out}
                weights = self._weights(sub, recurse=True)
                if weights:
                    op["weights"] = weights
            elif node.op == "get_attr":
                op = {"type": "op", "op": f"getattr {node.target}", "label": str(node.target).split(".")[-1], "outputs": out}
            else:
                name = f"Tensor.{node.target}" if node.op == "call_method" else _module_name(node.target)
                op = {"type": "op", "op": name, "inputs": inputs, "outputs": out}
                if node.op == "call_function" and (getattr(node.target, "__module__", "") or "").startswith("torch.distributed"):
                    op["kind"] = "collective"
            scope.spec["operations"][key] = op
            scope.refs[node.name] = f"{key}.out"
            self.home[node.name] = (path, f"{key}.out")

        self.module_graph = spec
        return spec


def module_graph(module: torch.nn.Module, *example_inputs: Any, **options: Any) -> dict:
    """Traces `module` with torch.fx and lowers it: the ModuleGraph spec as a dict."""
    gm = torch.fx.symbolic_trace(module)
    return FxModuleGraphLowering(gm, root=module, **options).lower(*example_inputs)


if __name__ == "__main__":
    import torch.nn as nn
    import torch.nn.functional as F

    class MLP(nn.Module):
        def __init__(self, d: int, f: int):
            super().__init__()
            self.fc1 = nn.Linear(d, f, bias=False)
            self.fc2 = nn.Linear(f, d, bias=False)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    class ResidualBlock(nn.Module):
        def __init__(self, d: int = 8, f: int = 32):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.mlp = MLP(d, f)

        def forward(self, x):
            return x + self.mlp(self.norm(x))

    torch.manual_seed(0)
    spec = module_graph(ResidualBlock(), torch.randn(4, 8))
    out = sys.argv[1] if len(sys.argv) > 1 else None
    text = json.dumps(spec, indent=1)
    if out:
        with open(out, "w") as fh:
            fh.write(text + "\n")
    else:
        print(text)
