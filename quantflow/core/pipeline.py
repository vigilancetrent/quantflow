"""Pipeline DAG composition.

A `Pipeline` is a directed acyclic graph of named nodes. Each node implements
either a `transform` (stateless or fit-once) or `fit_transform` (stateful)
contract. The pipeline topologically sorts the DAG, runs nodes in order, and
caches intermediate outputs in a context dict keyed by node name.

This is intentionally minimal — it does not try to replace sklearn's `Pipeline`
or Airflow. It exists so that the same composition object can drive a backtest
*and* live trading with only the data source swapped out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineNode:
    """A single node in the pipeline DAG.

    Parameters
    ----------
    name : str
        Unique identifier within the pipeline.
    fn : Callable
        The function to invoke. Receives the values of `inputs` (positional)
        plus `**params` (keyword).
    inputs : list[str]
        Names of upstream nodes whose outputs feed this node. Use the special
        name ``"input"`` to receive the pipeline's top-level input.
    params : dict[str, Any]
        Extra keyword arguments passed to ``fn`` on every call.
    """

    name: str
    fn: Callable[..., Any]
    inputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Run a DAG of nodes in dependency order."""

    def __init__(self, nodes: list[PipelineNode] | None = None):
        self._nodes: dict[str, PipelineNode] = {}
        if nodes:
            for n in nodes:
                self.add(n)

    def add(self, node: PipelineNode) -> Pipeline:
        if node.name in self._nodes:
            raise ValueError(f"Node {node.name!r} already in pipeline")
        if node.name == "input":
            raise ValueError("Node name 'input' is reserved")
        self._nodes[node.name] = node
        return self

    def _topological_order(self) -> list[str]:
        # Kahn's algorithm. Inputs that are not in the graph (e.g. "input")
        # are treated as already-satisfied roots.
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        edges: dict[str, list[str]] = {n: [] for n in self._nodes}
        for name, node in self._nodes.items():
            for upstream in node.inputs:
                if upstream in self._nodes:
                    edges[upstream].append(name)
                    in_degree[name] += 1

        queue = [n for n, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for downstream in edges[n]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)

        if len(order) != len(self._nodes):
            raise ValueError("Pipeline has a cycle")
        return order

    def run(self, data: Any) -> dict[str, Any]:
        """Execute the pipeline on ``data`` and return all node outputs."""
        ctx: dict[str, Any] = {"input": data}
        for name in self._topological_order():
            node = self._nodes[name]
            args = [ctx[u] for u in node.inputs] if node.inputs else [ctx["input"]]
            ctx[name] = node.fn(*args, **node.params)
        return ctx

    def __repr__(self) -> str:
        return f"Pipeline(nodes={list(self._nodes)})"
