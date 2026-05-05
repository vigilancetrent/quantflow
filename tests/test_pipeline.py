"""Pipeline DAG and registry tests."""

from __future__ import annotations

import pytest

from quantflow.core.pipeline import Pipeline, PipelineNode
from quantflow.core.registry import register, registry, resolve


def test_linear_pipeline():
    pipe = Pipeline(
        [
            PipelineNode(name="a", fn=lambda x: x + 1),
            PipelineNode(name="b", fn=lambda x: x * 2, inputs=["a"]),
            PipelineNode(name="c", fn=lambda x: x - 3, inputs=["b"]),
        ]
    )
    out = pipe.run(10)
    assert out["a"] == 11 and out["b"] == 22 and out["c"] == 19


def test_diamond_dag():
    pipe = Pipeline(
        [
            PipelineNode(name="src", fn=lambda x: x),
            PipelineNode(name="left", fn=lambda x: x + 1, inputs=["src"]),
            PipelineNode(name="right", fn=lambda x: x * 2, inputs=["src"]),
            PipelineNode(name="merge", fn=lambda a, b: a + b, inputs=["left", "right"]),
        ]
    )
    out = pipe.run(5)
    assert out["merge"] == (5 + 1) + (5 * 2)


def test_cycle_detection():
    pipe = Pipeline()
    pipe.add(PipelineNode(name="a", fn=lambda x: x, inputs=["b"]))
    pipe.add(PipelineNode(name="b", fn=lambda x: x, inputs=["a"]))
    with pytest.raises(ValueError, match="cycle"):
        pipe.run(1)


def test_duplicate_node_name():
    pipe = Pipeline()
    pipe.add(PipelineNode(name="a", fn=lambda x: x))
    with pytest.raises(ValueError):
        pipe.add(PipelineNode(name="a", fn=lambda x: x))


def test_registry_roundtrip():
    @register("widget", "thing_for_test")
    def thing(x):
        return x * 7

    assert resolve("widget", "thing_for_test")(3) == 21
    assert ("widget", "thing_for_test") in registry()


def test_registry_unknown():
    with pytest.raises(KeyError):
        resolve("widget", "definitely_not_registered")
