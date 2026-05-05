"""Core orchestration: pipeline DAG, plugin registry, configuration."""

from quantflow.core.pipeline import Pipeline, PipelineNode
from quantflow.core.registry import register, registry, resolve

__all__ = ["Pipeline", "PipelineNode", "register", "resolve", "registry"]
