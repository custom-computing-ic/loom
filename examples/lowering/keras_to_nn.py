"""Import a Keras model into the example layer-level NN-IR."""

from __future__ import annotations

from typing import Any

from tensorflow import keras

from nn_ir import NN_IR


def _as_list(value: Any) -> list[Any]:
    """Normalize Keras symbolic values without iterating tensors directly."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _shape(value: Any) -> list[int | None] | None:
    """Convert a Keras shape object to JSON/Pydantic-friendly dimensions."""
    try:
        return [int(dim) if dim is not None else None for dim in value.shape]
    except (AttributeError, TypeError, ValueError):
        return None


def _dtype(value: Any) -> str | None:
    """Return a stable string representation of a Keras dtype."""
    try:
        return str(value.dtype)
    except AttributeError:
        return None


def build(model: keras.Model, name: str | None = None):
    """Build and return an NN-IR graph for a supported Keras model.

    Currently the importer supports ``InputLayer`` and ``Dense`` layers.
    """
    graph = NN_IR.new_graph()
    graph.pmap["name"] = model.name if name is None else name

    layer_nodes: dict[int, int] = {}

    for layer in model.layers:
        # Keras exposes the symbolic Input as an InputLayer. It becomes an
        # explicit NN-IR source vertex rather than an ordinary computation.
        if isinstance(layer, keras.layers.InputLayer):
            output = _as_list(layer.output)[0]
            layer_nodes[id(layer)] = NN_IR.new_vertex(
                graph,
                "Input",
                name=layer.name,
                shape=_shape(output),
                dtype=_dtype(output),
            )
            continue

        if not isinstance(layer, keras.layers.Dense):
            raise TypeError(f"unsupported Keras layer: {layer.__class__.__name__}")

        output = _as_list(layer.output)[0]
        input_shape = _shape(_as_list(layer.input)[0])
        in_features = int(input_shape[-1]) if input_shape and input_shape[-1] is not None else 0
        dense = NN_IR.new_vertex(
            graph,
            "Dense",
            name=layer.name,
            shape=_shape(output),
            dtype=_dtype(output),
            in_features=in_features,
            out_features=int(layer.units),
            activation=layer.activation.__name__,
        )
        layer_nodes[id(layer)] = dense

        for index, weight in enumerate(layer.weights, start=1):
            parameter = NN_IR.new_vertex(
                graph,
                "Parameter",
                name=weight.name,
                shape=_shape(weight),
                dtype=_dtype(weight),
            )
            NN_IR.new_edge(graph, parameter, dense, "parameter", index=index)

    for layer in model.layers:
        target = layer_nodes.get(id(layer))
        if target is None:
            continue
        for index, tensor in enumerate(_as_list(getattr(layer, "input", []))):
            # _keras_history identifies the layer that produced each input
            # tensor; the tensor itself is not used as a graph vertex.
            history = getattr(tensor, "_keras_history", None)
            if history is None:
                continue
            source = layer_nodes.get(id(history[0]))
            if source is not None:
                NN_IR.new_edge(graph, source, target, "data", index=index)

    for index, tensor in enumerate(_as_list(model.output)):
        # Output is explicit in NN-IR so consumers can distinguish model
        # boundaries from ordinary layer-to-layer dataflow.
        output = NN_IR.new_vertex(
            graph,
            "Output",
            name="output" if len(_as_list(model.output)) == 1 else f"output_{index}",
            shape=_shape(tensor),
            dtype=_dtype(tensor),
        )
        history = getattr(tensor, "_keras_history", None)
        if history is not None:
            source = layer_nodes.get(id(history[0]))
            if source is not None:
                NN_IR.new_edge(graph, source, output, "data", index=0)

    return graph
