"""Build, update, and validate a small NN-IR graph."""

from nn_ir import NN_IR


def qint() -> dict[str, object]:
    """Return the quantization metadata used by the example graph."""
    return {"signed": True, "int_bits": 4, "frac_bits": 4}


def build_graph():
    """Construct a small valid NN-IR graph."""
    graph = NN_IR.new_graph()
    input_v = NN_IR.new_vertex(graph, "Input", shape=[4, 2], qint=qint())
    embedding_v = NN_IR.new_vertex(graph, "Embedding", in_features=2,
                                    out_features=3, qint=qint())
    relu_v = NN_IR.new_vertex(graph, "ReLU", qint=qint())
    pool_v = NN_IR.new_vertex(graph, "SumPool", axis=0, qint=qint())
    dense_v = NN_IR.new_vertex(graph, "Dense", in_features=3,
                                out_features=1, qint=qint())
    output_v = NN_IR.new_vertex(graph, "Output", qint=qint())

    for source, target in ((input_v, embedding_v), (embedding_v, relu_v),
                           (relu_v, pool_v), (pool_v, dense_v),
                           (dense_v, output_v)):
        NN_IR.new_edge(graph, source, target, "data")
    return graph


if __name__ == "__main__":
    graph = build_graph()
    pool = next(v for v in graph.vertices
                if graph.pmap[v]["_type"] == "SumPool")
    NN_IR.update_vertex(graph, pool, axis=0)
    print(NN_IR.validate(graph))


