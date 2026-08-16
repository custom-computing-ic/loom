# Loom examples

These examples demonstrate the main Loom workflows:

1. defining and validating a typed IR;
2. matching and rewriting graphs with AQL;
3. importing a Keras model and lowering it from NN-IR to OP-IR.

## Create the Conda environment

From the repository root, create the environment described by
[`environment.yml`](W:\cc\loom.git\examples\environment.yml):

```bash
conda env create -f examples/environment.yml
conda activate loom-examples
```

If the environment already exists, update it with:

```bash
conda env update -f examples/environment.yml --prune
```

The environment installs Python 3.12, Heterograph, TensorFlow, and Loom in
editable mode. The TensorFlow dependency is needed only by the Keras lowering
demo, but the shared environment supports all three examples.

## Demo 1: NN-IR validation

Directory: `validation`

Run:

```bash
python examples/validation/main.py
```

This demo defines a small neural-network schema with typed vertices such as
`Input`, `Embedding`, `Dense`, `ReLU`, `SumPool`, and `Output`. It constructs a
graph through `NN_IR.new_graph()`, `new_vertex()`, and `new_edge()`, updates a
vertex, and prints the validation report.

The important idea is that the schema validates both metadata and structural
invariants. The example checks operation arity, acyclicity, and the expected
number of output vertices.

## Demo 2: GraphProcessor matching and rewriting

Directory: `processor_tests`

Run:

```bash
python examples/processor_tests/main.py
```

This is a lightweight executable test suite for `GraphProcessor`. It demonstrates:

- AQL chain and fan-in matching;
- semantic filtering with `where`;
- one-pass rewrites;
- creation of vertices and `finalize` metadata updates;
- disjoint-match enforcement;
- incoming, outgoing, and bidirectional rewiring;
- validation of invalid rewire annotations and finalize mutations.

The tests use plain `HGraph` objects so the matching and rewriting behavior is
easy to isolate from IR schema details. A successful run prints `PASS` for each
case and a final summary.

## Demo 3: Keras NN-IR to OP-IR lowering

Directory: `lowering`

Run from the repository root:

```bash
python examples/lowering/main.py
```

Or run the supplied Unix-style helper from the demo directory:

```bash
cd examples/lowering
bash run-main.sh
```

This demo builds a two-layer Keras model:

```text
Input → Dense(relu) → Dense(softmax) → Output
```

`ImportKerasTask` imports the model into NN-IR using the `GraphSchema` factory API.
`lower_dense_task.py` defines a `GraphProcessor`-backed `LowerDenseTask`. Each pass
matches a Dense node with its data input, weight, and bias, then rewrites it to:

```text
data → MatMul → BiasAdd → Activation
weight ────────┘
bias ───────────────┘
```

The Pipeline verifies the NN-IR contract after `ImportKerasTask`, repeats the
lowering Task until it reaches a fixed point, then verifies the OP-IR contract.

## General execution note

Run commands from the repository root so the editable Loom installation and
example imports resolve consistently. Graph viewers may open an interactive
Heterograph WebView; close the viewer or interrupt the process after inspecting
the graph.
