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

## Demo 2: Processor matching and rewriting

Directory: `processor_tests`

Run:

```bash
python examples/processor_tests/main.py
```

This is a lightweight executable test suite for `Processor`. It demonstrates:

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

`keras_to_nn.py` imports the model into NN-IR using the `IRSchema` factory API.
`nn_op_lowering.py` defines a processor-backed `LowerDenseAction`. Each pass
matches a Dense node with its data input, weight, and bias, then rewrites it to:

```text
data → MatMul → BiasAdd → Activation
weight ────────┘
bias ───────────────┘
```

The action lowers one layer per pass because adjacent Dense matches overlap.
`FixedPointTask` repeats the iterative action until both layers are lowered. Pre and post
`CheckSchemaAction` instances validate NN-IR before lowering and OP-IR afterward.
FixedPointTask snapshots are enabled by default; the demo can display them with
`task.snapshot_view()` when that call is enabled in the entry point.

## General execution note

Run commands from the repository root so the editable Loom installation and
example imports resolve consistently. Graph viewers may open an interactive
Heterograph WebView; close the viewer or interrupt the process after inspecting
the graph.
