"""Build the example Keras model and import it into NN-IR."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
from tensorflow.keras.layers import Dense

from keras_to_nn import build as build_keras_to_nn
from loom.engine import (
    Contract,
    ContractResult,
    FixedPointTask,
    Pipeline,
    PipelineResult,
    Verifier,
)
from loom.ir import CheckSchemaAction

from nn_ir import NN_IR
from op_ir import OP_IR
from nn_op_lowering import LowerDenseAction

def build_keras_model(input_dim=16, hidden=32, output=10, batch_size=2):
    """Build the two-layer Keras model used by the lowering example."""
    inp = keras.Input(shape=(input_dim,), batch_size=batch_size, name="input")
    x = Dense(hidden, activation="relu", name="dense_1")(inp)
    out = Dense(output, activation="softmax", name="dense_2")(x)
    return keras.Model(inputs=inp, outputs=out, name="keras_dense")


class LoweringPipeline(Pipeline):
    """Import and lower a Keras graph through the NN-IR and OP-IR stages."""

    def __init__(self):
        super().__init__(name="keras-to-op-ir")
        self.lowering = FixedPointTask(
            pre=[CheckSchemaAction(NN_IR)],
            iterative=[LowerDenseAction()],
            post=[CheckSchemaAction(OP_IR)],
        )

    def execute(self, model, *, workflow: str = "lower") -> PipelineResult:
        graph = build_keras_to_nn(model)
        if workflow == "import":
            return PipelineResult(output=graph)
        if workflow != "lower":
            raise ValueError(f"unknown workflow {workflow!r}")
        task_result = self.lowering.execute(graph)
        return PipelineResult(
            output=graph,
            modified=task_result.modified,
        )


class SchemaContract(Contract):
    """Check that a workflow produces a graph valid for one IR schema."""

    def __init__(self, *, name, pipeline, workflow, schema):
        super().__init__(name=name, pipeline=pipeline, workflow=workflow)
        self.schema = schema

    def evaluate(self, pipeline_result: PipelineResult) -> ContractResult:
        report = self.schema.validate(pipeline_result.output)
        return ContractResult(
            passed=report.ok,
            output=pipeline_result.output,
            failures=[str(issue) for issue in report.errors],
            metrics={"errors": len(report.errors), "issues": len(report.issues)},
        )


if __name__ == "__main__":
    # The pre check validates the imported NN-IR, the iterative action lowers
    # one Dense layer per iteration, and the post check validates the OP-IR result.
    model = build_keras_model()
    pipeline = LoweringPipeline()
    verifier = Verifier([
        SchemaContract(
            name="valid-source-ir",
            pipeline=pipeline,
            workflow="import",
            schema=NN_IR,
        ),
        SchemaContract(
            name="valid-target-ir",
            pipeline=pipeline,
            workflow="lower",
            schema=OP_IR,
        ),
    ])
    results = verifier.verify(model)
    print({name: result.passed for name, result in results.items()})
    pipeline.lowering.snapshot_view()
