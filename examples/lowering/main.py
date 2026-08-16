"""Build the example Keras model and import it into NN-IR."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
from tensorflow.keras.layers import Dense

from import_keras_task import ImportKerasTask
from loom.core import Pipeline, PipelineResult, Verifier
from loom.graphite import GraphSchemaContract

from nn_ir import NN_IR
from op_ir import OP_IR
from lower_dense_task import LowerDenseTask

def build_keras_model(input_dim=16, hidden=32, output=10, batch_size=2):
    """Build the two-layer Keras model used by the lowering example."""
    inp = keras.Input(shape=(input_dim,), batch_size=batch_size, name="input")
    x = Dense(hidden, activation="relu", name="dense_1")(inp)
    out = Dense(output, activation="softmax", name="dense_2")(x)
    return keras.Model(inputs=inp, outputs=out, name="keras_dense")


class LoweringPipeline(Pipeline):
    """Import and lower a Keras graph through the NN-IR and OP-IR stages."""

    def builtin_task_factories(self):
        return {
            "import-keras": lambda pipeline: ImportKerasTask(pipeline),
            "lower-dense": lambda pipeline: LowerDenseTask(pipeline),
        }

    def __init__(self, task_factories=None):
        super().__init__(name="keras-to-op-ir", task_factories=task_factories)
        self.import_model = self.register("import-keras")
        self.lowering = self.register("lower-dense")
        self.source_verifier = Verifier([GraphSchemaContract(NN_IR)])
        self.target_verifier = Verifier([GraphSchemaContract(OP_IR)])

    def execute(self, model) -> PipelineResult:
        import_result = self.import_model.execute(model)
        source_contracts = self.source_verifier.verify(import_result)
        graph = import_result.output
        lower_result = self.repeat(self.lowering, graph)
        target_contracts = self.target_verifier.verify(lower_result)
        return PipelineResult(
            output=graph,
            modified=lower_result.modified,
            metadata={
                "contract_results": {
                    "import": source_contracts,
                    "lower": target_contracts,
                },
            },
            task_results={"import": import_result, "lower": lower_result},
        )


if __name__ == "__main__":
    model = build_keras_model()
    pipeline = LoweringPipeline()
    result = pipeline.execute(model)
    print({
        stage: {name: check.passed for name, check in checks.items()}
        for stage, checks in result.metadata["contract_results"].items()
    })
