"""Build the example Keras model and import it into NN-IR."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
from tensorflow.keras.layers import Dense

from keras_to_nn import build as build_keras_to_nn
from loom.ir import CheckSchemaRule
from loom.engine.runner import Runner, RulePhase

from nn_ir import NN_IR
from op_ir import OP_IR
from nn_op_lowering import LowerDenseRule

def build_keras_model(input_dim=16, hidden=32, output=10, batch_size=2):
    """Build the two-layer Keras model used by the lowering example."""
    inp = keras.Input(shape=(input_dim,), batch_size=batch_size, name="input")
    x = Dense(hidden, activation="relu", name="dense_1")(inp)
    out = Dense(output, activation="softmax", name="dense_2")(x)
    return keras.Model(inputs=inp, outputs=out, name="keras_dense")


if __name__ == "__main__":
    # The PRE check validates the imported NN-IR, the NORMAL rule lowers one
    # Dense layer per iteration, and the POST check validates the OP-IR result.
    model = build_keras_model()
    g = build_keras_to_nn(model)

    runner = Runner()
    runner.add_rule(CheckSchemaRule(NN_IR), phase=RulePhase.PRE)
    runner.add_rule(LowerDenseRule(), phase=RulePhase.NORMAL)
    runner.add_rule(CheckSchemaRule(OP_IR), phase=RulePhase.POST)
    
    runner.execute(g)

    runner.snapshot_view()

