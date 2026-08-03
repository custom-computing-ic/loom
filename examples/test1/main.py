#####
import sys
from pathlib import Path
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # show only errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Get the directory of the current script
root_dir = Path(__file__).resolve().parent / ".."

# Add a subdirectory (example: "include")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#####################################################

from keras2compute import ComputeIRBuilder
from loom.rewrite.graph_processor import GraphProcessor, MatchPolicy
from heterograph import WebView

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense


def build_keras_model(input_dim=16, hidden=32, output=10, batch_size=2):
    inp = keras.Input(shape=(input_dim,), batch_size=batch_size, name="input")
    x = Dense(hidden, activation="relu", name="dense_1")(inp)
    out = Dense(output, activation="softmax", name="dense_2")(x)
    return keras.Model(inputs=inp, outputs=out, name="keras_dense")

D, H, O, B = 16, 32, 10, 8  # dims + batch size
keras_model = build_keras_model(D, H, O, batch_size=B)

keras_builder = ComputeIRBuilder()
g1 = keras_builder.build(keras_model)


print(g1.pmap)
for vx in g1.vertices:
    print(f"{vx}: {g1.pmap[vx] }")
for eg in g1.edges:
    print(f"{eg}: {g1.pmap[eg] }")

g2 = g1.copy()

viewer = WebView()
viewer.add_graph(g1,  title="keras layer IR")

class MatchPolicyComputeIR(MatchPolicy):
    @staticmethod
    def vx_annotation(qgraph, vx, op=None, **kwargs):
        qgraph.pmap[vx]['op'] = op

    @staticmethod
    def eg_annotation(qgraph, eg, index=None, **kwargs):
        qgraph.pmap[eg]['index'] = index

    @staticmethod
    def vx_check(g, qgraph, vx, qvx):
        constraint_op = qgraph.pmap[qvx].get('op')
        if constraint_op is not None:
            op = g.pmap[vx].get('op')
            if op != constraint_op:
                return False
        return True

    @staticmethod
    def eg_check(g, qgraph, eg, qeg):
        constraint_index = qgraph.pmap[qeg].get('index')
        if constraint_index is not None:
            index = g.pmap[eg].get('index')
            if index != constraint_index:
                return False
        return True
    
 
gp = GraphProcessor(match_policy=MatchPolicyComputeIR)


def dense2ops(g, w, b, d, w2, b2, matmul,bias_add, act ):

    def set_op(vx, op, name=None):
        if name is None:
            name = f"{op}{vx}"
        g.pmap[vx]['op'] = op
        g.pmap[vx]['attrs']['name'] = name

    def set_edge(src, target, index, channel=None):
        eg = (src, target)
        if channel is not None:
            channel = 'data'
        g.pmap[eg]['index'] = index
        g.pmap[eg]['channel'] = channel

    def copy_shape(vx0, vx1):
        g.pmap[vx1]['shape'] = g.pmap[vx0]['shape']
        g.pmap[vx1]['dtype'] = g.pmap[vx0]['dtype']

    set_op(w2, "weight")
    set_op(matmul, "matmul")
    set_op(b2, "bias")
    set_op(bias_add, "bias_add")
    set_op(act, "activation")

    set_edge(w2, matmul, 1, 'parameter')
    set_edge(matmul, bias_add, 0, 'data')
    set_edge(b2, bias_add, 1, 'parameter')

    set_edge(bias_add, act, 0, 'data')

    for v in (matmul, bias_add, act):
        copy_shape(d, v)
    
    return True




i = 1
while True:
    ret=gp.run(g2,  find="""
                        w={1}>d;
                        b={2}>d;
                        d{dense}""",
                    max_n=1, 
                    rewrite="""
                        matmul{rewire_in:d}=>bias_add=>act{rewire_out:d};
                        w2 => matmul; b2 => bias_add""", 
                    post=dense2ops)
    if ret['modified'] == False:
        break
    viewer.add_graph(g2,  title=f"Compute IR [pass #{i}]")
    i += 1



viewer.run()


# ret=gp.run(g2, find="""i=>d;
#                        w=>d;
#                        b=>d;
#                        d=>o""")

#print("ret:", ret)
# ret['match_pattern'].view()

#g2.view()

