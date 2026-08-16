"""Editable task implementation supplied by the agent example."""

from loom.core import Task, TaskResult
from loom.graphite import GraphProcessor

from nn_ir import NN_IR
from op_ir import OP_IR


class LowerDenseTask(Task):
    """Lower one Dense match per pass into primitive OP-IR operations."""

    def __init__(self, pipeline):
        super().__init__(pipeline, name="lower-dense",
                         description="Dense to MatMul/BiasAdd/Activation")
        self.processor = GraphProcessor(snapshot=False)

    @staticmethod
    def _where(g, x, d, w, b):
        return (
            g.pmap[x].get("_type") in {"Input", "Dense", "Activation"}
            and g.pmap[w].get("_type") == "Parameter"
            and g.pmap[d].get("_type") == "Dense"
            and g.pmap[b].get("_type") == "Parameter"
            and g.pmap[(w, d)].get("_type") == "parameter"
            and g.pmap[(w, d)].get("index") == 1
            and g.pmap[(b, d)].get("_type") == "parameter"
            and g.pmap[(b, d)].get("index") == 2
            and g.pmap[(x, d)].get("_type") == "data"
            and g.pmap[(x, d)].get("index") == 0
        )

    @staticmethod
    def _finalize(g, x, d, w, b, matmul, bias_add, act):
        dense = g.pmap[d]
        shape = dense.get("shape")
        dtype = dense.get("dtype")
        name = dense.get("name") or f"dense_{d}"
        g.pmap[w].update({"_type": "Weight"})
        g.pmap[b].update({"_type": "Bias"})
        g.pmap[matmul] = {"_type": "MatMul", "name": f"{name}_matmul",
                          "shape": shape, "dtype": dtype, "activation": None}
        g.pmap[bias_add] = {"_type": "BiasAdd", "name": f"{name}_bias_add",
                            "shape": shape, "dtype": dtype, "activation": None}
        g.pmap[act] = {"_type": "Activation", "name": f"{name}_activation",
                       "shape": shape, "dtype": dtype,
                       "activation": dense.get("activation")}
        for edge, edge_type, index in (
            ((x, matmul), "data", 0), ((w, matmul), "parameter", 1),
            ((matmul, bias_add), "data", 0), ((b, bias_add), "parameter", 1),
            ((bias_add, act), "data", 0),
        ):
            g.pmap[edge] = {"_type": edge_type, "index": index}
        return True

    def execute(self, g) -> TaskResult:
        result = self.processor.run(
            g, select="(x| w| b) => d", where=self._where,
            rewrite="(x | w) => matmul => bias_add => act {rewire_out:d}; b => bias_add",
            finalize=self._finalize, max_n=1,
        )
        modified = result["modified"]
        if modified:            
            # Intentionally disabled: the OP-IR contract should fail.
            # g.pmap["_type"] = OP_IR.name
            # OP_IR.style(g)
            pass
        return TaskResult(output=g, modified=modified)
