from heterograph import HGraph

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from typing import Dict

class ComputeIRBuilder:
    """
    Import a Keras model into layer-level Compute IR.

    The builder is intentionally operator-agnostic: it imports the Keras
    computation graph without decomposing layers into primitive math ops.

    Graph schema
    ------------
    g.pmap contains:

        title : str
            Graph name.

        ir : str
            Always "compute".

        frontend : str
            Source frontend, here "keras".

    Vertex schema
    -------------
    g.pmap[v] contains:

        op : str
            Operation type, typically the lowercase Keras layer class name,
            plus the special values "parameter" and "output".

        shape : tuple | None
            Output tensor shape.

        dtype : str | None
            Output tensor dtype.

        const : bool
            True for constant parameter nodes.

        attrs : dict
            Operator attributes, typically copied from layer.get_config().

    Edge schema
    -----------
    g.pmap[(u, v)] contains:

        channel : str
            Semantic role of the edge. Typical values are:
            - "data"
            - "parameter"
            - "control" (not used in this builder, but reserved for future use)
        index: int | None
            input ordering
    """

    def __init__(self):
        pass

    def _ginit(self, g: HGraph) -> None:
        g.pmap = {
            "title": "",
            "ir": "keras",
        }

    def _vinit(self, g: HGraph, v: int) -> None:
        g.pmap[v] = {
            "op": None,
            "shape": None,
            "dtype": None,
            "const": False,
            "attrs": {},
        }

    def _einit(self, g: HGraph, e: tuple[int, int]) -> None:
        g.pmap[e] = {
            "index": None,
            "channel": None,
        }

    def _as_list(self, x):
        """
        Normalize a Keras input/output object to a Python list.

        Keras symbolic tensors must not be iterated directly, so only actual
        Python container types are unpacked.
        """
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def _get_shape(self, x):
        """
        Return a tensor shape as a tuple when available.
        """
        try:
            return tuple(x.shape)
        except Exception:
            return None

    def _get_dtype(self, x):
        """
        Return a tensor dtype as a string when available.
        """
        try:
            return str(x.dtype)
        except Exception:
            return None
        

    def build(self, model: keras.Model, name: str | None = None) -> HGraph:
        """
        Build a layer-level Compute IR graph from a Keras model.

        Args:
            model: Source Keras model.
            name: Optional graph title.

        Returns:
            HGraph: Layer-level Compute IR.
        """
        g = HGraph(
            ginit=self._ginit,
            vinit=self._vinit,
            einit=self._einit,
        )
        g.pmap["title"] = model.name if name is None else name

        layer_nodes: Dict[int, int] = {}

        # Create one node per Keras layer.
        for layer in model.layers:
            v = g.add_vx()
            layer_nodes[id(layer)] = v

            g.pmap[v]["op"] = layer.__class__.__name__.lower()
            g.pmap[v]["attrs"] = dict(layer.get_config())

            try:
                out0 = self._as_list(layer.output)[0]
            except Exception:
                out0 = None

            g.pmap[v]["shape"] = self._get_shape(out0)
            g.pmap[v]["dtype"] = self._get_dtype(out0)

            # Attach layer parameters as constant nodes.
            for i, w in enumerate(layer.weights):

                v_w = g.add_vx()

                g.pmap[v_w]["op"] = "parameter"
                g.pmap[v_w]["shape"] = self._get_shape(w)
                g.pmap[v_w]["dtype"] = self._get_dtype(w)
                g.pmap[v_w]["const"] = True
                g.pmap[v_w]["attrs"] = {
                    "name": getattr(w, "name", None),
                }                

                g.add_edge(v_w, v)

                e = (v_w, v)
                g.pmap[e]["index"] = i + 1
                g.pmap[e]["channel"] = "parameter"

        # Connect layer-to-layer dataflow.
        for layer in model.layers:
            v_dst = layer_nodes[id(layer)]

            try:
                inputs = self._as_list(layer.input)
            except Exception:
                inputs = []

            for t in inputs:
                kh = getattr(t, "_keras_history", None)
                if kh is None:
                    continue

                src_layer = kh[0]
                v_src = layer_nodes.get(id(src_layer))
                if v_src is None:
                    continue

                if not g.check_edge((v_src, v_dst)):
                    g.add_edge(v_src, v_dst)

                    e = (v_src, v_dst)

                    g.pmap[e]["index"] = 0
                    g.pmap[e]["channel"] = "data"

        # Add explicit output node(s).
        outputs = self._as_list(model.output)

        for i, out_t in enumerate(outputs):

            v_out = g.add_vx()

            g.pmap[v_out]["op"] = "output"
            g.pmap[v_out]["shape"] = self._get_shape(out_t)
            g.pmap[v_out]["dtype"] = self._get_dtype(out_t)

            if len(outputs) > 1:
                g.pmap[v_out]["attrs"]["index"] =  i
            g.pmap[v_out]["attrs"]["name"] = f"output_{i}" if len(outputs) > 1 else "output"

            kh = getattr(out_t, "_keras_history", None)
            if kh is None:
                continue

            src_layer = kh[0]
            v_src = layer_nodes.get(id(src_layer))
            if v_src is None:
                continue

            # create edge with explicit metadata
            g.add_edge(v_src, v_out)

            e = (v_src, v_out)
            g.pmap[e]["index"] = 0
            g.pmap[e]["channel"] = "data"
        
        self._apply_compute_ir_styles(g)
        return g
    
    def _apply_compute_ir_styles(self, g):
        """
        Apply visualization styling for Compute IR graphs.

        The styling is intentionally operator-agnostic and highlights
        the structural properties of the graph rather than specific
        neural network operations.

        Vertex colouring is based on node role:

            input       → green
            output      → red
            const       → amber
            compute     → blue

        Edge colouring highlights semantic channels.

        Parameters
        ----------
        g : HGraph
            Compute IR graph.
        """

        # ------------------------------------------------------------
        # Colour palette
        # ------------------------------------------------------------

        PALETTE = {
            "input":   ("#43A047", "white"),   # green
            "output":  ("#E53935", "white"),   # red
            "const":   ("#FFB300", "black"),   # amber
            "compute": ("#1E88E5", "white"),   # blue
        }

        EDGE_COLOUR = {
            "data": "#424242",
            "parameter": "#7B1FA2",
        }

        # ------------------------------------------------------------
        # Vertex role detection
        # ------------------------------------------------------------

        def node_role(_g, vx):
            p = _g.pmap[vx]

            if p["op"] == "input":
                return "input"

            if p["op"] == "output":
                return "output"

            if p.get("const"):
                return "const"

            return "compute"

        # ------------------------------------------------------------
        # Label formatter
        # ------------------------------------------------------------

        def fmt_shape(shape):
            if shape is None:
                return "?"
            if isinstance(shape, (list, tuple)):
                return "(" + ",".join(str(x) for x in shape) + ")"
            return str(shape)

        def node_label(_g, vx):
            p = _g.pmap[vx]
            name = p.get("attrs", {}).get("name")
            op = p.get("op", "?")
            shape = fmt_shape(p.get("shape"))
            dtype = p.get("dtype")

            if dtype:
                return f"{{{name}\<{op}\>|{vx}|{shape}|{dtype}}}"

            return f"{{{name}\<{op}\>|{vx}|{shape}}}"

        # ------------------------------------------------------------
        # Graph style
        # ------------------------------------------------------------

        g.style = {
            "rankdir": "LR",
            "fontname": "Helvetica",
            "fontsize": "14",
            "labelloc": "t",
            "label": g.pmap.get("title", "Compute IR"),
        }

        # ------------------------------------------------------------
        # Vertex style
        # ------------------------------------------------------------

        g.vstyle = {
            "shape": "Mrecord",
            "style": "filled,bold",
            "fontname": "Helvetica",
            "fontsize": "9",
            "penwidth": "1.3",

            "label": node_label,

            "fillcolor": lambda _g, vx: PALETTE[node_role(_g, vx)][0],
            "fontcolor": lambda _g, vx: PALETTE[node_role(_g, vx)][1],
        }

        # ------------------------------------------------------------
        # Edge style
        # ------------------------------------------------------------

        g.estyle = {
            "arrowhead": "open",
            "fontname": "Helvetica",
            "fontsize": "8",
            "penwidth": "1.2",

            "label": lambda _g, e: f'{_g.pmap[e].get("index")}',

            "color": lambda _g, e: EDGE_COLOUR.get(
                _g.pmap[e].get("channel", "data"),
                "#616161"
            ),

            "fontcolor": lambda _g, e: EDGE_COLOUR.get(
                _g.pmap[e].get("channel", "data"),
                "#616161"
            ),
        }



