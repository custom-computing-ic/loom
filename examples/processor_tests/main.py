"""Small executable experiments for GraphProcessor matching and rewriting.

Run from the repository root with::

    python examples/processor_rewriting/main.py
"""

from heterograph import HGraph
from loom.engine import Processor


def graph(n, edges=()):
    g = HGraph()
    g.add_vx(n)
    for source, target in edges:
        g.add_edge(source, target)
    return g


def assert_edges(g, expected):
    assert set(g.edges) == set(expected), f"edges: {g.edges!r}, expected: {expected!r}"


def expect_error(fn, message):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError(f"expected RuntimeError: {message}")


def test_basic_match():
    g = graph(3, [(0, 1), (1, 2)])
    p = Processor()
    result = p.run(g, select="a => b => c")
    assert result["matches"] == [{"a": 0, "b": 1, "c": 2}]
    assert result["modified"] is False


def test_where_and_no_match():
    g = graph(3, [(0, 1), (1, 2)])
    p = Processor()
    result = p.run(g, select="a => b", where=lambda g, a, b: a == 1)
    assert result["matches"] == [{"a": 1, "b": 2}]
    assert p.run(g, select="a => b", where=lambda g, a, b: False)["matches"] == []


def test_disjoint_rewrites_and_single_pass():
    g = graph(6, [(0, 1), (1, 2), (3, 4), (4, 5)])
    p = Processor()
    result = p.run(g, select="a => b => c", rewrite="a => c")
    assert result["modified"] is True
    assert_edges(g, [(0, 2), (3, 5)])

    # A three-node chain produces overlapping two-node matches. This is
    # expected and must be rejected before rewriting begins.
    g = graph(3, [(0, 1), (1, 2)])
    p = Processor()
    expect_error(
        lambda: p.run(g, select="a => b", rewrite="a => x"),
        "overlapping two-node matches in a three-node chain",
    )

    # Separate two-node matches are rewritten successfully in one pass.
    g = graph(4, [(0, 1), (2, 3)])
    p = Processor()
    result = p.run(g, select="a => b", rewrite="a => x")
    assert result["modified"] is True
    assert_edges(g, [(0, 4), (2, 5)])




def test_created_nodes_and_finalize():
    g = graph(2, [(0, 1)])
    seen = {}
    p = Processor()

    def finalize(g, a, b, c):
        seen.update(a=a, b=b, c=c)
        g.pmap[c]["created"] = True
        return True

    result = p.run(g, select="a => b", rewrite="a => b => c", finalize=finalize)
    assert result["modified"] is True
    assert seen == {"a": 0, "b": 1, "c": 2}
    assert_edges(g, [(0, 1), (1, 2)])
    assert g.pmap[2]["created"] is True


def test_finalize_requires_boolean_result():
    g = graph(2, [(0, 1)])
    p = Processor()

    def invalid_finalize(g, a, b):
        return None

    expect_error(
        lambda: p.run(
            g,
            select="a => b",
            rewrite="a => b",
            finalize=invalid_finalize,
        ),
        "finalize must return bool",
    )


def test_finalize_rejects_structural_mutation():
    g = graph(3, [(0, 1), (1, 2)])
    p = Processor()

    def remove_vertex(g, a, b, c):
        g.rm_vx(b)
        return False

    expect_error(
        lambda: p.run(
            g, select="a => b => c", rewrite="a => c", finalize=remove_vertex
        ),
        "finalize must not preempt deleted-vertex cleanup",
    )

    g = graph(3, [(0, 1), (1, 2)])
    p = Processor()

    def remove_edge(g, a, b, c):
        g.rm_edge((a, b))
        return False

    expect_error(
        lambda: p.run(
            g, select="a => b => c", rewrite="a => c", finalize=remove_edge
        ),
        "finalize must not preempt LHS-edge cleanup",
    )


def test_rewire_directions():
    base = [(0, 1), (1, 2), (3, 4), (4, 5)]
    g = graph(6, base)
    
    p = Processor()
    p.run(g, select="a => b => c", rewrite="a => x {rewire: b}=> c")               
    assert_edges(g, [(0, 6), (6, 2), (3, 7), (7, 5)])

    g = graph(6, base)
    p = Processor()
    p.run(g, select="a => b => c", rewrite="a => c; c {rewire_in:b}")
    assert_edges(g, [(0, 2), (3, 5)])

    g = graph(6, base)
    p = Processor()
    expect_error(
        lambda: p.run(g, select="a => b => c", rewrite="a => c; b {rewire_out:c}"),
         "rewire source [c] must be deleted by the rewrite"
    )    


def test_rewire_external_edges_to_created_receiver():

    g = graph(5, [(0, 1), (1, 2), (3, 1), (1, 4)]) 
    p = Processor()

    expect_error(
    lambda:p.run(
        g,
        select="a => b => c",
        rewrite="a => x {rewire:b} => c",
    ),
     "Overlapping matches - rewrites require disjoint matches")



def test_internal_rewire_validation():
    g = graph(3, [(0, 1), (1, 2)])
    p = Processor()
    expect_error(
        lambda: p.run(
            g, select="a => b => c", rewrite="a => c; b {rewire_out:c}"
        ),
        "missing explicit RHS edge",
    )

    # The internal edge a -> b would be redirected to a -> x, but the RHS
    # contains no a -> x edge. Since a is preserved, this must be rejected.
    g = graph(2, [(0, 1)])
    p = Processor()
    expect_error(
        lambda: p.run(
            g, select="a => b", rewrite="a; x {rewire_in:b}"
        ),
        "rewire_in creates an undeclared internal edge",
    )


def test_overlap_and_annotation_validation():
    g = graph(3, [(0, 1), (1, 2)])
    p = Processor()
    expect_error(
        lambda: p.run(g, select="a => b", rewrite="a"),
        "overlapping matches",
    )
    expect_error(
        lambda: p.run(g, select="a => b", rewrite="a {rewire:b}"),
        "rewire source must be deleted",
    )
    expect_error(
        lambda: p.run(g, select="a => b", rewrite="a {rewire:z}"),
        "rewire target must be preserved",
    )


TESTS = [
    test_basic_match,
    test_where_and_no_match,
    test_disjoint_rewrites_and_single_pass,
    test_created_nodes_and_finalize,
    test_finalize_requires_boolean_result,
    test_finalize_rejects_structural_mutation,
    test_rewire_directions,
    test_rewire_external_edges_to_created_receiver,
    test_internal_rewire_validation,
    test_overlap_and_annotation_validation,
]


if __name__ == "__main__":
    failures = []
    for test in TESTS:
        try:
            test()
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")

    print(f"\n{len(TESTS) - len(failures)} passed, {len(failures)} failed")
    raise SystemExit(1 if failures else 0)
