# Agent interface example

This example exposes only `lower-dense` as an editable task. The pipeline and
its contracts remain owned by the runner. The editable implementation lives in
the `draft/` directory. `PlaceholderAgent` is the seam where a Pydantic
AI-backed agent can later be connected.
