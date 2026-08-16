# Loom

Loom provides small, domain-neutral building blocks for executing Tasks,
composing them in Pipelines, and verifying their results with Contracts. It is
designed to keep an AI-generated Task bounded and independently verifiable.

Graph-IR support currently lives in `loom.graphite`. It is intentionally
separate from the orchestration core and can later move to Graphite without
affecting users of `loom.core`, such as Artisan.

## Architecture

```text
AI-generated or human-written Task
                |
                v
           TaskResult
                |
                v
Pipeline composes Tasks and returns PipelineResult
                |
                v
Verifier evaluates Contracts for the selected result
```

`TaskResult` and `PipelineResult` both extend `Result`. A Contract evaluates
the explicit result supplied to it; it does not own or run a Pipeline.

## Core API

```python
from loom.core import (
    Contract,
    ContractResult,
    Pipeline,
    PipelineResult,
    Result,
    Task,
    TaskResult,
    Verifier,
)
```

A Task is the smallest executable unit of domain work:

```python
class NormalizeTask(Task):
    def __init__(self):
        super().__init__(name="normalize")

    def execute(self, artifact) -> TaskResult:
        changed = normalize(artifact)
        return TaskResult(output=artifact, modified=changed)
```

A Pipeline owns sequencing, branching, and fixed-point repetition. The default
iteration limit is configured in its constructor and can be overridden for one
call to `repeat()`.

```python
class CompilerPipeline(Pipeline):
    def __init__(self, lower_task):
        super().__init__(name="compile", max_iterations=1000)
        self.lower_task = lower_task

    def execute(self, graph) -> PipelineResult:
        lower_result = self.repeat(self.lower_task, graph)
        return PipelineResult(
            output=graph,
            modified=lower_result.modified,
            task_results={"lower": lower_result},
        )
```

`repeat()` stops when its Task returns `modified=False`. Its result metadata
records the iteration count and whether it converged.

## Contracts and verification

Contracts judge a result after a Task or Pipeline has executed. A caller can
invoke a Verifier explicitly, or a domain-specific Pipeline can do so as part
of its own orchestration. Contracts can validate an AI-generated Task directly,
a whole Pipeline result, or a selected Task result from a Pipeline result.

```python
task_result = candidate_task.execute(input)
checks = Verifier([candidate_contract]).verify(task_result)

pipeline_result = pipeline.execute(input)
checks = Verifier([end_to_end_contract]).verify(pipeline_result)
checks = Verifier([candidate_contract]).verify(
    pipeline_result.task_results["candidate"]
)
```

A Contract implements `evaluate(result: Result) -> ContractResult`. The
Verifier keeps all normal failures so they can be used as feedback for the next
agent revision.

## Graphite support

`loom.graphite` provides the current Heterograph-specific implementation:

```python
from loom.graphite import (
    DfsMatchStrategy,
    GraphProcessor,
    GraphSchema,
    GraphSchemaContract,
    IsoMatchStrategy,
)
```

- `GraphProcessor` matches AQL patterns and performs in-place graph rewrites.
- `GraphSchema` defines and validates typed Heterograph IRs.
- `GraphSchemaContract` validates the output of any `Result` against a graph
  schema.

For example:

```python
processor = GraphProcessor(snapshot=False)
result = processor.run(
    graph,
    select="a => b => c",
    rewrite="a => c",
)
```

Graphite Tasks call `GraphProcessor` directly. Loom core does not define an
Action abstraction and does not require a graph or an IR.

## Agent workflow

`loom.agent` provides a restricted interface for an external agent to develop
and revise selected Task implementations. The agent receives an
`AgentContext`, which exposes the pipeline runner, the names of editable tasks,
and their forge locations. It does not receive the pipeline implementation or
the human-written contracts.

```python
from pathlib import Path

from loom.agent import AgentLoop, PipelineRunner

runner = PipelineRunner(
    LoweringPipeline,
    editable_tasks={"lower-dense"},
    forge={
        "lower-dense": Path("examples/agent/forge/lower_dense_task.py"),
    },
)
result = AgentLoop(runner).run(input_value, agent)
```

The pipeline merges its builtin task factories with the agent-provided
factories. Only the explicitly editable task slots can be replaced. Each
attempt loads the candidate task from the forge and executes the unchanged
pipeline and contracts.

Task exceptions are passed to the agent as normal exceptions. Contract
failures are raised by `Verifier` as `ContractException`, carrying the
execution result and contract results. An agent can use this feedback to
revise the forged task and retry until the contracts pass or the loop reaches
its attempt limit.

The lowering integration example is in `examples/agent`. Its forge currently
contains an intentionally disabled OP-IR update so the contract-failure path
can be exercised; uncomment those lines to restore the successful task.

### Pydantic AI task generation

Select a provider, model, and the task names that `TaskGen` may revise:

```bash
pip install loom-agent
```

```python
from loom.agent import PydanticAIProvider, TaskGen

agent = TaskGen(
    provider=PydanticAIProvider(provider="openai", model="gpt-5.6-sol"),
    tasks={"lower-dense"},
)
result = AgentLoop(runner).run(input_value, agent)
```

`TaskGen` sends only the selected forge module and execution feedback to its
provider. It writes the returned replacement module into that forge path and
returns a factory for the next pipeline attempt. `PydanticAIProvider` owns the
provider-specific model routing; another backend can implement `Provider`
without changing `TaskGen`.

## Examples

Loom requires Python 3.10+ and Heterograph for the Graphite examples. Create
the supplied environment:

```bash
conda env create -f examples/environment.yml
conda activate loom-examples
```

Run the examples from the repository root:

```bash
python examples/validation/main.py
python examples/processor_tests/main.py
python examples/lowering/main.py
python examples/agent/main.py
```
