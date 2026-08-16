"""Establish the agent-facing interface for the lowering pipeline.

The placeholder agent below can later be replaced by a Pydantic AI adapter.
Only the ``lower-dense`` task factory is exposed as editable.
"""

from pathlib import Path
import importlib.util
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "lowering"))

from loom.agent import AgentLoop, AgentContext, PipelineRunner, TaskFactories

from main import LoweringPipeline, build_keras_model


TASK_PATH = Path(__file__).parent / "draft" / "lower_dense_task.py"


def build_task_factory(path):
    """Create a factory that loads an editable task module per attempt."""
    def factory(pipeline):
        spec = importlib.util.spec_from_file_location("agent_task", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load task module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.LowerDenseTask(pipeline)
    return factory


class PlaceholderAgent:
    def initial_tasks(self, context: AgentContext) -> TaskFactories:
        return {"lower-dense": build_task_factory(
            context.draft["lower-dense"]
        )}

    def revise_tasks(self, context: AgentContext, *, failure: Exception) -> TaskFactories:
        print(f"agent feedback: {type(failure).__name__}: {failure}")
        return {"lower-dense": build_task_factory(
            context.draft["lower-dense"]
        )}


if __name__ == "__main__":
    runner = PipelineRunner(
        LoweringPipeline,
        editable_tasks={"lower-dense"},
        draft={"lower-dense": TASK_PATH},
    )
    result = AgentLoop(runner).run(build_keras_model(), PlaceholderAgent())
    print(result.output)
