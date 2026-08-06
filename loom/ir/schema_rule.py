"""Rules for checking graphs against IR schemas."""

from heterograph import HGraph

from .ir_schema import IRSchema
from ..engine.rule import Rule


class CheckSchemaRule(Rule):
    """Validate a graph against an :class:`IRSchema` without modifying it."""

    def __init__(self, schema: IRSchema):
        super().__init__(
            name=f"check-{schema.name}",
            description=f"validate graph as {schema.name}",
        )
        self.schema = schema

    def apply(self, g: HGraph) -> bool:
        """Validate without changing the graph; fail on validation errors."""
        report = self.schema.validate(g)
        if not report.ok:
            raise RuntimeError(str(report))
        return False
