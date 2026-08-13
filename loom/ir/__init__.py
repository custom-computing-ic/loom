"""IR schema validation and lowering framework."""

from .ir_schema import (
    Issue,
    IRSchema,
    Severity,
    ValidationReport,
)
from .schema_rule import CheckSchemaAction

__all__ = [
    "Issue",
    "IRSchema",
    "Severity",
    "ValidationReport",
    "CheckSchemaAction",
]
