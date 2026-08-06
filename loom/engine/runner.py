from enum import Enum

from .rule import Rule


class RulePhase(Enum):
    """Execution phase assigned to a runner rule."""
    PRE = "pre"
    NORMAL = "normal"
    POST = "post"

class Runner:
    """Execute phased graph rules until NORMAL reaches a fixed point."""

    def __init__(self, *, snapshot=True):
        self._rules: dict[RulePhase, list[Rule]] = {
            RulePhase.PRE: [],
            RulePhase.NORMAL: [],
            RulePhase.POST: [],
        }
        self.snapshot = snapshot
        self.webview = None
        self._snapshot_count = 0
        if snapshot:
            from heterograph import WebView
            self.webview = WebView()

    def add_rule(self, rule, *, phase: RulePhase = RulePhase.NORMAL):
        """Register a rule in the selected execution phase."""

        if not isinstance(rule, Rule):
            raise TypeError(
                "add_rule expects a Rule instance."
            )
        if not isinstance(phase, RulePhase):
            raise TypeError("phase must be a RulePhase.")
        self._rules[phase].append(rule)

    def _snapshot(self, g, title):
        """Record the current graph when snapshot rendering is enabled."""
        if self.webview is not None:
            self.webview.add_graph(g, title=title)
            self._snapshot_count += 1

    def snapshot_view(self):
        """Display runner snapshots."""
        if self.webview is None:
            raise RuntimeError("snapshot_view() requires Runner(snapshot=True).")
        return self.webview.run()

    def _apply_rules(self, rules, g, *, phase, verbose=False):
        """Apply one phase in registration order and record modifications."""
        modified = False
        for rule in rules:
            changed = rule.apply(g)
            if not isinstance(changed, bool):
                raise RuntimeError(
                    f"rule {rule.name} returned {type(changed).__name__}; "
                    "Rule.apply() must return bool"
                )
            if changed:
                modified = True
                if verbose:
                    description = "" if rule.description is None else f" ({rule.description})"
                    print(f"  - Rule {rule.name}{description} modified graph")
                self._snapshot(g, f"{phase.value} / {rule.name}")
        return modified

    def execute(self, g, *, max_iterations: int | None = None, verbose=False):
        """Run the pipeline and return iteration and modification metadata.

        Args:
            g: HGraph
            max_iterations: optional safety cap
            verbose: print rule activity

        Returns:
            dict:
                {
                    "iterations": int,
                    "modified": bool
                }
        """

        iteration = 0
        ever_modified = False

        self._snapshot(g, "initial")

        if verbose:
            print("[Runner] PRE rules")
        ever_modified |= self._apply_rules(
            self._rules[RulePhase.PRE], g, phase=RulePhase.PRE, verbose=verbose
        )

        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break

            iteration += 1
            modified_this_round = False

            if verbose:
                print(f"[Runner] Iteration {iteration}")

            modified_this_round = self._apply_rules(
                self._rules[RulePhase.NORMAL], g,
                phase=RulePhase.NORMAL, verbose=verbose
            )
            ever_modified |= modified_this_round

            if not modified_this_round:
                break

        if verbose:
            print("[Runner] POST rules")
        ever_modified |= self._apply_rules(
            self._rules[RulePhase.POST], g, phase=RulePhase.POST, verbose=verbose
        )

        return {
            "iterations": iteration,
            "modified": ever_modified
        }
