from .rule import Rule, RuleFn

class Runner:
    """
    Executes a sequence of rules until fixed point.

    Rules are applied sequentially.
    If any rule modifies the graph, a new iteration begins.

    Stops when:
        - No rule modifies the graph in a full pass
        - max_iterations is reached (optional)
    """

    def __init__(self):
        self._rules: list[Rule] = []

    def add_rule(self, rule):
        """
        Add a rule to the engine.

        Accepts:
            - Rule instance
            - Callable (wrapped automatically in RuleFn)
        """

        if isinstance(rule, Rule):
            self._rules.append(rule)
        else:
            raise TypeError(
                "add_rule expects a Rule instance."
            )

    def run(self, g, *, max_iterations: int | None = None, verbose=False):
        """
        Execute rules until fixed point.

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

        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break

            iteration += 1
            modified_this_round = False

            if verbose:
                print(f"[Runner] Iteration {iteration}")

            for rule in self._rules:
                changed = rule.apply(g)

                if changed:
                    modified_this_round = True
                    ever_modified = True

                    if verbose:
                        description = "" if rule.description is None else f" ({rule.description})"
                        print(f"  - Rule {rule.name}{description} modified graph")

            if not modified_this_round:
                break

        return {
            "iterations": iteration,
            "modified": ever_modified
        }
