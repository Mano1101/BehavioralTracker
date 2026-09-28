"""Custom variables / derived columns (Setup page, Standard Tracking only)
-- Upgrade Plan Tier 2 #7, EthoVision's own custom variables: one or more
user-written "name = expression" lines, evaluated against whatever columns
raw_tracking.csv already has at that point (plus a couple of known scalars
like scale_factor/FPS) and added as new columns, e.g.
"distance_cm = Distance_pixels / scale_factor".

Expressions are evaluated with a small hand-rolled AST whitelist
(safe_eval_expr), never eval()/exec()/df.eval() -- there is no way to reach
attribute access, subscripting, or an arbitrary function call from here, so
a malicious or just-broken expression can't escape the sandbox.
"""

import ast
import operator
import re

import numpy as np

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_custom_variables(text):
    """Parse a Custom Variables text box into an ordered list of
    (name, expression) tuples. One 'name = expression' per line; blank
    lines and '#'-prefixed comment lines are ignored.

    Returns (definitions, None) on success, or ([], error_message) if any
    line is malformed (missing '=', an invalid name, an empty expression,
    or a name reused across two lines) -- parsing stops at the first bad
    line so the Setup page can show one clear error instead of a partial
    result.
    """
    if not text or not text.strip():
        return [], None

    defs = []
    seen_names = set()
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            return [], f"line {lineno}: missing '=' -- expected 'name = expression'"
        name, _, expr = line.partition("=")
        name = name.strip()
        expr = expr.strip()
        if not _NAME_RE.match(name):
            return [], f"line {lineno}: '{name}' is not a valid variable name"
        if not expr:
            return [], f"line {lineno}: '{name}' has an empty expression"
        if name in seen_names:
            return [], f"line {lineno}: '{name}' is defined more than once"
        seen_names.add(name)
        defs.append((name, expr))
    return defs, None


# ---------------------------------------------------------------------
# safe_eval_expr -- a small AST-walking evaluator that only ever executes
# arithmetic/comparison/boolean operators and a fixed whitelist of math
# functions against names supplied explicitly in `context`. Anything else
# (attribute access, subscripting, imports, arbitrary calls, comprehensions,
# lambdas, ...) falls through to the final "not allowed" raise below.
# ---------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
_BOOLOPS = {ast.And: operator.and_, ast.Or: operator.or_}
_CMPOPS = {
    ast.Gt: operator.gt, ast.Lt: operator.lt, ast.GtE: operator.ge, ast.LtE: operator.le,
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
_FUNCTIONS = {
    "abs": np.abs, "sqrt": np.sqrt, "round": np.round,
    "floor": np.floor, "ceil": np.ceil,
    "log": np.log, "log10": np.log10, "log2": np.log2, "exp": np.exp,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "min": min, "max": max,
}


def safe_eval_expr(expr, context):
    """Evaluate a single expression string against `context` (a dict of
    name -> scalar or pandas Series), using only whitelisted syntax.
    Raises ValueError for anything outside that whitelist, an unknown
    name, or a plain syntax error -- never lets a real Python exception
    class (AttributeError, KeyError, ...) leak an implementation detail,
    and never actually calls eval()/exec()."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression syntax: {exc}") from exc
    return _eval_node(tree.body, context)


def _eval_node(node, context):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"a {type(node.value).__name__} literal is not allowed")

    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"unknown name '{node.id}'")
        return context[node.id]

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator '{type(node.op).__name__}' is not allowed")
        return op(_eval_node(node.left, context), _eval_node(node.right, context))

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator '{type(node.op).__name__}' is not allowed")
        return op(_eval_node(node.operand, context))

    if isinstance(node, ast.BoolOp):
        op = _BOOLOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator '{type(node.op).__name__}' is not allowed")
        values = [_eval_node(v, context) for v in node.values]
        result = values[0]
        for v in values[1:]:
            result = op(result, v)
        return result

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("chained comparisons (e.g. a < b < c) are not supported")
        op = _CMPOPS.get(type(node.ops[0]))
        if op is None:
            raise ValueError(f"comparison '{type(node.ops[0]).__name__}' is not allowed")
        return op(_eval_node(node.left, context), _eval_node(node.comparators[0], context))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only calling a whitelisted function by name is allowed")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        func_name = node.func.id
        func = _FUNCTIONS.get(func_name)
        if func is None:
            raise ValueError(f"function '{func_name}' is not allowed")
        args = [_eval_node(a, context) for a in node.args]
        return func(*args)

    # Everything else -- Attribute, Subscript, Lambda, comprehensions,
    # Import, Starred, f-strings, ... -- is rejected here.
    raise ValueError(f"'{type(node).__name__}' expressions are not allowed")


def apply_custom_variables(df, custom_variable_defs, scalar_context=None):
    """Evaluate a list of (name, expression) definitions (as returned by
    parse_custom_variables) against df's own columns plus the optional
    scalars in scalar_context (e.g. {"scale_factor": ..., "FPS": ...}),
    adding each as a new column of df.

    Definitions are evaluated in order, so a later one may reference an
    earlier one's newly-added column (chaining). A definition is skipped
    -- with a warning appended to the returned list, never raised -- when
    its name collides with a column that already exists, or when its
    expression fails to evaluate for any reason (unknown column name,
    disallowed syntax, ...). One bad definition never takes down the rest
    or the tracking run it's part of.

    Returns (df, warnings). df is both mutated in place and returned.
    """
    warnings = []
    base_scalars = dict(scalar_context) if scalar_context else {}
    for name, expr in custom_variable_defs:
        if name in df.columns:
            warnings.append(f"'{name}' skipped: a column with that name already exists")
            continue
        context = {col: df[col] for col in df.columns}
        context.update(base_scalars)
        try:
            result = safe_eval_expr(expr, context)
        except Exception as exc:
            warnings.append(f"'{name}' skipped: {exc}")
            continue
        df[name] = result
    return df, warnings
