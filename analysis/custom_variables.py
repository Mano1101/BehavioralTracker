"""
Custom Variables (Upgrade Plan Tier 2 #7) -- EthoVision lets you write a
JavaScript expression against a trial's existing measures to add a
derived column to the export (e.g. converting a distance already in
pixels to real-world units). This is the same idea, aimed at
tracking/location.py's per-frame raw_tracking.csv: one or more
"name = expression" lines, each expression written against the columns
already in that DataFrame (plus a couple of known scalars such as
scale_factor/FPS) -- e.g. "distance_cm = Distance_pixels / scale_factor".

Deliberately NOT Python's eval()/exec() or pandas' DataFrame.eval(): this
is a small, hand-rolled AST walker with an explicit whitelist of node
types (arithmetic, comparisons, boolean combination, and a short list of
math functions). There is no attribute access, no subscripting, no
comprehensions, no lambdas/def, and a bare name only ever resolves to
something already in the caller-supplied namespace -- so there's no path
to import/exec/file I/O or to the classic "walk object.__class__.__bases__
__subclasses__()" trick used to defeat an eval() sandboxed only by
clearing __builtins__. This is meant to keep a typo or a stray function
call from doing anything worse than failing to compute one column, not to
withstand a determined attacker -- same trust level as any other setting
in this desktop app, which the researcher running it already controls.
"""

import ast

import numpy as np

# Every function name an expression is allowed to call. All are vectorized
# (numpy ufuncs), so they work the same on a whole column (pandas Series)
# as on a single scalar value.
FUNCTIONS = {
    "abs": np.abs, "round": np.round, "min": np.minimum, "max": np.maximum,
    "sqrt": np.sqrt, "log": np.log, "log10": np.log10, "exp": np.exp,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "clip": np.clip,
    "where": np.where,
}

_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_UNARYOPS = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}
_COMPARE_OPS = {
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}


def _eval(node, names):
    if isinstance(node, ast.Expression):
        return _eval(node.body, names)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool, str)):
            return node.value
        raise ValueError(f"'{node.value!r}' isn't a supported constant")

    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"'{node.id}' isn't an existing column or known value")
        return names[node.id]

    if isinstance(node, ast.BinOp):
        op_fn = _BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"'{type(node.op).__name__}' isn't a supported operator")
        return op_fn(_eval(node.left, names), _eval(node.right, names))

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return np.logical_not(_eval(node.operand, names))
        op_fn = _UNARYOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"'{type(node.op).__name__}' isn't a supported operator")
        return op_fn(_eval(node.operand, names))

    if isinstance(node, ast.BoolOp):
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        values = [_eval(v, names) for v in node.values]
        result = values[0]
        for v in values[1:]:
            result = combine(result, v)
        return result

    if isinstance(node, ast.Compare):
        left = _eval(node.left, names)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            op_fn = _COMPARE_OPS.get(type(op))
            if op_fn is None:
                raise ValueError(f"'{type(op).__name__}' isn't a supported comparison")
            right = _eval(comparator, names)
            piece = op_fn(left, right)
            result = piece if result is None else np.logical_and(result, piece)
            left = right
        return result

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise ValueError(
                "only these functions are allowed: " + ", ".join(sorted(FUNCTIONS)))
        if node.keywords:
            raise ValueError("keyword arguments aren't supported here")
        args = [_eval(a, names) for a in node.args]
        return FUNCTIONS[node.func.id](*args)

    raise ValueError(
        f"'{type(node).__name__}' isn't supported here -- only arithmetic, comparisons, and "
        "a few whitelisted functions are allowed"
    )


def safe_eval_expr(expr, names):
    """Evaluate a single expression string against `names` (a dict mapping
    each allowed bare name -- a DataFrame column, a pandas Series, or a
    plain scalar -- to its value). Raises ValueError with a short, human-
    readable reason on anything from a syntax error to an unknown name to
    a disallowed construct; never raises anything else on bad input (a
    genuine internal error, e.g. from a whitelisted numpy function itself,
    still surfaces normally)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid expression ({exc.msg})") from exc
    return _eval(tree, names)


def parse_custom_variables(text):
    """Setup page's Custom Variables box: one 'name = expression' per line
    (blank lines and '#'-prefixed comments are skipped) -> ([(name, expr),
    ...], None), or ([], "<message>") for the first malformed line. Blank
    text is valid and parses to no definitions at all -- this is an
    optional feature. Pure/Qt-free so it's easy to unit test on its own
    (see test_custom_variables.py), same pattern as parse_zone_
    associations in qt_app/main_window.py."""
    text = text.strip()
    if not text:
        return [], None
    definitions = []
    seen = set()
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            return [], f"Line {line_no} ('{line}') is missing '=' -- expected 'name = expression'."
        name, expr = line.split("=", 1)
        name = name.strip()
        expr = expr.strip()
        if not name.isidentifier():
            return [], (f"Line {line_no}: '{name}' isn't a valid variable name (letters, digits, "
                        "underscore; can't start with a digit).")
        if not expr:
            return [], f"Line {line_no}: '{name}' has no expression after '='."
        if name in seen:
            return [], f"Line {line_no}: '{name}' is defined more than once."
        seen.add(name)
        definitions.append((name, expr))
    return definitions, None


def apply_custom_variables(df, definitions, scalar_context=None):
    """Adds one new column per (name, expression) pair to `df`, IN ORDER --
    so a later expression can reference an earlier custom column too (e.g.
    a speed_cm_s expression using a distance_cm defined just above it).
    `scalar_context` supplies extra names that aren't DataFrame columns
    (scale_factor, FPS, ...).

    Returns (df, warnings): df always carries every column that computed
    successfully. A single bad definition (unknown name, disallowed
    construct, a name colliding with a real tracking column, a runtime
    error like a numpy shape mismatch) is skipped with one warning message
    rather than aborting the whole run -- a typo in one custom variable
    shouldn't cost an otherwise-successful tracking run its
    raw_tracking.csv."""
    if not definitions:
        return df, []
    original_columns = set(df.columns)
    scalar_context = dict(scalar_context or {})
    warnings = []
    for name, expr in definitions:
        if name in original_columns:
            warnings.append(
                f"'{name}' is already a tracking column -- skipped the custom variable of the "
                "same name (pick a different name)."
            )
            continue
        names = {col: df[col] for col in df.columns}
        names.update(scalar_context)
        try:
            df[name] = safe_eval_expr(expr, names)
        except Exception as exc:
            warnings.append(f"'{name} = {expr}' could not be computed ({exc}) -- skipped.")
    return df, warnings
