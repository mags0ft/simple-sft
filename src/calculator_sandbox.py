"""
This module implements a safe calculator tool that can evaluate mathematical
expressions without using eval() directly on user input, thus preventing code
execution vulnerabilities. It uses the ast module to parse and evaluate
expressions in a controlled manner.
"""

import ast
import math
from logging_manager import logger


def safe_factorial(x):
    """
    Safe factorial to prevent DoS via extremely large inputs.
    """

    val = int(x)

    if val > 100 or val < 0:
        raise ValueError("Factorial argument out of safe bounds")

    return math.factorial(val)


def safe_gamma(x):
    """
    Safe gamma to prevent DoS via extremely large inputs.
    """

    if x > 100 or x < -100:
        raise ValueError("Gamma argument out of safe bounds")

    return math.gamma(x)


def safe_pow(a, b):
    """
    Safe power function to prevent huge integer calculation DoS.
    """

    return math.pow(a, b)


_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "factorial": safe_factorial,
    "gamma": safe_gamma,
}

_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: float(a) + float(b),
    ast.Sub: lambda a, b: float(a) - float(b),
    ast.Mult: lambda a, b: float(a) * float(b),
    ast.Div: lambda a, b: float(a) / float(b),
    ast.Pow: safe_pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda x: +x,
    ast.USub: lambda x: -x,
}


def _eval(node, depth: int = 0):
    """
    Internal eval function that recursively evaluates an AST node according to
    the allowed operations.
    """

    if depth > 50:
        logger.error("Expression too deeply nested at depth %d", depth)
        raise ValueError("Expression too deeply nested")

    if isinstance(node, ast.Expression):
        return _eval(node.body, depth + 1)

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)

        raise ValueError("Invalid constant")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)

        if op_type not in _ALLOWED_BINOPS:
            raise ValueError("Operator not allowed")

        return _ALLOWED_BINOPS[op_type](
            _eval(node.left, depth + 1), _eval(node.right, depth + 1)
        )

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)

        if op_type not in _ALLOWED_UNARYOPS:
            raise ValueError("Unary operator not allowed")

        return _ALLOWED_UNARYOPS[op_type](_eval(node.operand, depth + 1))

    elif isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[node.id])

        raise ValueError(f"Unknown constant: {node.id}")

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Invalid function call")

        func_name = node.func.id

        if func_name not in _ALLOWED_FUNCS:
            raise ValueError(f"Function not allowed: {func_name}")

        if len(node.args) != 1:
            raise ValueError("Only single-argument functions allowed")

        return _ALLOWED_FUNCS[func_name](_eval(node.args[0], depth + 1))

    raise ValueError("Invalid expression")


def sandboxed_calculator_tool(args: dict[str, str]) -> dict:
    """
    The calculator tool that evaluates a mathematical expression provided in
    the "expression" key of the args dictionary.
    """

    expr = args.get("expression", "").strip()
    logger.debug("Calculator called with expression: %s", expr)

    if len(expr) > 200:
        raise ValueError("Expression too long")

    expr = expr.replace("^", "**")

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree)
    except Exception as e:
        logger.exception("Calculator failed to evaluate expression: %s", expr)
        raise ValueError(f"Invalid expression: {e}")

    return {"result": float(result)}
