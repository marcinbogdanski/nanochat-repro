import ast
import operator

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    # ast.Pow: operator.pow,   # not needed in GSM8K, and potential resource hog
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def _evaluate(node):
    """Recursively evaluate AST tree, allow only a basic calculator and 'strawberry'.count('r')"""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    elif isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary operator {type(node.op)}")
        return op(_evaluate(node.left), _evaluate(node.right))
    elif isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPS.get(type(node.op))
        if operation is None:
            raise ValueError(f"Unsupported unary operator {type(node.op)}")
        return operation(_evaluate(node.operand))
    elif isinstance(node, ast.Constant):
        if type(node.value) in (int, float):               # explicitly only int and float, not bool or complex
            return node.value
        else:
            raise ValueError(f"Unsupported constant type {type(node.value)}")
    elif isinstance(node, ast.Call):
        # allow only exact shape of: "abc".count("a")
        if (
            isinstance(node.func, ast.Attribute)           # must be attribute "abc".count
            and node.func.attr == "count"                  # only method called "count"
            and isinstance(node.func.value, ast.Constant)  # must be called on a literal constant "abc"
            and type(node.func.value.value) is str         # the constant must specifically be a string
            and len(node.args) == 1                        # exactly one argument to count()
            and isinstance(node.args[0], ast.Constant)     # argument must be a constant literal constant
            and type(node.args[0].value) is str            # the argument must specifically be a string
            and not node.keywords                          # no keyword arguments allowed
        ):
            return node.func.value.value.count(node.args[0].value)
        raise ValueError("Unsupported function call")
    else:
        raise ValueError(f"Unsupported AST node type: {type(node)}")

def ast_eval(expr):
    try:
        node = ast.parse(expr, mode='eval')
        return _evaluate(node)
    except (SyntaxError, ValueError, ArithmeticError):  # invalid syntax, unsupported op, div by zero
        return None


class CalculatorAndCounter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.python_start = self.tokenizer.encode_single_token('<|python_start|>')
        self.python_end = self.tokenizer.encode_single_token('<|python_end|>')
        self.output_start = self.tokenizer.encode_single_token('<|output_start|>')
        self.output_end = self.tokenizer.encode_single_token('<|output_end|>')

    @property
    def tool_trigger_token(self):
        return self.python_end

    def _find_start_of_python_block(self, tokens):
        # Find most recent python_start or python_end token
        for i in range(len(tokens)-2, -1, -1):
            if tokens[i] == self.python_start:
                return i + 1  # return index of first token after python_start
            elif tokens[i] == self.python_end:
                return None  # [python_end, ... python_end] - invalid block
        return None  # no python_start found

    def handle_tool_call(self, tokens):
        assert tokens[-1] == self.python_end and len(tokens) >= 2
        
        start_idx = self._find_start_of_python_block(tokens)
        if start_idx is None:
            return None  # nothing to do
        expr_str = self.tokenizer.decode(tokens[start_idx:-1])  # -1 skip python_end
        result = ast_eval(expr_str)
        if result is None:
            return None
        tokens_result = [self.output_start] + self.tokenizer.encode(str(result)) + [self.output_end]
        return tokens_result

