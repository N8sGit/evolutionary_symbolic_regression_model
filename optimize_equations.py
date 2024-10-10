from sympy import trigsimp, simplify, powsimp, cancel, symbols, Basic, count_ops, symbols, lambdify, log, sqrt, exp, div 
from safe_operators import safe_log, safe_exp, safe_sqrt, safe_div
import time
import numpy as np
from sympy import count_ops, symbols, lambdify, log as sympy_log, sqrt as sympy_sqrt, exp as sympy_exp


def substitute_safe_functions(expr):
    # Substitute safe functions with regular SymPy functions
    expr = expr.subs(safe_log, sympy_log)
    expr = expr.subs(safe_sqrt, sympy_sqrt)
    expr = expr.subs(safe_exp, sympy_exp)
    # Replace safe_div(x, y) with x / y
    expr = expr.replace(
        lambda expr: expr.func == safe_div,
        lambda expr: expr.args[0] / expr.args[1]
    )
    return expr

def substitute_back_safe_functions(expr):
    # Substitute back regular functions with safe functions
    expr = expr.subs(sympy_log, safe_log)
    expr = expr.subs(sympy_sqrt, safe_sqrt)
    expr = expr.subs(sympy_exp, safe_exp)
    # Replace x / y with safe_div(x, y)
    expr = expr.replace(
        lambda expr: expr.is_Div,
        lambda expr: safe_div(expr.args[0], expr.args[1])
    )
    return expr


def decompose_and_simplify(expr):
    """
    Recursively decompose a nested expression, simplify each subexpression, and recombine.
    
    Parameters:
    - expr: The symbolic expression to decompose and simplify.
    
    Returns:
    - The fully simplified expression.
    """
    # Base case: If the expression is atomic (cannot be decomposed), return it as is.
    if not isinstance(expr, Basic) or expr.is_Atom:
        return expr

    # Recursively simplify the arguments of the expression
    simplified_args = [decompose_and_simplify(arg) for arg in expr.args]

    # Reconstruct the expression with simplified subexpressions
    simplified_expr = expr.func(*simplified_args)

    # Apply simplification strategies to the recomposed expression
    simplified_expr = simplify(simplified_expr)  # General simplification
    simplified_expr = trigsimp(simplified_expr)  # Trigonometric simplification
    simplified_expr = powsimp(simplified_expr, deep=True)  # Simplify powers
    simplified_expr = cancel(simplified_expr)  # Cancel common terms

    return simplified_expr

def optimize_equations(loaded_eqs):
    """
    Decompose and simplify the loaded symbolic equations with handling of safe functions.

    Parameters:
    - loaded_eqs: List of symbolic equations.

    Returns:
    - List of optimized symbolic expressions.
    """
    simplified_eqs = []
    
    for eq in loaded_eqs:
        # 1. Substitute safe functions with regular functions
        substituted_expr = substitute_safe_functions(eq)
        
        # 2. Decompose and simplify the expression
        simplified_expr = decompose_and_simplify(substituted_expr)
        
        # 3. Substitute back safe functions
        simplified_expr = substitute_back_safe_functions(simplified_expr)
        
        # 4. Append the optimized symbolic expression
        simplified_eqs.append(simplified_expr)
    
    return simplified_eqs

def compare_equations(orig_eq, opt_eq, X, y_true):
    """
    Compare two equations in terms of length, complexity, execution time, and accuracy.
    
    Parameters:
    - orig_eq: Original symbolic equation.
    - opt_eq: Optimized symbolic equation.
    - X: Input data for evaluating execution time and accuracy.
    - y_true: Ground truth values to compare accuracy.
    
    Returns:
    - Dictionary of comparison metrics.
    """
    metrics = {}

    # 1. Length difference (simplistic)
    orig_len = len(str(orig_eq))
    opt_len = len(str(opt_eq))
    metrics['length_diff'] = orig_len - opt_len
    print(f"LENGTH DIFF: {metrics['length_diff']}")

    # 2. Complexity difference (based on number of operations)
    orig_complexity = count_ops(orig_eq)
    opt_complexity = count_ops(opt_eq)
    metrics['complexity_diff'] = orig_complexity - opt_complexity
    print(f"COMPLEXITY DIFF: {metrics['complexity_diff']} (Original: {orig_complexity}, Optimized: {opt_complexity})")

    # 3. Execution time comparison
    feature_symbols = symbols('HouseAge AveRooms PRICE')
    custom_modules = {'safe_log': safe_log, 'safe_sqrt': safe_sqrt, 'safe_exp': safe_exp, 'safe_div': safe_div, 'numpy': np}
    
    orig_func = lambdify(feature_symbols, orig_eq, modules=[custom_modules, 'numpy'])
    opt_func = lambdify(feature_symbols, opt_eq, modules=[custom_modules, 'numpy'])

    start_time = time.time()
    orig_pred = np.array([orig_func(*x) for x in X])
    orig_exec_time = time.time() - start_time

    start_time = time.time()
    opt_pred = np.array([opt_func(*x) for x in X])
    opt_exec_time = time.time() - start_time

    metrics['orig_exec_time'] = orig_exec_time
    metrics['opt_exec_time'] = opt_exec_time
    metrics['exec_time_diff'] = orig_exec_time - opt_exec_time
    print(f"EXECUTION TIME DIFF: {metrics['exec_time_diff']:.6f} seconds")

    # 4. Accuracy difference (using MSE as an example)
    orig_mse = np.mean((orig_pred - y_true) ** 2)
    opt_mse = np.mean((opt_pred - y_true) ** 2)
    metrics['accuracy_diff'] = orig_mse - opt_mse
    print(f"ACCURACY DIFF: {metrics['accuracy_diff']:.6f} (Original MSE: {orig_mse:.6f}, Optimized MSE: {opt_mse:.6f})")

    return metrics