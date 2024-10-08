import operator
import math
import random
import numpy as np
from deap import gp, creator, base, tools, algorithms
from logging_config import logging
from safe_operators import safe_div, safe_exp, safe_log, safe_sqrt
from sympy import simplify, symbols, sympify

def perform_symbolic_regression_deap(X: np.ndarray, y: np.ndarray, feature_names: list, generations: int = 20) -> list:
    """
    Perform symbolic regression using DEAP to find symbolic equations that map X to y.
    
    Parameters:
    - X: Input features as a NumPy array.
    - y: Target variable as a NumPy array.
    - feature_names: List of feature names corresponding to X.
    - generations: Number of generations for the genetic programming algorithm.
    
    Returns:
    - List of SymPy expressions representing the best symbolic equations.
    """
    
    # Ensure unique creator registrations to prevent errors
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    # 1. Create a primitive set for symbolic regression with the correct number of input variables
    pset = gp.PrimitiveSet("MAIN", X.shape[1])  # Input variables count based on X's shape

    # 2. Rename arguments to match feature names
    for i, fname in enumerate(feature_names):
        pset.renameArguments(**{f'ARG{i}': fname})

    # 3. Add primitives to the primitive set
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(safe_div, 2)  # Use the safe division function
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(safe_exp, 1)  # Use the safe exponential function
    pset.addPrimitive(safe_log, 1)  # Use the safe logarithm function
    pset.addPrimitive(safe_sqrt, 1)  # Use the safe square root function

    # 4. Remove constants to prevent constant expressions
    # pset.addTerminal(1)  # Removed to prevent expressions like sin(cos(1))
    # pset.addTerminal(0)  # Already removed earlier

    # 5. Create symbols with feature names (if needed)
    x_symbols = symbols(feature_names)

    # 6. Create the fitness function (minimizing mean squared error)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)  # Increased min_=2 to discourage constants
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("compile", gp.compile, pset=pset)

    # 7. Register expr_mut before mutate
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # 8. Fitness evaluation function with safety checks for infinities and NaNs
    def eval_individual(individual, X, y):
        func = toolbox.compile(expr=individual)

        try:
            # Vectorized evaluation
            predictions = func(*X.T)  # Pass columns as separate arguments

            # Convert predictions to a NumPy array
            predictions = np.array(predictions)

            # If predictions are scalar, broadcast to match y
            if predictions.ndim == 0:
                predictions = np.full(y.shape, predictions)
            elif predictions.ndim == 1:
                if predictions.shape[0] != y.shape[0]:
                    raise ValueError(f"Predictions length {predictions.shape[0]} does not match target length {y.shape[0]}")
            else:
                raise ValueError(f"Unexpected predictions shape: {predictions.shape}")

            # Check for infinities or NaN values in the predictions
            if np.any(np.isinf(predictions)) or np.any(np.isnan(predictions)):
                return float('inf'),  # Penalize individual with infinite fitness score

            # Calculate MSE
            mse = np.mean((predictions - y) ** 2)
            logging.debug(f"Individual: {individual}, MSE: {mse}")
            return mse,

        except Exception as e:
            # If any error occurs during evaluation (e.g., math domain error), penalize with a high fitness value
            logging.error(f"Error evaluating individual {individual}: {e}")
            return float('inf'),

    toolbox.register("evaluate", eval_individual, X=X, y=y)
    toolbox.register("select", tools.selTournament, tournsize=5)  # Increased tournament size for diversity
    toolbox.register("mate", gp.cxOnePoint)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # 9. Evolutionary algorithm parameters
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 10. Run the evolutionary algorithm
    logging.info("Starting the evolution process for symbolic regression.")
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, generations, stats, halloffame=hof, verbose=True)

    # 11. Extract the best individual
    best_individual = hof[0]
    logging.info(f"Best individual: {best_individual}")

    # 12. Convert the best individual to a symbolic expression using SymPy
    sympy_expr_str = str(best_individual)
    logging.debug(f"String representation of the best individual: {sympy_expr_str}")

    # Convert the string to a SymPy expression
    try:
        best_expression_sympy = sympify(sympy_expr_str)
        best_expression_sympy = simplify(best_expression_sympy)
        logging.debug(f"Simplified SymPy expression: {best_expression_sympy}")
    except Exception as e:
        logging.error(f"Error converting to SymPy expression: {e}")
        best_expression_sympy = sympy_expr_str  # Fallback to string if parsing fails

    return [best_expression_sympy]