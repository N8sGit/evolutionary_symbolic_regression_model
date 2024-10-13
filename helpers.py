import csv
from sympy import sympify

def get_next_batch_id(file_path='equations_catalogue.csv'):
    """
    Retrieves the next batch_id by reading the last batch_id from the CSV file and incrementing it.
    If the file is empty or doesn't exist, returns 1 as the first batch_id.
    """
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            batch_ids = [int(row[3]) for row in reader if row]  # Extract all batch IDs from the 4th column
            return max(batch_ids) + 1 if batch_ids else 1  # Return next batch ID
    except FileNotFoundError:
        return 1  # File doesn't exist, start with batch_id = 1

def map_equations_to_csv(equations, dataset_name, latent_dimensions, file_path='equations_catalogue.csv'):
    """
    Maps symbolic equations to a structured CSV with columns for equation, dataset, latent dimension, and batch ID.

    Parameters:
    - equations: A list of symbolic equations to store.
    - dataset_name: The name of the dataset (e.g., 'CaliforniaHousing').
    - latent_dimensions: A list of latent dimension names corresponding to the equations (e.g., ['latent_dim_1', 'latent_dim_2']).
    - file_path: The path to the CSV file (default: 'equations_catalogue.csv').
    """
    # Ensure that the number of latent dimensions matches the number of equations
    assert len(equations) == len(latent_dimensions), "Number of equations must match the number of latent dimensions."

    # Get the next batch ID
    batch_id = get_next_batch_id(file_path)

    # Open the CSV file in append mode
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Check if the file is empty to write the header
        if f.tell() == 0:
            writer.writerow(['equation', 'dataset', 'latent_dimension', 'batch_id'])  # Write header if file is new

        # Write each equation with its corresponding details
        for eq, latent_dim in zip(equations, latent_dimensions):
            writer.writerow([str(eq), dataset_name, latent_dim, batch_id])

def load_equations(file_path='equations_catalogue.csv', batch_id=int):
    """
    Loads the equations from the CSV file by batch id

    Parameters:
    - file_path: The path to the CSV file (default: 'equations_catalogue.csv').

    Returns:
    - A list of symbolic equations from chosen batch.
    """
    equations = []

    # Step 1: Read all rows from the CSV file
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)  # Load all rows into memory

    # Step 2: Find the latest batch_id (assuming batch_id is an integer)
    if rows:
        batch_id = max(int(row['batch_id']) for row in rows)

    # Step 3: Filter rows with the batch_id and convert the equations back to symbolic form
    for row in rows:
        if int(row['batch_id']) == batch_id:
            equations.append(sympify(row['equation']))

    return equations
