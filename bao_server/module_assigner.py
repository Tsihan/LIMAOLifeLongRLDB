import re
import numpy as np
from kmodes.kprototypes import KPrototypes

class Kproto_MultiArrayProcessor:
    def __init__(self, file_path, num_other=1, num_hashjoin=2,num_nestedloop=3,  num_operator=7):
        """
        Read data from file, where each data point may span multiple lines.
        Example format:
        
        a[0]:
        (array([...]), (array([...]),
         (array([...]), (array([...]), (array([...]), 'movie_companies'), ...))
         ...
        
        Each data point contains a nested structure with multiple arrays.
        Each array is a vector of length (num_operator+3) (here 10), where the first num_operator (7)
        elements are categorical (0/1) and the last 3 elements are numeric.
        """
        self.file_path = file_path
        self.num_other = num_other
        self.num_hashjoin = num_hashjoin
        self.num_nestedloop = num_nestedloop
        self.num_operator = num_operator
        # Store data for groups a, b, and c (each element is a parsed nested structure)
        self.data_groups = {"a": [], "b": [], "c": []}
        self._read_file(file_path)
        
        # Process each group: extract arrays, determine the maximum number of arrays per data point,
        # and pad/truncate each data point to a fixed vector length.
        self.processed_data = {}       # Processed matrix for each group (each row is a data point)
        self.max_array_count = {}      # Maximum number of arrays in any data point for each group
        self.categorical_indices = {}  # Categorical feature indices for each group (first num_operator values in each array)
        
        for group in self.data_groups:
            proc_matrix, max_count = self._process_group(self.data_groups[group])
            self.processed_data[group] = proc_matrix
            self.max_array_count[group] = max_count
            # Each array contributes num_operator categorical features.
            # For each array, the categorical indices span from i*(num_operator+3) to i*(num_operator+3)+num_operator-1.
            cat_idx = []
            for i in range(max_count):
                cat_idx.extend(list(range(i * (num_operator + 3), i * (num_operator + 3) + num_operator)))
            # Ensure a non-empty list is passed to KPrototypes.
            self.categorical_indices[group] = cat_idx if cat_idx else [0]
        
        # Initialize three KPrototypes models (cluster numbers are adjustable)
        self.kproto = {
            "a": KPrototypes(n_clusters=num_other, init='Huang', random_state=42),
            "b": KPrototypes(n_clusters=num_hashjoin, init='Huang', random_state=42),
            "c": KPrototypes(n_clusters=num_nestedloop, init='Huang', random_state=42)
        }
        # Train each model using its respective processed data and categorical indices.
        for group in self.processed_data:
            self.kproto[group].fit_predict(
                self.processed_data[group],
                categorical=self.categorical_indices[group]
            )
    
    def _read_file(self, file_path):
        """
        Read the file line by line, supporting multi-line data points.
        Rules:
          - If a line matches the regex "^[abc]\[\d+\]:", it is the start of a new data point.
          - Otherwise, the line is appended to the previous data point.
        After reading, use a controlled eval (allowing only numpy.array) to parse the string into a Python object,
        and store it in self.data_groups based on its group.
        """
        pattern = re.compile(r"^[abc]\[\d+\]:")
        current_entry = ""
        current_group = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.rstrip()  # Remove trailing newline
                if not line:
                    continue
                if pattern.match(line):
                    # If there is an existing data point, parse and save it.
                    if current_entry:
                        self._process_entry(current_group, current_entry)
                    # New data point: extract the group (first character) and the content after the colon.
                    parts = line.split(":", 1)
                    if len(parts) != 2:
                        print("Skipping line with incorrect format:", line)
                        continue
                    current_group = parts[0].strip()[0].lower()
                    current_entry = parts[1].strip()
                else:
                    # Append non-starting lines to the current data point.
                    current_entry += " " + line.strip()
            # Process the last data point.
            if current_entry:
                self._process_entry(current_group, current_entry)
    
    def _process_entry(self, group, entry_str):
        """
        Parse a single data point string (entry_str) using controlled eval (only allowing numpy.array),
        and store the result in self.data_groups[group].
        """
        try:
            data_obj = eval(entry_str, {"array": np.array})
            if group in self.data_groups:
                self.data_groups[group].append(data_obj)
            else:
                print(f"Unknown group {group}, data point: {entry_str}")
        except Exception as e:
            print(f"Error parsing entry: {entry_str}, error message: {e}")
    
    def _extract_arrays(self, data):
        """
        Recursively extract numpy arrays from the nested structure.
        Each array must be one-dimensional and of length (num_operator+3).
        Returns a list of arrays in the order they are found.
        """
        arrays = []
        if isinstance(data, np.ndarray):
            if data.ndim == 1 and len(data) == (self.num_operator + 3):
                arrays.append(data)
            else:
                try:
                    for item in data:
                        arrays.extend(self._extract_arrays(item))
                except Exception:
                    pass
        elif isinstance(data, (list, tuple)):
            for item in data:
                arrays.extend(self._extract_arrays(item))
        return arrays

    def _process_group(self, data_points):
        """
        Process all data points in a group (a, b, or c):
          1. For each data point, recursively extract all arrays (each must be one-dimensional of length 10).
          2. Determine the maximum number of arrays (max_count) in this group.
          3. If a data point has fewer arrays, pad it with zero arrays; if it has more, truncate them (or use another strategy).
          4. Concatenate all arrays of the data point into a single long vector.
        Returns the processed matrix and max_count.
        """
        processed_points = []
        counts = []
        extracted_list = []
        for dp in data_points:
            arrays = self._extract_arrays(dp)
            if not arrays:
                arrays = [np.zeros(self.num_operator + 3)]
            extracted_list.append(arrays)
            counts.append(len(arrays))
        max_count = max(counts) if counts else 1
        if max_count == 0:
            max_count = 1
        for arrays in extracted_list:
            if len(arrays) < max_count:
                for _ in range(max_count - len(arrays)):
                    arrays.append(np.zeros(10))
            elif len(arrays) > max_count:
                arrays = arrays[:max_count]
            flattened = np.concatenate(arrays)
            processed_points.append(flattened)
        return np.array(processed_points), max_count

    def predict(self, group, data_point):
        """
        Predict the cluster for a new data point.
        Parameters:
          - group: 'a', 'b', or 'c' (selects which model to use).
          - data_point: A nested data structure similar to the training data.
        Processing:
          1. Recursively extract all valid arrays.
          2. If the number of arrays is insufficient, pad with zero arrays; if too many, truncate.
          3. Concatenate them into a vector and use the corresponding model to predict.
        Returns the predicted cluster label.
        """
        if group not in self.data_groups:
            raise ValueError("group must be 'a', 'b', or 'c'")
        arrays = self._extract_arrays(data_point)
        req_count = self.max_array_count[group]
        if len(arrays) < req_count:
            for _ in range(req_count - len(arrays)):
                arrays.append(np.zeros(10))
        elif len(arrays) > req_count:
            arrays = arrays[:req_count]
        flattened = np.concatenate(arrays)
        dp = np.array([flattened])
        label = self.kproto[group].predict(dp, categorical=self.categorical_indices[group])[0]
        # chanege the label from numpy.uint16 to int
        label = int(label)
        return label

# -------------------------
# Testing code: Predict separately for a[0], b[0], and c[0]
if __name__ == '__main__':
    # Assume the data is stored in "/mydata/LIMAOLifeLongRLDB/module_assigner_init.txt"
    processor = Kproto_MultiArrayProcessor("/mydata/LIMAOLifeLongRLDB/module_assigner_init.txt")
    
    # Construct sample a[0] data point (nested structure similar to file format)
    a0 = (
        np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        (
            np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
            (
                np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                (
                    np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                    (np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'movie_companies'),
                    (np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'company_type')
                ),
                (np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'title')
            ),
        ),
        (np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'movie_info'),
        (np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'info_type')
    )
    
    # Construct sample b[0] data point
    b0 = (
        np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.79680973, 0.]),
        (
            np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.79680936, 0.]),
            (
                np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.79680138, 0.]),
                (
                    np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.7967428, 0.04842076]),
                    (np.array([0., 0., 0., 1., 0., 0., 0., 0.79372568, 0.79674019, 0.13593427]), 'movie_companies'),
                    (np.array([0., 0., 0., 1., 0., 0., 0., 0.05590502, 0.04496154, 0.]), 'company_type')
                ),
                (np.array([0., 0., 0., 0., 1., 0., 0., 0.84615984, 0.16337451, 0.]), 'title')
            ),
            (np.array([0., 0., 0., 0., 1., 0., 0., 0.96396017, 0.10472773, 0.02832433]), 'movie_info'),
            (np.array([0., 0., 0., 0., 0., 1., 0., 0.05590502, 0.04496154, 0.]), 'info_type')
        )
    )
    
    # Construct sample c[0] data point
    c0 = (
        np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        (
            np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
            (
                np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                (
                    np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.7967428, 0.04842076]),
                    (np.array([0., 0., 0., 1., 0., 0., 0., 0.79372568, 0.79674019, 0.13593427]), 'movie_companies'),
                    (np.array([0., 0., 0., 1., 0., 0., 0., 0.05590502, 0.04496154, 0.]), 'company_type')
                ),
                (np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'title')
            ),
        ),
        (np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'movie_info'),
        (np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'info_type')
    )
    
    print("Prediction for group a:", processor.predict("a", a0))
    print("Prediction for group b:", processor.predict("b", b0))
    print("Prediction for group c:", processor.predict("c", c0))
