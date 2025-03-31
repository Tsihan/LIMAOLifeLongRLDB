import re
import numpy as np
import pickle
from kmodes.kprototypes import KPrototypes
from featurize import ALL_TYPES
from net import NUM_OTHER_HUB, NUM_HASHJOIN_HUB, NUM_NESTEDLOOP_HUB
class Kproto_MultiArrayProcessor:
    def __init__(self, file_path=None, num_other=1, num_hashjoin=2, num_nestedloop=3, num_operator=7):
        """
        Initialize the object. If file_path is provided, read data from the file and initialize.
        If file_path is None, assume the object will be loaded via load_from_disk.
        """
        if file_path is not None:
            self.file_path = file_path
            self.num_other = num_other
            self.num_hashjoin = num_hashjoin
            self.num_nestedloop = num_nestedloop
            self.num_operator = num_operator
            # Store raw data points by groups
            self.data_groups = {"a": [], "b": [], "c": []}
            self._read_file(file_path)

            # Initialize dictionaries for processed data, max array count, and categorical indices per group
            self.processed_data = {}       # Processed matrix for each group (each row is a data point)
            self.max_array_count = {}      # Maximum number of arrays for data points in each group
            self.categorical_indices = {}  # Categorical feature indices for each group

            for group in self.data_groups:
                proc_matrix, max_count = self._process_group(self.data_groups[group])
                self.processed_data[group] = proc_matrix
                self.max_array_count[group] = max_count
                # For each array, the first num_operator features are categorical
                cat_idx = []
                for i in range(max_count):
                    cat_idx.extend(list(range(i * (num_operator + 3), i * (num_operator + 3) + num_operator)))
                self.categorical_indices[group] = cat_idx if cat_idx else [0]

            # Initialize KPrototypes models
            self.kproto = {
                "a": KPrototypes(n_clusters=num_other, init='Huang', random_state=42),
                "b": KPrototypes(n_clusters=num_hashjoin, init='Huang', random_state=42),
                "c": KPrototypes(n_clusters=num_nestedloop, init='Huang', random_state=42)
            }
            # Train the models with the processed data for each group
            for group in self.processed_data:
                self.kproto[group].fit_predict(
                    self.processed_data[group],
                    categorical=self.categorical_indices[group]
                )
        else:
            # When file_path is None, assume the object is loaded via deserialization.
            pass

    def _read_file(self, file_path):
        """
        Read the file line by line, supporting multi-line data points.
        When a new data point is detected, use controlled eval to parse the string and store it in the appropriate group.
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
                    if current_entry:
                        self._process_entry(current_group, current_entry)
                    parts = line.split(":", 1)
                    if len(parts) != 2:
                        print("Skipping improperly formatted line:", line)
                        continue
                    current_group = parts[0].strip()[0].lower()
                    current_entry = parts[1].strip()
                else:
                    current_entry += " " + line.strip()
            if current_entry:
                self._process_entry(current_group, current_entry)

    def _process_entry(self, group, entry_str):
        """
        Parse the data point string using controlled eval (only allowing numpy.array),
        and store the parsed object in the corresponding group.
        """
        try:
            data_obj = eval(entry_str, {"array": np.array})
            if group in self.data_groups:
                self.data_groups[group].append(data_obj)
            else:
                print(f"Unknown group {group}, data point: {entry_str}")
        except Exception as e:
            print(f"Error parsing entry: {entry_str}, error: {e}")

    def _extract_arrays(self, data):
        """
        Recursively extract numpy arrays from the nested structure.
        Each array must be one-dimensional and of length (num_operator+3).
        Returns a list of arrays.
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
        Process the data points in a group:
          1. Recursively extract all arrays from each data point.
          2. Determine the maximum number of arrays (max_count) in this group.
          3. Pad data points with fewer arrays with zero arrays; truncate if more.
          4. Concatenate the arrays of each data point into a single long vector.
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
          - group: 'a', 'b', or 'c' (choosing which model to use).
          - data_point: A nested data structure similar to the training data.
        Process:
          1. Recursively extract valid arrays.
          2. Pad or truncate according to the model requirements.
          3. Concatenate into a vector and use the corresponding model to predict.
        Returns the predicted cluster label (as an int).
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
        return int(label)

    def save(self, file_path):
        """
        Serialize the current object and save it to the specified file.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_disk(cls, file_path):
        """
        Load a serialized object from the specified file and return it.
        """
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj

# -------------------------
# Testing code example:
if __name__ == '__main__':
    # Construct a sample data point for group 'a' (similar to the file format)
    sample_a = (
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
    
    from time import time
    start_time = time()
    processor = Kproto_MultiArrayProcessor("/mydata/LIMAOLifeLongRLDB/module_assigner_init.txt",
                                                           num_other=NUM_OTHER_HUB, num_hashjoin=NUM_HASHJOIN_HUB, num_nestedloop=NUM_NESTEDLOOP_HUB,num_operator=len(ALL_TYPES))
    end_time = time()
    print("Time taken to read data:", end_time - start_time)
    
    start_time = time()
    print("Prediction for group a:", processor.predict("a", sample_a))
    end_time = time()
    print("Time taken for prediction:", end_time - start_time)
    
    # Save the object to disk
    processor.save("/mydata/LIMAOLifeLongRLDB/kproto_processor.pkl")
    
    # Later, load the object from disk directly
    start_time = time()
    loaded_processor = Kproto_MultiArrayProcessor.load_from_disk("/mydata/LIMAOLifeLongRLDB/kproto_processor.pkl")
    end_time = time()
    print("Time taken to load object:", end_time - start_time)
    
    start_time = time()
    print("Prediction for group a from loaded object:", loaded_processor.predict("a", sample_a))
    end_time = time()
    print("Time taken for prediction:", end_time - start_time)
