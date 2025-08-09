import pandas as pd
import numpy as np
import os
import glob
import ast
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm  

class UnitelmaBinaryDataset:
    def __init__(self, origin_path, end_path, dropout_threshold=0.5, test_size=0.2, random_state=42):
        self.origin = origin_path
        self.end = end_path
        self.dropout_threshold = dropout_threshold
        self.test_size = test_size
        self.random_state = random_state

    def load_dataset(self):
        files = glob.glob(f"{self.origin}/*/*.csv")
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.origin}")
        dataframes = []
        for file in files:
            df = pd.read_csv(file)
            dataframes.append(df)
        full_df = pd.concat(dataframes, ignore_index=True)
        return full_df

    def process(self, df):
        matrices = []
        labels = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            ts = np.asarray(ast.literal_eval(row['timeseries']), dtype=float)
            label = 1 if row['dropout'] > self.dropout_threshold else 0
            matrices.append(ts)
            labels.append(label)
        matrices = np.array(matrices)
        labels = np.array(labels)
        return matrices, labels

    def write_ts(self, X, y, split_name, write_header=False):
        os.makedirs(self.end, exist_ok=True)
        path = os.path.join(self.end, f"Unitelma_BINARY_{split_name}.ts")
        mode = "w" if write_header else "a"
        with open(path, mode, encoding="utf-8") as f:
            if write_header:
                header = [
                    "@problemName Unitelma_dropout_binary",
                    "@timeStamps false",
                    "@missing false",
                    "@univariate false",
                    f"@dimensions {X.shape[2]}",
                    "@equalLength true",
                    f"@seriesLength {X.shape[1]}",
                    "@classLabel true 0 1",
                    "@data"
                ]
                f.write("\n".join(header) + "\n")

            X_formatted = X.swapaxes(1, 2)
            # tqdm per monitorare scrittura file
            for i, sample in tqdm(enumerate(X_formatted), total=len(X_formatted), desc=f"Writing {split_name}"):
                dims = []
                for dim in sample:
                    dims.append(",".join(map(str, dim)))
                timeseries_str = ":".join(dims)
                line = f"{timeseries_str}:{y[i]}"
                f.write(line + "\n")

    def run(self):
        print("Loading dataset...")
        df = self.load_dataset()
        print(f"Loaded {len(df)} samples")

        X, y = self.process(df)
        print(f"Matrix shape: {X.shape}, Labels distribution: {np.bincount(y)}")

        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        self.write_ts(X_train, y_train, "TRAIN", write_header=True)
        self.write_ts(X_test, y_test, "TEST", write_header=True)

        metadata = {
            "dropout_threshold": self.dropout_threshold,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_dims": X.shape[2],
            "series_length": X.shape[1]
        }
        with open(os.path.join(self.end, "binary_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Binary dataset created at {self.end}")

if __name__ == "__main__":
    config = {
        "origin_path": "UniTS-main\\dataset\\Unitelma\\Unitelma_raw",
        "end_path": "UniTS-main\\dataset\\Unitelma\\Unitelma_processed",
        "dropout_threshold": 0.5,
        "test_size": 0.2,
        "random_state": 42
    }
    processor = UnitelmaBinaryDataset(**config)
    processor.run()
