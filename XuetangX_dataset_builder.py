# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class XetangBinaryDataset:
    def __init__(self, dataset_root, dataset_out):
        self.dataset_root = dataset_root
        self.dataset_out = dataset_out
        self.type_of_dataset = None

        self.courses_start_end_date = None
        self.user_info = None
        self.log_test_data = None
        self.log_train_data = None
        self.truth_test_data = None
        self.truth_train_data = None
        self.activities_order = None  

    def load_data(self):
        """Load Xetang dataset CSV files"""
        print("Loading dataset from:", self.dataset_root)

        self.courses_start_end_date = pd.read_csv(os.path.join(self.dataset_root, "course_info.csv"), header=0)
        self.user_info = pd.read_csv(os.path.join(self.dataset_root, "user_info.csv"), header=0)

        self.log_train_data = pd.read_csv(os.path.join(self.dataset_root, "prediction_log/train_log.csv"), header=0)
        self.truth_train_data = pd.read_csv(os.path.join(self.dataset_root, "prediction_log/train_truth.csv"), header=0)
        self.log_test_data = pd.read_csv(os.path.join(self.dataset_root, "prediction_log/test_log.csv"), header=0)
        self.truth_test_data = pd.read_csv(os.path.join(self.dataset_root, "prediction_log/test_truth.csv"), header=0)

        activities_train = list(self.log_train_data["action"])
        activities_test = list(self.log_test_data["action"])
        seen = set()
        self.activities_order = [x for x in activities_train + activities_test if not (x in seen or seen.add(x))]
        print(f"Activities order preserved ({len(self.activities_order)}): {self.activities_order}")

    def _trajectory_creator_for_each_student_(self, df_log, start_date, end_date):
        """Makes trajectory for each student while preserving the order of activities"""
        student_trajectory = []
        df_log = df_log.copy()
        df_log['time'] = pd.to_datetime(df_log['time'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
        df_log['date'] = df_log['time'].dt.date

        for day in pd.date_range(start_date, end_date, freq='D'):
            day_date = day.date()
            daily_activities = [0] * len(self.activities_order)
            if day_date in df_log['date'].values:
                day_logs = df_log[df_log['date'] == day_date]
                daily_activities = [
                    len(day_logs[day_logs['action'] == activity]) for activity in self.activities_order
                ]
            student_trajectory.append(daily_activities)

        return np.array(student_trajectory).T  #shape: activities × days

    def _trajectory_creator_for_all_students_(self, is_train_set=True):
        """Make trajectories for all students"""
        if is_train_set:
            df_log = self.log_train_data
            df_labels = self.truth_train_data
            self.type_of_dataset = "train"
        else:
            df_log = self.log_test_data
            df_labels = self.truth_test_data
            self.type_of_dataset = "test"

        all_students_trajectories = {}

        for _, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc=f"Creating {self.type_of_dataset} trajectories"):
            enroll_id = row["enroll_id"]
            label = row["truth"]

            student_logs = df_log[df_log["enroll_id"] == enroll_id]
            if student_logs.empty:
                continue

            course_id = student_logs.iloc[0]["course_id"]
            course_info = self.courses_start_end_date[self.courses_start_end_date["course_id"] == course_id]
            if course_info.empty:
                continue

            starting_date = pd.to_datetime(course_info["start"].values[0])
            ending_date = pd.to_datetime(course_info["end"].values[0])

            student_trajectory = self._trajectory_creator_for_each_student_(student_logs, starting_date, ending_date)

            all_students_trajectories[enroll_id] = {
                "trajectory": student_trajectory,
                "label": label,
                "length_of_activities": len(student_trajectory),
                "n_days": student_trajectory.shape[1],
            }

        return all_students_trajectories

    def create_ts_file(self, all_traj, split_name):
        """Make .ts file in UEA format"""
        trajectories = []
        labels = []
        max_length = 0
        n_dims = 0

        for enroll_id, data in all_traj.items():
            traj = data["trajectory"]
            dim_strings = [",".join(map(str, series)) for series in traj]
            trajectories.append(":".join(dim_strings))
            labels.append(str(data["label"]))
            n_dims = data["length_of_activities"]
            if max_length < data["n_days"]:
                max_length = data["n_days"]

        ts_path = os.path.join(self.dataset_out, f"Xetang_Binary_{split_name.upper()}.ts")

        header = [
            "@problemName Xetang_dropout_binary",
            "@timeStamps false",
            "@missing false",
            "@univariate false",
            f"@dimensions {n_dims}",
            "@equalLength true",
            f"@seriesLength {max_length}",
            "@classLabel true 0 1",
            "@data"
        ]

        with open(ts_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
            for tr, label in zip(trajectories, labels):
                f.write(f"{tr}:{label}\n")

        print(f"Created {ts_path} with {len(trajectories)} samples")
        return ts_path

    def create_binary_dataset(self):
        """Main pipeline"""
        self.load_data()
        print("\nProcessing training set...")
        train_trajectories = self._trajectory_creator_for_all_students_(is_train_set=True)
        print("\nProcessing test set...")
        test_trajectories = self._trajectory_creator_for_all_students_(is_train_set=False)

        if len(train_trajectories) == 0 or len(test_trajectories) == 0:
            print("No trajectories created. Please check dataset format.")
            return

        self.create_ts_file(train_trajectories, "TRAIN")
        self.create_ts_file(test_trajectories, "TEST")

        metadata = {
            "n_activities": len(self.activities_order),
            "activities_order": self.activities_order
        }
        with open(os.path.join(self.dataset_out, "xetang_binary_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print("Binary dataset created successfully!")
        return train_trajectories, test_trajectories


if __name__ == "__main__":
    Original_Dataset_Unprocessed_ROOT = r"UniTS-main\\dataset\\XuetangX\\XuetangX_raw"
    Dataset_Processed_OUT = r"UniTS-main\\dataset\\XuetangX\\XuetangX_processed"

    if not os.path.exists(Dataset_Processed_OUT):
        os.makedirs(Dataset_Processed_OUT)

    processor = XetangBinaryDataset(
        dataset_root=Original_Dataset_Unprocessed_ROOT,
        dataset_out=Dataset_Processed_OUT
    )

    processor.create_binary_dataset()
