import pandas as pd
import numpy as np
import os
import zipfile
import datetime
from tqdm import tqdm
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore', category=FutureWarning)

class KDDCup15BinaryDataset:
    def __init__(self, dataset_root, dataset_out):
        self.dataset_root = dataset_root
        self.dataset_out = dataset_out

    def _extract_zip_file_if_needed_(self, zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_contents = zip_ref.namelist()
            files_to_extract = []
            for file_name in zip_contents:
                full_path = os.path.join(self.dataset_root, file_name)
                if not os.path.exists(full_path):
                    files_to_extract.append(file_name)
            if files_to_extract:
                print(f"Extracting {len(files_to_extract)} files from {zip_path}...")
                zip_ref.extractall(self.dataset_root)
                print(f"Extraction completed in: {self.dataset_root}")

    def load_data(self):
        print("Loading dataset...")

        try:
            self.courses_start_end_date = pd.read_csv(os.path.join(self.dataset_root, "date.csv"))
            self.courses_info = pd.read_csv(os.path.join(self.dataset_root, "object.csv"))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        try:
            self._extract_zip_file_if_needed_(os.path.join(self.dataset_root, "train.zip"))
            self._extract_zip_file_if_needed_(os.path.join(self.dataset_root, "test.zip"))

            self.enrollment_train_data = pd.read_csv(os.path.join(self.dataset_root, "train/enrollment_train.csv"))
            self.log_train_data = pd.read_csv(os.path.join(self.dataset_root, "train/log_train.csv"))
            self.truth_train_data = pd.read_csv(os.path.join(self.dataset_root, "train/truth_train.csv"), header=None)

            self.enrollment_test_data = pd.read_csv(os.path.join(self.dataset_root, "test/enrollment_test.csv"))
            self.log_test_data = pd.read_csv(os.path.join(self.dataset_root, "test/log_test.csv"))
            self.truth_test_data = pd.read_csv(os.path.join(self.dataset_root, "test/truth_test.csv"), header=None)

        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        print("Dataset loaded successfully.")

    def _trajectory_creator_for_each_student_(self, df_log, activities, start_date, end_date):        
        student_trajectory = []

        df_log = df_log.copy()
        df_log['time'] = pd.to_datetime(df_log['time'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
        df_log['date'] = df_log['time'].dt.date
        start_date = start_date.date()
        end_date = end_date.date()
        
        for day in pd.date_range(start_date, end_date, freq='D'): 
            day_date = day.date()
            daily_activities = [0] * len(activities)
            
            if day_date in df_log['date'].values:
                day_logs = df_log[df_log['date'] == day_date]
                daily_activities = [
                    len(day_logs[day_logs['event'] == activity]) for activity in activities]

            student_trajectory.append(daily_activities)
            
        return np.array(student_trajectory).T

    def _trajectory_creator_for_all_students_(self, is_train_set=True):
        if is_train_set:
            df_log = self.log_train_data
            df_enrollment = self.enrollment_train_data
            df_labels = self.truth_train_data
        else:
            df_log = self.log_test_data
            df_enrollment = self.enrollment_test_data
            df_labels = self.truth_test_data

        all_students_trajectories = {}
        activities = df_log["event"].unique()
        
        for idx, enrollment_row in tqdm(df_enrollment.iterrows(), 
                                        desc="Creating trajectories", 
                                        total=len(df_enrollment)):
            enrollment_id = enrollment_row["enrollment_id"]
            course_id = enrollment_row["course_id"]
            
            # Get course dates
            course_dates = self.courses_start_end_date[self.courses_start_end_date["course_id"] == course_id]
            if len(course_dates) == 0:
                continue
                
            starting_date = pd.to_datetime(course_dates["from"].values[0])
            ending_date = pd.to_datetime(course_dates["to"].values[0])
            
            # Get binary label
            label_row = df_labels[df_labels.iloc[:,0] == enrollment_id]
            if len(label_row) == 0:
                continue
            label = int(label_row.iloc[0, 1])
            
            # Create trajectory
            student_trajectory = self._trajectory_creator_for_each_student_(
                df_log[df_log["enrollment_id"] == enrollment_id], 
                activities, starting_date, ending_date)
            
            all_students_trajectories[enrollment_id] = {
                "trajectory": student_trajectory,
                "label": label,
                "n_days": len(student_trajectory[0])
            }
        
        return all_students_trajectories

    def create_binary_dataset(self, test_size=0.2, random_state=42):
        print("CREATING KDDCup BINARY DATASET...")

        self.load_data()

        print("\nProcessing training data...")
        train_trajectories = self._trajectory_creator_for_all_students_(is_train_set=True)

        print("\nProcessing test data...")
        test_trajectories = self._trajectory_creator_for_all_students_(is_train_set=False)

        all_trajectories = {**train_trajectories, **test_trajectories}

        trajectories_list = []
        labels_list = []

        for enrollment_id, data in all_trajectories.items():
            traj = data["trajectory"]
            dim_strings = [",".join(map(str, series)) for series in traj]
            trajectories_list.append(":".join(dim_strings))
            labels_list.append(data["label"])

        first_traj = list(all_trajectories.values())[0]["trajectory"]
        n_dims = len(first_traj)
        max_length = max(data["n_days"] for data in all_trajectories.values())

        if len(set(labels_list)) == 1:
            stratify = None
        else:
            stratify = labels_list

        indices = np.arange(len(trajectories_list))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=stratify
        )

        def write_split(name, idx_list, write_header=False):
            os.makedirs(self.dataset_out, exist_ok=True)
            path = os.path.join(self.dataset_out, f"KDDCup_Binary_{name}.ts")
            mode = "w" if write_header else "a"
            with open(path, mode, encoding="utf-8") as f:
                if write_header:
                    header = [
                        "@problemName KDDCup_binary_dropout",
                        "@timeStamps false", "@missing false", "@univariate false",
                        f"@dimensions {n_dims}", "@equalLength true", f"@seriesLength {max_length}",
                        "@classLabel true 0 1", "@data"
                    ]
                    f.write("\n".join(header) + "\n")
                for i in idx_list:
                    line = f"{trajectories_list[i]}:{labels_list[i]}"
                    f.write(line + "\n")
            print(f"Created {path} with {len(idx_list)} samples")

        write_split("TRAIN", train_idx, write_header=True)
        write_split("TEST", test_idx, write_header=True)

        print("Binary dataset created successfully!")



if __name__ == "__main__":
    
    dataset_root = "UniTS-main\\dataset\\KDDCup15\\KDDCup15_raw"             
    dataset_out = "UniTS-main\\dataset\\KDDCup15\\KDDCup15_processed"      

    processor = KDDCup15BinaryDataset(dataset_root, dataset_out)

    processor.create_binary_dataset(
        test_size=0.2,
        random_state=42
    )
