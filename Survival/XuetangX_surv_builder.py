import pandas as pd
import numpy as np
import os
import zipfile
import sys
import datetime
from tqdm import tqdm
import warnings
import glob
import time
from sklearn.model_selection import train_test_split
import json

warnings.filterwarnings('ignore', category=FutureWarning)

class XuetangXSurvivalDataset:
    def __init__(self, dataset_root, dataset_out, time_bins=45, auto_optimize=True):
        self.dataset_root = dataset_root
        self.dataset_out = dataset_out
        self.time_bins = time_bins
        self.auto_optimize = auto_optimize
        self.type_of_dataset = None
        
        self.courses_start_end_date = None
        self.user_info = None
        self.log_test_data = None
        self.log_train_data = None
        self.truth_test_data = None
        self.truth_train_data = None
        
        self.activity_drop_threshold = 0.1
        self.smoothing_window = 7
        self.lookback_days = 14
        self.max_normal_gap = 21
        self.min_activity_threshold = 1
        self.window_size = 14
        self.min_active_days_in_window = 1
        self.high_sparsity_threshold = 0.8
        self.medium_sparsity_threshold = 0.5

    def load_data(self):
        """Load XuetangX dataset"""
        print("Loading dataset from:", self.dataset_root)
        print("Loading all CSV files from the dataset root...")

        try:
            self.courses_start_end_date = pd.read_csv(os.path.join(self.dataset_root, "course_info.csv"), header=0)
            self.user_info = pd.read_csv(os.path.join(self.dataset_root, "user_info.csv"), header=0)
            print("Course and user info loaded")
        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure the course_info.csv and/or user_info.csv files are present.")
            return
        
        try:
            self.log_train_data = pd.read_csv(os.path.join(self.dataset_root, "prediction_log/train_log.csv"), header=0)
            self.truth_train_data = pd.read_csv(os.path.join(self.dataset_root, "prediction_log/train_truth.csv"), header=0)
            self.log_test_data = pd.read_csv(os.path.join(self.dataset_root, "prediction_log/test_log.csv"), header=0)
            self.truth_test_data = pd.read_csv(os.path.join(self.dataset_root, "prediction_log/test_truth.csv"), header=0)
            print("Log and truth data loaded")
        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure the train and test files are present.")
            return
               
        print("Dataset loaded successfully.")

    def detect_dropout_original(self, timeseries):
        """Original dropout detection method."""
        n_days, n_features = timeseries.shape
        daily_activity = np.sum(timeseries, axis=1)
        
        if len(daily_activity) > self.smoothing_window:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            smoothed = np.convolve(daily_activity, kernel, mode='same')
        else:
            smoothed = daily_activity
            
        for day in range(self.smoothing_window, len(smoothed)):
            lookback_start = max(0, day - self.lookback_days)
            recent_avg = np.mean(smoothed[lookback_start:day])
            current_activity = smoothed[day]
            if recent_avg > 0 and current_activity < recent_avg * self.activity_drop_threshold:
                return day
        return n_days - 1

    def detect_dropout_gap_based(self, timeseries):
        daily_activity = np.sum(timeseries, axis=1)
        active_days = np.where(daily_activity > 0)[0]
        
        if len(active_days) < 3:
            return 0 if len(active_days) == 0 else active_days[-1]
        
        gaps = np.diff(active_days)
        long_gaps = np.where(gaps > self.max_normal_gap)[0]
        if len(long_gaps) > 0:
            dropout_day = active_days[long_gaps[-1]]
            return dropout_day
        return active_days[-1]

    def detect_dropout_window_based(self, timeseries):
        n_days, n_features = timeseries.shape
        daily_activity = np.sum(timeseries, axis=1)
        active_days = daily_activity > self.min_activity_threshold
        
        if np.sum(active_days) < 5:
            return self.find_last_meaningful_activity(timeseries)
        
        for start_day in range(0, n_days - self.window_size, 7):
            window_end = min(start_day + self.window_size, n_days)
            window_active = active_days[start_day:window_end]
            active_count = np.sum(window_active)
            if active_count < self.min_active_days_in_window:
                last_active = np.where(active_days[:start_day])[0]
                if len(last_active) > 0:
                    return last_active[-1]
                else:
                    return start_day
        return n_days - 1

    def find_last_meaningful_activity(self, timeseries):
        daily_activity = np.sum(timeseries, axis=1)
        non_zero_activity = daily_activity[daily_activity > 0]
        if len(non_zero_activity) == 0:
            return 0
        threshold = np.median(non_zero_activity) * 0.1
        for day in range(len(daily_activity) - 1, -1, -1):
            if daily_activity[day] > threshold:
                return day
        return 0

    def detect_dropout_hybrid(self, timeseries):
        n_days, n_features = timeseries.shape
        daily_activity = np.sum(timeseries, axis=1)
        sparsity_ratio = np.sum(daily_activity == 0) / n_days
        if sparsity_ratio > self.high_sparsity_threshold:
            return self.detect_dropout_gap_based(timeseries)
        elif sparsity_ratio > self.medium_sparsity_threshold:
            return self.detect_dropout_window_based(timeseries)
        else:
            return self.detect_dropout_original(timeseries)

    def analyze_data_characteristics(self, all_trajectories):
        """Analyze XuetangX data characteristics for parameter optimization."""
        activity_patterns = []
        gap_distributions = []
        sparsity_levels = []
        
        print("ANALYZING XUETANGX DATA CHARACTERISTICS...")
        
        for enroll_id, data in tqdm(all_trajectories.items(), desc="Analyzing trajectories"):
            timeseries = data["trajectory"].T  # (days, activities)
            daily_activity = np.sum(timeseries, axis=1)
            non_zero_days = daily_activity[daily_activity > 0]
            if len(non_zero_days) > 0:
                activity_patterns.append({
                    'mean_activity': np.mean(non_zero_days),
                    'std_activity': np.std(non_zero_days),
                    'median_activity': np.median(non_zero_days),
                    'activity_range': np.max(non_zero_days) - np.min(non_zero_days)
                })
            active_days = np.where(daily_activity > 0)[0]
            if len(active_days) > 1:
                gaps = np.diff(active_days)
                gap_distributions.extend(gaps)
            sparsity = np.sum(daily_activity == 0) / len(daily_activity)
            sparsity_levels.append(sparsity)
        
        return {
            'activity_patterns': activity_patterns,
            'gaps': gap_distributions,
            'sparsity': sparsity_levels
        }

    def compute_optimal_parameters(self, data_characteristics):
        """Compute optimal parameters for XuetangX dataset."""
        gaps = np.array(data_characteristics['gaps'])
        sparsity = np.array(data_characteristics['sparsity'])
        activity_patterns = data_characteristics['activity_patterns']
        
        valid_gaps = gaps[gaps > 0]
        if len(valid_gaps) == 0:
            valid_gaps = [7]
        
        normal_gap_threshold = np.percentile(valid_gaps, 85)
        max_normal_gap = max(10, min(30, int(normal_gap_threshold)))
        
        all_activities = [p['median_activity'] for p in activity_patterns if p['median_activity'] > 0]
        if all_activities:
            min_activity_threshold = max(1, np.percentile(all_activities, 15))
        else:
            min_activity_threshold = 1.0
        
        avg_gap = np.mean(valid_gaps) if len(valid_gaps) > 0 else 7
        smoothing_window = max(5, min(14, int(avg_gap / 2)))
        lookback_days = max(10, min(25, smoothing_window * 2))
        window_size = max(10, min(25, int(np.percentile(valid_gaps, 70))))
        
        high_sparsity_threshold = min(0.95, np.percentile(sparsity, 85))
        medium_sparsity_threshold = max(0.5, np.percentile(sparsity, 40))
        
        avg_sparsity = np.mean(sparsity)
        if avg_sparsity > 0.8:
            activity_drop_threshold = 0.08
        elif avg_sparsity > 0.5:
            activity_drop_threshold = 0.15
        else:
            activity_drop_threshold = 0.25
        
        optimal_params = {
            'max_normal_gap': max_normal_gap,
            'min_activity_threshold': min_activity_threshold,
            'smoothing_window': smoothing_window,
            'lookback_days': lookback_days,
            'window_size': window_size,
            'high_sparsity_threshold': high_sparsity_threshold,
            'medium_sparsity_threshold': medium_sparsity_threshold,
            'activity_drop_threshold': activity_drop_threshold,
            'min_active_days_in_window': max(2, window_size // 6)
        }
        
        print("\nOPTIMAL PARAMETERS COMPUTED FOR XUETANGX:")
        print("=" * 52)
        for param, value in optimal_params.items():
            print(f"  {param:25}: {value:.3f}")
        return optimal_params

    def auto_optimize_parameters(self, all_trajectories):
        """Auto-optimize parameters for XuetangX."""
        if not self.auto_optimize:
            print("Auto-optimization disabled.")
            return
        print("AUTO-OPTIMIZING PARAMETERS FOR XUETANGX...")
        data_char = self.analyze_data_characteristics(all_trajectories)
        optimal_params = self.compute_optimal_parameters(data_char)
        for param, value in optimal_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        print("Parameters optimized for XuetangX!")

    def find_realistic_dropout_point(self, timeseries):
        """Find realistic dropout point for XuetangX courses."""
        daily_activity = np.sum(timeseries, axis=1)
        n_days = len(daily_activity)
        active_period_end = np.where(daily_activity > 0)[0]
        if len(active_period_end) > 0:
            last_active = active_period_end[-1]
            return min(n_days - 1, max(10, last_active // 2))
        return 10

    def _trajectory_creator_for_each_student_(self, df_log, activities, start_date, end_date):
        """Create trajectories per student (optimized for .ts creation)."""
        student_trajectory = []
        
        if df_log.empty:
            for _ in pd.date_range(start_date, end_date, freq='D'):
                student_trajectory.append([0] * len(activities))
            return np.array(student_trajectory).T
        
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
                daily_activities = [len(day_logs[day_logs['action'] == activity]) for activity in activities]
            student_trajectory.append(daily_activities)

        return np.array(student_trajectory).T

    def _trajectory_creator_for_all_students_(self, is_train_set=True):
        """Create trajectories for all students."""
        if is_train_set:
            df_log = self.log_train_data
            df_labels = self.truth_train_data
            self.type_of_dataset = "train"
        else:
            df_log = self.log_test_data
            df_labels = self.truth_test_data
            self.type_of_dataset = "test"

        all_students_trajectories = {}
        activities = df_log["action"].unique()
        print(f"Found {len(activities)} activities: {activities}")
        
        for idx, row in tqdm(df_labels.iterrows(), desc="Creating trajectories", total=len(df_labels)):
            enroll_id = row["enroll_id"]
            label = row["truth"]
            student_logs = df_log[df_log["enroll_id"] == enroll_id]
            if student_logs.empty:
                print(f"No logs found for enrollment {enroll_id}, skipping...")
                continue
                
            course_id = student_logs.iloc[0]["course_id"]
            course_info = self.courses_start_end_date[self.courses_start_end_date["course_id"] == course_id]
            if course_info.empty:
                print(f"No course info found for course {course_id}, skipping...")
                continue
            
            starting_date = pd.to_datetime(course_info["start"].values[0])
            ending_date = pd.to_datetime(course_info["end"].values[0])
            
            student_trajectory = self._trajectory_creator_for_each_student_(
                student_logs, activities, starting_date, ending_date)
            
            all_students_trajectories[enroll_id] = {
                "trajectory": student_trajectory,
                "label": label,
                "lenght of activities": len(student_trajectory),  # keep original naming
                "n_days": len(student_trajectory[0]),
                "traj_lenght": len(student_trajectory) * len(student_trajectory[0]),
                "course_id": course_id
            }
        return all_students_trajectories

    def create_survival_labels(self, all_trajectories):
        """Create survival analysis labels for XuetangX."""
        survival_times = []
        event_indicators = []
        never_engaged_count = 0
        early_dropout_count = 0
        
        print("Creating survival labels for XuetangX DROPOUT prediction...")
        
        for enroll_id, data in tqdm(all_trajectories.items(), desc="Processing enrollments"):
            timeseries = data["trajectory"].T  # (days, activities)
            binary_label = data["label"]
            n_days = timeseries.shape[0]
            
            if binary_label == 1:  # DROPOUT
                dropout_day = self.detect_dropout_hybrid(timeseries)
                if dropout_day < 10:
                    daily_activity = np.sum(timeseries, axis=1)
                    total_activity = np.sum(daily_activity)
                    activity_after_dropout = np.sum(daily_activity[dropout_day:])
                    if activity_after_dropout > total_activity * 0.3:
                        dropout_day = max(10, self.find_realistic_dropout_point(timeseries))
                    early_dropout_count += 1
                survival_times.append(dropout_day)
                event_indicators.append(1)  # Event observed
            else:  # COMPLETION
                completion_day = n_days - 1
                survival_times.append(completion_day)
                event_indicators.append(0)  # Censored
        
        print(f"Never engaged (filtered): {never_engaged_count}")
        print(f"Early dropouts corrected: {early_dropout_count}")
        return (np.array(survival_times), np.array(event_indicators))

    def discretize_survival_times(self, survival_times, event_indicators):
        """Discretize survival times into bins."""
        max_time = np.max(survival_times)
        bin_edges = np.linspace(0, max_time, self.time_bins + 1)
        discretized_times = np.digitize(survival_times, bin_edges) - 1
        discretized_times = np.clip(discretized_times, 0, self.time_bins - 1)
        return discretized_times, bin_edges

    def validate_survival_times(self, survival_times, event_indicators, all_trajectories):
        """Validate survival analysis results."""
        print("\nENHANCED VALIDATION OF XUETANGX DROPOUT SURVIVAL ANALYSIS:")
        very_early_dropouts = np.sum((survival_times < 10) & (event_indicators == 1))
        early_dropouts = np.sum((survival_times < 20) & (event_indicators == 1))
        realistic_dropouts = np.sum((survival_times >= 20) & (event_indicators == 1))
        never_enrolled = np.sum((survival_times == 0) & (event_indicators == 0))
        
        print(f"Very early dropouts (<10 days): {very_early_dropouts}")
        print(f"Early dropouts (10-20 days): {early_dropouts - very_early_dropouts}")
        print(f"Realistic dropouts (≥20 days): {realistic_dropouts}")
        print(f"Never enrolled (filtered): {never_enrolled}")
        
        total_dropouts = np.sum(event_indicators)
        if total_dropouts > 0:
            print(f"Very early dropout rate: {very_early_dropouts/total_dropouts*100:.1f}% (target: <10%)")
        
        dropout_times = survival_times[event_indicators == 1]
        completion_times = survival_times[event_indicators == 0]
        if len(dropout_times) > 0:
            print(f"Avg DROPOUT time: {np.mean(dropout_times):.1f} days")
        if len(completion_times) > 0:
            print(f"Avg COMPLETION time: {np.mean(completion_times):.1f} days")
        print(f"Logic check (dropout < completion): {np.mean(dropout_times) < np.mean(completion_times) if len(dropout_times) > 0 and len(completion_times) > 0 else 'N/A'}")

    def create_ts_files_for_survival(self, all_traj, survival_times, event_indicators, discretized_times, split_name):
        """Create .ts files for survival analysis."""
        trajectories, survival_labels = [], []
        max_lenght = 0
        n_dims = 0
        
        idx = 0
        for enroll_id, data in all_traj.items():
            traj = data["trajectory"]
            dim_strings = [",".join(map(str, series)) for series in traj]
            trajectories.append(":".join(dim_strings))
            survival_info = f"({survival_times[idx]}:{event_indicators[idx]}:{discretized_times[idx]})"
            survival_labels.append(survival_info)
            n_dims = data["lenght of activities"]
            if max_lenght < data["n_days"]:
                max_lenght = data["n_days"]
            idx += 1
        
        ts_path = os.path.join(self.dataset_out, f"XuetangX_Survival_{split_name.upper()}.ts")

        header = [
            "@problemName XuetangX_dropout_survival",
            "@timeStamps false",
            "@missing false",
            "@univariate false",
            f"@dimensions {n_dims}",
            "@equalLength true",
            f"@seriesLength {max_lenght}",
            "@classLabel false",
            "@survivalAnalysis true",
            f"@timeBins {self.time_bins}",
            "@data"
        ]

        with open(ts_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
            for tr, surv_label in zip(trajectories, survival_labels):
                f.write(f"{tr}:{surv_label}\n")

        print(f"Created {ts_path} with {len(trajectories)} samples")
        return ts_path

    def save_current_params(self):
        """Save current parameter state."""
        return {
            'activity_drop_threshold': self.activity_drop_threshold,
            'smoothing_window': self.smoothing_window,
            'lookback_days': self.lookback_days,
            'max_normal_gap': self.max_normal_gap,
            'min_activity_threshold': self.min_activity_threshold,
            'window_size': self.window_size,
            'min_active_days_in_window': self.min_active_days_in_window,
            'high_sparsity_threshold': self.high_sparsity_threshold,
            'medium_sparsity_threshold': self.medium_sparsity_threshold
        }

    def calculate_optimal_time_bins(self, max_survival_time, course_duration_type="auto"):
        """Calculate optimal number of time bins based on course duration."""
        if course_duration_type == "auto":
            if max_survival_time <= 40:
                course_duration_type = "short"
            elif max_survival_time <= 100:
                course_duration_type = "medium"
            else:
                course_duration_type = "long"
        
        if course_duration_type == "short":
            optimal_bins = max(4, min(8, int(max_survival_time / 7)))
        elif course_duration_type == "medium":
            optimal_bins = max(15, min(30, int(max_survival_time / 3.5)))
        else:  # long
            optimal_bins = max(30, min(60, int(max_survival_time / 7)))
        
        print(f"OPTIMAL TIME BINS CALCULATION:")
        print(f"Max survival time: {max_survival_time} giorni")
        print(f"Course type: {course_duration_type}")
        print(f"Recommended bins: {optimal_bins}")
        print(f"Avg days per bin: {max_survival_time/optimal_bins:.1f}")
        return optimal_bins

    def create_survival_dataset(self, test_size=0.2, random_state=42):
        """Complete pipeline to create the survival dataset."""
        print("CREATING XUETANGX SURVIVAL DATASET...")
        
        self.load_data()
        
        print("\nProcessing training data...")
        train_trajectories = self._trajectory_creator_for_all_students_(is_train_set=True)
        print("\nProcessing test data...")
        test_trajectories = self._trajectory_creator_for_all_students_(is_train_set=False)
        
        all_trajectories = {**train_trajectories, **test_trajectories}
        if len(all_trajectories) == 0:
            print("No trajectories created. Please check data format and paths.")
            return

        if self.auto_optimize:
            self.auto_optimize_parameters(all_trajectories)
        
        survival_times, event_indicators = self.create_survival_labels(all_trajectories)
        
        self.validate_survival_times(survival_times, event_indicators, all_trajectories)
        
        max_time = np.max(survival_times)
        optimal_bins = self.calculate_optimal_time_bins(max_time, "auto")
        if self.auto_optimize:
            self.time_bins = optimal_bins
            print(f"Time bins auto-updated: {self.time_bins}")
        discretized_times, bin_edges = self.discretize_survival_times(survival_times, event_indicators)
        
        # Stats
        print("\nXUETANGX DROPOUT SURVIVAL ANALYSIS STATISTICS:")
        print(f"Total enrollments: {len(survival_times)}")
        print(f"DROPOUT events observed: {np.sum(event_indicators)} ({np.sum(event_indicators)/len(event_indicators)*100:.1f}%)")
        print(f"COMPLETIONS (censored): {np.sum(1-event_indicators)} ({np.sum(1-event_indicators)/len(event_indicators)*100:.1f}%)")
        if np.sum(event_indicators) > 0:
            print(f"Average time to dropout: {np.mean(survival_times[event_indicators==1]):.1f} days")
        if np.sum(1-event_indicators) > 0:
            print(f"Average completion time: {np.mean(survival_times[event_indicators==0]):.1f} days")
        print(f"Time bins for hazard prediction: {self.time_bins}")

        enroll_ids = list(all_trajectories.keys())
        stratify = None if (np.sum(event_indicators) == 0 or np.sum(1-event_indicators) == 0) else event_indicators
        indices = np.arange(len(enroll_ids))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=stratify)
        
        train_enroll_ids = [enroll_ids[i] for i in train_idx]
        test_enroll_ids = [enroll_ids[i] for i in test_idx]
        train_dict = {eid: all_trajectories[eid] for eid in train_enroll_ids}
        test_dict = {eid: all_trajectories[eid] for eid in test_enroll_ids}
        
        train_survival = survival_times[train_idx]
        test_survival = survival_times[test_idx]
        train_events = event_indicators[train_idx]
        test_events = event_indicators[test_idx]
        train_discrete = discretized_times[train_idx]
        test_discrete = discretized_times[test_idx]
        
        print(f"Train: {len(train_dict)} samples, Events: {np.sum(train_events)}/{len(train_events)}")
        print(f"Test: {len(test_dict)} samples, Events: {np.sum(test_events)}/{len(test_events)}")
        
        # Write .ts
        self.create_ts_files_for_survival(train_dict, train_survival, train_events, train_discrete, "TRAIN")
        self.create_ts_files_for_survival(test_dict, test_survival, test_events, test_discrete, "TEST")
        
        # Metadata
        metadata = {
            'time_bins': self.time_bins,
            'bin_edges': bin_edges.tolist(),
            'max_time': float(np.max(survival_times)),
            'n_dimensions': list(all_trajectories.values())[0]["lenght of activities"],
            'max_length': max(data["n_days"] for data in all_trajectories.values()),
            'optimized_params': self.save_current_params()
        }
        with open(os.path.join(self.dataset_out, "xuetangx_survival_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("XuetangX Survival dataset created successfully!")
        return survival_times, event_indicators, bin_edges


if __name__ == "__main__":
    Original_Dataset_Unprocessed_ROOT = r"C:\\Users\\HF\\Documents\\Python Script\\UniTS-main\\dataset\\XuetangX\\XuetangX_raw"
    Dataset_Processed_OUT = r"C:\\Users\\HF\\Documents\\Python Script\\UniTS-main\\dataset\\XuetangX\\XuetangX_survival"
    
    if not os.path.exists(Dataset_Processed_OUT):
        os.makedirs(Dataset_Processed_OUT)

    survival_processor = XuetangXSurvivalDataset(
        dataset_root=Original_Dataset_Unprocessed_ROOT, 
        dataset_out=Dataset_Processed_OUT,
        time_bins=40,
        auto_optimize=True
    )
    
    survival_times, event_indicators, bin_edges = survival_processor.create_survival_dataset(
        test_size=0.2,
        random_state=42
    )

