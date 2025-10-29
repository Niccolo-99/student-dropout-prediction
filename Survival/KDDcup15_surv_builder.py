import pandas as pd
import numpy as np
import os
import zipfile
import sys
import datetime
from tqdm import tqdm
import warnings
import time
from sklearn.model_selection import train_test_split
import json

warnings.filterwarnings('ignore', category=FutureWarning)

class KDDCup15SurvivalDataset:
    def __init__(self, dataset_root, dataset_out, time_bins=50, auto_optimize=True):
        self.dataset_root = dataset_root
        self.dataset_out = dataset_out
        self.time_bins = time_bins
        self.auto_optimize = auto_optimize
        
        self.activity_drop_threshold = 0.1
        self.smoothing_window = 7
        self.lookback_days = 14
        self.max_normal_gap = 21
        self.min_activity_threshold = 1
        self.window_size = 14
        self.min_active_days_in_window = 1
        self.high_sparsity_threshold = 0.8
        self.medium_sparsity_threshold = 0.5

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
        print("Loading full dataset...")

        try:
            self.courses_start_end_date = pd.read_csv(os.path.join(self.dataset_root, "date.csv"))
            self.courses_info = pd.read_csv(os.path.join(self.dataset_root, "object.csv"))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
        try:
            self._extract_zip_file_if_needed_(os.path.join(self.dataset_root, "train.zip"))
            self._extract_zip_file_if_needed_(os.path.join(self.dataset_root, "test.zip"))

            self.enrollment_train_data = pd.read_csv(os.path.join(self.dataset_root, "train\\enrollment_train.csv"))
            self.log_train_data = pd.read_csv(os.path.join(self.dataset_root, "train\\log_train.csv"))
            self.truth_train_data = pd.read_csv(os.path.join(self.dataset_root, "train\\truth_train.csv"), header=None)
            self.enrollment_test_data = pd.read_csv(os.path.join(self.dataset_root, "test\\enrollment_test.csv"))
            self.log_test_data = pd.read_csv(os.path.join(self.dataset_root, "test\\log_test.csv"))
            self.truth_test_data = pd.read_csv(os.path.join(self.dataset_root, "test\\truth_test.csv"), header=None)
        except FileNotFoundError as e:
            print(f"Error: {e}")
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
        n_days, n_features = timeseries.shape
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
        """Analyze data to determine optimal parameters."""
        activity_patterns = []
        gap_distributions = []
        sparsity_levels = []
        
        print("ANALYZING KDDCup DATA CHARACTERISTICS...")
        
        for enrollment_id, data in tqdm(all_trajectories.items(), desc="Analyzing trajectories"):
            timeseries = data["trajectory"].T  
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
        """Compute optimal parameters with KDDCup-specific constraints."""
        gaps = np.array(data_characteristics['gaps'])
        sparsity = np.array(data_characteristics['sparsity'])
        activity_patterns = data_characteristics['activity_patterns']
        
        valid_gaps = gaps[gaps > 0]
        if len(valid_gaps) == 0:
            valid_gaps = [7]
        
        normal_gap_threshold = np.percentile(valid_gaps, 85)
        max_normal_gap = max(7, min(21, int(normal_gap_threshold))) 
        
        all_activities = [p['median_activity'] for p in activity_patterns if p['median_activity'] > 0]
        if all_activities:
            min_activity_threshold = max(0.5, np.percentile(all_activities, 15))
        else:
            min_activity_threshold = 1.0
        
        avg_gap = np.mean(valid_gaps) if len(valid_gaps) > 0 else 7
        smoothing_window = max(3, min(10, int(avg_gap / 2)))
        lookback_days = max(7, min(21, smoothing_window * 2))
        window_size = max(7, min(21, int(np.percentile(valid_gaps, 70))))
        
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
        
        print("\nOPTIMAL PARAMETERS FOR KDDCup:")
        for param, value in optimal_params.items():
            print(f"  {param:25}: {value:.3f}")
        
        return optimal_params

    def auto_optimize_parameters(self, all_trajectories):
        """Auto-optimize parameters based on KDDCup data."""
        if not self.auto_optimize:
            print("  Auto-optimization disabled.")
            return
        
        print("AUTO-OPTIMIZING PARAMETERS FOR KDDCup...")
        
        data_char = self.analyze_data_characteristics(all_trajectories)
        optimal_params = self.compute_optimal_parameters(data_char)
        
        for param, value in optimal_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        
        print(" Parameters optimized for KDDCup!")

    def find_realistic_dropout_point(self, timeseries):
        """Find realistic dropout point for KDDCup courses."""
        daily_activity = np.sum(timeseries, axis=1)
        n_days = len(daily_activity)
        
        active_period_end = np.where(daily_activity > 0)[0]
        if len(active_period_end) > 0:
            last_active = active_period_end[-1]
            return min(n_days - 1, max(7, last_active // 2))  
        
        return 7

    def _trajectory_creator_for_each_student_(self, df_log, activities, start_date, end_date):        
        """Create trajectories for KDDCup students."""
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
        """Create trajectories for all students."""
        if is_train_set:
            df_log = self.log_train_data
            df_enrollment = self.enrollment_train_data
            df_labels = self.truth_train_data
            self.type_of_dataset = "train"
        else:
            df_log = self.log_test_data
            df_enrollment = self.enrollment_test_data
            df_labels = self.truth_test_data
            self.type_of_dataset = "test"

        all_students_trajectories = {}
        activities = df_log["event"].unique()
        
        for idx, enrollment_row in tqdm(df_enrollment.iterrows(), 
                                        desc="Creating trajectories", 
                                        total=len(df_enrollment)):
            enrollment_id = enrollment_row["enrollment_id"]
            course_id = enrollment_row["course_id"]
            
            course_dates = self.courses_start_end_date[self.courses_start_end_date["course_id"] == course_id]
            if len(course_dates) == 0:
                continue
                
            starting_date = pd.to_datetime(course_dates["from"].values[0])
            ending_date = pd.to_datetime(course_dates["to"].values[0])
            
            label_row = df_labels[df_labels.iloc[:,0] == enrollment_id]
            if len(label_row) == 0:
                continue
            label = label_row.iloc[0, 1]
            
            student_trajectory = self._trajectory_creator_for_each_student_(
                df_log[df_log["enrollment_id"] == enrollment_id], 
                activities, starting_date, ending_date)
            
            all_students_trajectories[enrollment_id] = {
                "trajectory": student_trajectory,
                "label": label,
                "length_of_activities": len(student_trajectory),
                "n_days": len(student_trajectory[0]),
                "course_id": course_id
            }
        
        return all_students_trajectories

    def create_survival_labels(self, all_trajectories):
        """Create survival analysis labels for KDDCup."""
        survival_times = []
        event_indicators = []
      
        never_engaged_count = 0
        early_dropout_count = 0
        
        print("Creating survival labels for KDDCup DROPOUT prediction...")
        
        for enrollment_id, data in tqdm(all_trajectories.items(), desc="Processing enrollments"):
            timeseries = data["trajectory"].T  # Transpose to (days, features)
            binary_label = data["label"]
            n_days = timeseries.shape[0]
            
            if binary_label == 1:  # DROPOUT
                dropout_day = self.detect_dropout_hybrid(timeseries)
                
                if dropout_day < 7:
                    daily_activity = np.sum(timeseries, axis=1)
                    total_activity = np.sum(daily_activity)
                    activity_after_dropout = np.sum(daily_activity[dropout_day:])
                    
                    if activity_after_dropout > total_activity * 0.3:
                        dropout_day = max(7, self.find_realistic_dropout_point(timeseries))
                    
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

    def analyze_activity_sparsity(self, all_trajectories):
        """Analyze sparsity of KDDCup data."""
        all_sparsity_stats = []
        
        for enrollment_id, data in all_trajectories.items():
            timeseries = data["trajectory"].T
            daily_activity = np.sum(timeseries, axis=1)
            
            total_days = len(daily_activity)
            zero_days = np.sum(daily_activity == 0)
            sparsity_ratio = zero_days / total_days
            
            all_sparsity_stats.append({
                'enrollment_id': enrollment_id,
                'dropout': data['label'],
                'sparsity_ratio': sparsity_ratio,
                'mean_activity': np.mean(daily_activity),
                'total_days': total_days
            })
        
        sparsity_df = pd.DataFrame(all_sparsity_stats)
        
        print("KDDCup SPARSITY ANALYSIS:")
        print(f"Average sparsity ratio: {sparsity_df['sparsity_ratio'].mean():.3f}")
        print(f"Enrollments with >80% zero days: {(sparsity_df['sparsity_ratio'] > 0.8).sum()}")
        
        dropout_sparsity = sparsity_df[sparsity_df['dropout'] == 1]['sparsity_ratio']
        completion_sparsity = sparsity_df[sparsity_df['dropout'] == 0]['sparsity_ratio']
        
        print(f"SPARSITY BY OUTCOME:")
        print(f"Dropout students - avg sparsity: {dropout_sparsity.mean():.3f}")
        print(f"Completion students - avg sparsity: {completion_sparsity.mean():.3f}")
        
        return sparsity_df

    def discretize_survival_times(self, survival_times, event_indicators):
        """Discretize survival times into bins."""
        max_time = np.max(survival_times)
        bin_edges = np.linspace(0, max_time, self.time_bins + 1)
        discretized_times = np.digitize(survival_times, bin_edges) - 1
        discretized_times = np.clip(discretized_times, 0, self.time_bins - 1)
        return discretized_times, bin_edges

    def validate_survival_times(self, survival_times, event_indicators, all_trajectories):
        """Validate survival analysis results."""
        print("\nVALIDATION OF KDDCup DROPOUT SURVIVAL ANALYSIS:")
        
        very_early_dropouts = np.sum((survival_times < 7) & (event_indicators == 1))
        early_dropouts = np.sum((survival_times < 14) & (event_indicators == 1))
        realistic_dropouts = np.sum((survival_times >= 14) & (event_indicators == 1))
        never_enrolled = np.sum((survival_times == 0) & (event_indicators == 0))
        
        print(f"Very early dropouts (<7 days): {very_early_dropouts} ")
        print(f"Early dropouts (7-14 days): {early_dropouts - very_early_dropouts}")
        print(f"Realistic dropouts (≥14 days): {realistic_dropouts}")
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

    def write_survival_split(self, trajectories_list, survival_times, event_indicators, discretized_times, split_name, n_dims, max_length, write_header=False):
        """Write survival format .ts files."""
        os.makedirs(self.dataset_out, exist_ok=True)
        path = os.path.join(self.dataset_out, f"KDDCup_Survival_{split_name}.ts")
        mode = "w" if write_header else "a"
        
        with open(path, mode, encoding="utf-8") as f:
            if write_header:
                header = [
                    "@problemName KDDCup_dropout_survival",
                    "@timeStamps false", "@missing false", "@univariate false",
                    f"@dimensions {n_dims}", "@equalLength true", f"@seriesLength {max_length}",
                    "@classLabel false", "@survivalAnalysis true", f"@timeBins {self.time_bins}",
                    "@data"
                ]
                f.write("\n".join(header) + "\n")

            for i, traj_str in enumerate(trajectories_list):
                survival_info = f"{survival_times[i]}:{event_indicators[i]}:{discretized_times[i]}"
                line = f"{traj_str}:({survival_info})"
                f.write(line + "\n")
        
        print(f"Created {path} with {len(trajectories_list)} samples")

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

    def create_survival_dataset(self, test_size=0.2, random_state=42):
        """Main method to create survival dataset."""
        print("CREATING KDDCup SURVIVAL DATASET...")
        
        self.load_data()
        
        print("\nProcessing training data...")
        train_trajectories = self._trajectory_creator_for_all_students_(is_train_set=True)
        
        print("\nProcessing test data...")
        test_trajectories = self._trajectory_creator_for_all_students_(is_train_set=False)
        all_trajectories = {**train_trajectories, **test_trajectories}
        sparsity_df = self.analyze_activity_sparsity(all_trajectories)
        
        if self.auto_optimize:
            self.auto_optimize_parameters(all_trajectories)
        
        survival_times, event_indicators = self.create_survival_labels(all_trajectories)
        
        self.validate_survival_times(survival_times, event_indicators, all_trajectories)
        
        discretized_times, bin_edges = self.discretize_survival_times(survival_times, event_indicators)
        
        print("\nKDDCup DROPOUT SURVIVAL ANALYSIS STATISTICS:")
        print(f"Total enrollments: {len(survival_times)}")
        print(f"DROPOUT events observed: {np.sum(event_indicators)} ({np.sum(event_indicators)/len(event_indicators)*100:.1f}%)")
        print(f"COMPLETIONS (censored): {np.sum(1-event_indicators)} ({np.sum(1-event_indicators)/len(event_indicators)*100:.1f}%)")
        if np.sum(event_indicators) > 0:
            print(f"  Average time to dropout: {np.mean(survival_times[event_indicators==1]):.1f} days")
        if np.sum(1-event_indicators) > 0:
            print(f"  Average completion time: {np.mean(survival_times[event_indicators==0]):.1f} days")
        print(f"  Time bins for hazard prediction: {self.time_bins}")
        
        trajectories_list = []
        labels_list = []
        enrollment_ids = []
        
        for enrollment_id, data in all_trajectories.items():
            traj = data["trajectory"]
            dim_strings = [",".join(map(str, series)) for series in traj]
            trajectories_list.append(":".join(dim_strings))
            labels_list.append(data["label"])
            enrollment_ids.append(enrollment_id)
        
        first_traj = list(all_trajectories.values())[0]["trajectory"]
        n_dims = len(first_traj)
        max_length = max(data["n_days"] for data in all_trajectories.values())
        
        if np.sum(event_indicators) == 0 or np.sum(1-event_indicators) == 0:
            stratify = None
        else:
            stratify = event_indicators
        
        indices = np.arange(len(trajectories_list))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, 
            random_state=random_state, stratify=stratify
        )
        
        # Split all arrays
        train_trajectories_list = [trajectories_list[i] for i in train_idx]
        test_trajectories_list = [trajectories_list[i] for i in test_idx]
        
        train_survival = survival_times[train_idx]
        test_survival = survival_times[test_idx]
        train_events = event_indicators[train_idx]
        test_events = event_indicators[test_idx]
        train_discrete = discretized_times[train_idx]
        test_discrete = discretized_times[test_idx]
        
        print(f"Train: {len(train_trajectories_list)} samples, Events: {np.sum(train_events)}/{len(train_events)}")
        print(f"Test: {len(test_trajectories_list)} samples, Events: {np.sum(test_events)}/{len(test_events)}")
        
        self.write_survival_split(train_trajectories_list, train_survival, train_events, train_discrete, "TRAIN", n_dims, max_length, True)
        self.write_survival_split(test_trajectories_list, test_survival, test_events, test_discrete, "TEST", n_dims, max_length, True)
        
        metadata = {
            'time_bins': self.time_bins,
            'bin_edges': bin_edges.tolist(),
            'max_time': float(np.max(survival_times)),
            'n_dimensions': n_dims,
            'max_length': max_length,
            'optimized_params': self.save_current_params()
        }
        
        with open(os.path.join(self.dataset_out, "kddcup_survival_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("KDDCup Survival dataset created successfully!")
        return survival_times, event_indicators, bin_edges

if __name__ == "__main__":
    Original_Dataset_Unprocessed_ROOT = r"C:\\Users\\HF\Documents\\Python Script\\UniTS-main\\dataset\\KDDcup\\kddcup15"
    Dataset_Processed_OUT = r"C:\\Users\\HF\\Documents\\Python Script\\UniTS-main\dataset\\KDDcup\\kddcup15_survival"
    
    if not os.path.exists(Dataset_Processed_OUT):
        os.makedirs(Dataset_Processed_OUT)

    survival_processor = KDDCup15SurvivalDataset(
        dataset_root=Original_Dataset_Unprocessed_ROOT, 
        dataset_out=Dataset_Processed_OUT,
        time_bins=4, 
        auto_optimize=True
    )
    
    
    survival_times, event_indicators, bin_edges = survival_processor.create_survival_dataset(
        test_size=0.2,
        random_state=42
    )

