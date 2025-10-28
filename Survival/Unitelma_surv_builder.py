import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import ast
from tqdm.auto import tqdm
import json

class UnitelmaDatasetProcessor:
    
    def __init__(self, origin_path, end_path, dropout_threshold, time_bins=50, 
                 sparse_mode=True, auto_optimize=True, **kwargs):
        self.origin = origin_path
        self.end = end_path
        self.dropout_threshold = dropout_threshold  
        self.time_bins = time_bins
        self.sparse_mode = sparse_mode
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

    def load_dataset(self):
        dataset_dictionary = {}
        files = glob.glob(f"{self.origin}/*/*.csv")
        
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.origin}")
            
        for file in files:
            try:
                dataset_dictionary[file] = pd.read_csv(file, header=0)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
        return dataset_dictionary

    def analyze_activity_sparsity(self, df_dictionary):
        all_sparsity_stats = []
        daily_activities = []
        
        for key in df_dictionary.keys():
            df = df_dictionary[key]
            
            for _, row in df.iterrows():
                timeseries = np.array(ast.literal_eval(row['timeseries']))
                daily_activity = np.sum(timeseries, axis=1)
                
                total_days = len(daily_activity)
                zero_days = np.sum(daily_activity == 0)
                sparsity_ratio = zero_days / total_days
                
                daily_activities.extend(daily_activity)
                all_sparsity_stats.append({
                    'student_id': row['student_id'],
                    'dropout': row['dropout'],
                    'sparsity_ratio': sparsity_ratio,
                    'mean_activity': np.mean(daily_activity),
                    'total_days': total_days
                })
        
        sparsity_df = pd.DataFrame(all_sparsity_stats)
        
        print("SPARSITY ANALYSIS:")
        print(f"Average sparsity ratio: {sparsity_df['sparsity_ratio'].mean():.3f}")
        print(f"Students with >80% zero days: {(sparsity_df['sparsity_ratio'] > 0.8).sum()}")
        
        dropout_sparsity = sparsity_df[sparsity_df['dropout'] >= self.dropout_threshold]['sparsity_ratio']
        completion_sparsity = sparsity_df[sparsity_df['dropout'] < self.dropout_threshold]['sparsity_ratio']
        
        print(f"SPARSITY BY OUTCOME:")
        print(f"Dropout students - avg sparsity: {dropout_sparsity.mean():.3f}")
        print(f"Completion students - avg sparsity: {completion_sparsity.mean():.3f}")
        
        return sparsity_df

    def analyze_original_labels(self, df_dictionary):
        all_labels = []
        label_1_activities = []
        label_0_activities = []
        
        for key in df_dictionary.keys():
            df = df_dictionary[key]
            
            for _, row in df.iterrows():
                all_labels.append(row['dropout'])
                
                timeseries = np.array(ast.literal_eval(row['timeseries']))
                daily_activity = np.sum(timeseries, axis=1)
                total_activity = np.sum(daily_activity)
                
                if row['dropout'] >= self.dropout_threshold:
                    label_1_activities.append(total_activity)
                else:
                    label_0_activities.append(total_activity)
        
        all_labels = np.array(all_labels)
        
        print("Original Label Analysis:")
        print(f"Label = 1: {np.sum(all_labels >= self.dropout_threshold)} ({np.sum(all_labels >= self.dropout_threshold)/len(all_labels)*100:.1f}%)")
        print(f"Label = 0: {np.sum(all_labels < self.dropout_threshold)} ({np.sum(all_labels < self.dropout_threshold)/len(all_labels)*100:.1f}%)")
        print(f"Label=1 avg activity: {np.mean(label_1_activities):.2f}")
        print(f"Label=0 avg activity: {np.mean(label_0_activities):.2f}")

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

    def test_dropout_detection_methods(self, df_dictionary, sample_size=10):
        print("\nTESTING DROPOUT DETECTION METHODS:")
        
        sample_count = 0
        method_results = {
            'original': [], 'gap_based': [], 'window_based': [], 'hybrid': []
        }
        
        for key in df_dictionary.keys():
            df = df_dictionary[key]
            
            for _, row in df.iterrows():
                if sample_count >= sample_size:
                    break
                timeseries = np.array(ast.literal_eval(row['timeseries']))
                student_id = row['student_id']
                dropout_label = row['dropout']
                daily_activity = np.sum(timeseries, axis=1)
                sparsity = np.sum(daily_activity == 0) / len(daily_activity)
                
                print(f"\nStudent {student_id} (dropout: {dropout_label:.3f}, sparsity: {sparsity:.3f}):")
                
                results = {
                    'original': self.detect_dropout_original(timeseries),
                    'gap_based': self.detect_dropout_gap_based(timeseries),
                    'window_based': self.detect_dropout_window_based(timeseries),
                    'hybrid': self.detect_dropout_hybrid(timeseries)
                }
                
                for method, day in results.items():
                    method_results[method].append(day)
                    print(f"{method:12}: Day {day:3d}")
                
                sample_count += 1
            
            if sample_count >= sample_size:
                break
        
        print(f"\nSUMMARY ACROSS {sample_size} STUDENTS:")
        for method, days in method_results.items():
            if days:
                avg_day = np.mean(days)
                std_day = np.std(days)
                print(f"{method:12}: {avg_day:6.1f} ± {std_day:5.1f} days")

    def analyze_data_characteristics(self, df_dictionary):
        """Analyze data to determine optimal parameters."""
        all_timeseries = []
        all_labels = []
        activity_patterns = []
        gap_distributions = []
        sparsity_levels = []
        
        print("ANALYZING DATA CHARACTERISTICS...")
        
        for key in df_dictionary.keys():
            df = df_dictionary[key]
            for _, row in df.iterrows():
                timeseries = np.array(ast.literal_eval(row['timeseries']))
                daily_activity = np.sum(timeseries, axis=1)
                
                all_timeseries.append(timeseries)
                all_labels.append(row['dropout'])
                
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
            'timeseries': all_timeseries,
            'labels': all_labels,
            'activity_patterns': activity_patterns,
            'gaps': gap_distributions,
            'sparsity': sparsity_levels
        }

    def compute_optimal_parameters(self, data_characteristics):
        """Compute optimal parameters with realistic constraints."""
        gaps = np.array(data_characteristics['gaps'])
        sparsity = np.array(data_characteristics['sparsity'])
        activity_patterns = data_characteristics['activity_patterns']
        
        valid_gaps = gaps[gaps > 0]
        if len(valid_gaps) == 0:
            valid_gaps = [7]
        
        normal_gap_threshold = np.percentile(valid_gaps, 85) 
        max_normal_gap = max(14, min(35, int(normal_gap_threshold)))
        
        all_activities = [p['median_activity'] for p in activity_patterns if p['median_activity'] > 0]
        if all_activities:
            min_activity_threshold = max(1, np.percentile(all_activities, 15)) 
        else:
            min_activity_threshold = 1.0
        
        avg_gap = np.mean(valid_gaps) if len(valid_gaps) > 0 else 7
        smoothing_window = max(5, min(14, int(avg_gap / 2))) 
        lookback_days = max(14, min(28, smoothing_window * 3))  
        window_size = max(14, min(28, int(np.percentile(valid_gaps, 70))))
        
        # Sparsity thresholds
        high_sparsity_threshold = min(0.95, np.percentile(sparsity, 85))  
        medium_sparsity_threshold = max(0.6, np.percentile(sparsity, 40))  
        
        # Activity drop threshold
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
            'min_active_days_in_window': max(3, window_size // 6)  
        }
        
        print("\nOPTIMAL PARAMETERS COMPUTED (WITH CONSTRAINTS):")
        for param, value in optimal_params.items():
            print(f"  {param:25}: {value:.3f}")
        
        return optimal_params

    def auto_optimize_parameters(self, df_dictionary):
        """Automatically optimize all parameters based on data."""
        if not self.auto_optimize:
            print("Auto-optimization disabled. Using default parameters.")
            return
        
        print("AUTO-OPTIMIZING PARAMETERS...")
        data_char = self.analyze_data_characteristics(df_dictionary)
        optimal_params = self.compute_optimal_parameters(data_char)
        
        for param, value in optimal_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        
        print("Parameters optimized!")

    def create_survival_labels(self, timeseries_matrix, binary_labels):
        n_students = len(timeseries_matrix)
        survival_times = []
        event_indicators = []
        
        never_engaged_count = 0
        early_dropout_count = 0
        
        print("Creating survival labels for DROPOUT prediction...")
        
        for i in tqdm(range(n_students)):
            timeseries = timeseries_matrix[i]
            binary_label = binary_labels[i]
            n_days = timeseries.shape[0]
            
            if binary_label == 1:  # DROPOUT
                dropout_day = self.detect_dropout_hybrid(timeseries) if self.sparse_mode else self.detect_dropout_original(timeseries)
                
                if dropout_day < 14:
                    daily_activity = np.sum(timeseries, axis=1)
                    total_activity = np.sum(daily_activity)
                    activity_after_dropout = np.sum(daily_activity[dropout_day:])
                    
                    if activity_after_dropout > total_activity * 0.3:
                        dropout_day = max(14, self.find_realistic_dropout_point(timeseries))
                    
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

    def find_realistic_dropout_point(self, timeseries):
        """Find more realistic dropout point for edge cases."""
        daily_activity = np.sum(timeseries, axis=1)
        n_days = len(daily_activity)
        
        active_period_end = np.where(daily_activity > 0)[0]
        if len(active_period_end) > 0:
            last_active = active_period_end[-1]
            return min(n_days - 1, max(14, last_active // 2))
        
        return 14 

    def validate_survival_times(self, survival_times, event_indicators, timeseries_matrix):
        print("\nENHANCED VALIDATION OF DROPOUT SURVIVAL ANALYSIS:")
        
        very_early_dropouts = np.sum((survival_times < 7) & (event_indicators == 1))
        early_dropouts = np.sum((survival_times < 14) & (event_indicators == 1))
        realistic_dropouts = np.sum((survival_times >= 14) & (event_indicators == 1))
        never_enrolled = np.sum((survival_times == 0) & (event_indicators == 0))
        
        print(f"Very early dropouts (<7 days): {very_early_dropouts} ⚠️")
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
        
       
        dropout_sparsity = []
        completion_sparsity = []
        
        for i, timeseries in enumerate(timeseries_matrix):
            daily_activity = np.sum(timeseries, axis=1)
            sparsity = np.sum(daily_activity == 0) / len(daily_activity)
            
            if event_indicators[i] == 1:
                dropout_sparsity.append(sparsity)
            else:
                completion_sparsity.append(sparsity)
        
        if len(dropout_sparsity) > 0:
            print(f"DROPOUT students sparsity: {np.mean(dropout_sparsity):.3f}")
        if len(completion_sparsity) > 0:
            print(f"COMPLETION students sparsity: {np.mean(completion_sparsity):.3f}")
        
        print(f"Logic check (dropout < completion): {np.mean(dropout_times) < np.mean(completion_times) if len(dropout_times) > 0 and len(completion_times) > 0 else 'N/A'}")

    def discretize_survival_times(self, survival_times, event_indicators):
        max_time = np.max(survival_times)
        bin_edges = np.linspace(0, max_time, self.time_bins + 1)
        discretized_times = np.digitize(survival_times, bin_edges) - 1
        discretized_times = np.clip(discretized_times, 0, self.time_bins - 1)
        return discretized_times, bin_edges

    def process_dataset(self, df_dictionary):
        """Main processing with auto-optimization."""
        if self.auto_optimize:
            self.auto_optimize_parameters(df_dictionary)
        
        all_students = {}
        matrix_of_all_students = []
        all_labels = []
        student_ids = []
        
        print("PROCESSING DATASET...")
        
        for key in df_dictionary.keys():
            df = df_dictionary[key]
            for _, row in tqdm(df.iterrows(), total=len(df)):
                matrix_str = row['timeseries']
                label = 1 if row['dropout'] > self.dropout_threshold else 0
                stud_id = row['student_id']

                arr = np.asarray(ast.literal_eval(matrix_str), dtype=float)
                all_labels.append(label)
                student_ids.append(stud_id)
                matrix_of_all_students.append(arr)
                
                all_students[stud_id] = {
                    "matrix": arr, "label": label,
                    "n_days": arr.shape[0], "n_feat": arr.shape[1]
                }
                
        matrix_array = np.array(matrix_of_all_students)
        labels_array = np.array(all_labels)
        
        survival_times, event_indicators = self.create_survival_labels(matrix_array, labels_array)
        self.validate_survival_times(survival_times, event_indicators, matrix_array)
        
        discretized_times, bin_edges = self.discretize_survival_times(survival_times, event_indicators)
        
        print("\nDROPOUT SURVIVAL ANALYSIS STATISTICS:")
        print(f"Total students: {len(survival_times)}")
        print(f"DROPOUT events observed: {np.sum(event_indicators)} ({np.sum(event_indicators)/len(event_indicators)*100:.1f}%)")
        print(f"COMPLETIONS (censored): {np.sum(1-event_indicators)} ({np.sum(1-event_indicators)/len(event_indicators)*100:.1f}%)")
        if np.sum(event_indicators) > 0:
            print(f"Average time to dropout: {np.mean(survival_times[event_indicators==1]):.1f} days")
        if np.sum(1-event_indicators) > 0:
            print(f"Average completion time: {np.mean(survival_times[event_indicators==0]):.1f} days")
        print(f"Time bins for hazard prediction: {self.time_bins}")
        
        return (all_students, matrix_array, labels_array, survival_times, event_indicators, discretized_times, bin_edges)

    def write_survival_split(self, X, survival_times, event_indicators, discretized_times, split_name, write_header=False):
        os.makedirs(self.end, exist_ok=True)
        path = os.path.join(self.end, f"Unitelma_Survival_{split_name}.ts")
        mode = "w" if write_header else "a"
        
        with open(path, mode, encoding="utf-8") as f:
            if write_header:
                header = [
                    "@problemName Unitelma_dropout_survival",
                    "@timeStamps false", "@missing false", "@univariate false",
                    f"@dimensions {X.shape[2]}", "@equalLength true", f"@seriesLength {X.shape[1]}",
                    "@classLabel false", "@survivalAnalysis true", f"@timeBins {self.time_bins}",
                    "@data"
                ]
                f.write("\n".join(header) + "\n")

            X_formatted = X.swapaxes(1, 2)
            for i, sample in enumerate(X_formatted):
                dims = []
                for dim in sample:
                    dims.append(",".join(map(str, dim)))
                timeseries_str = ":".join(dims)
                
                survival_info = f"{survival_times[i]}:{event_indicators[i]}:{discretized_times[i]}"
                line = f"{timeseries_str}:({survival_info})"
                f.write(line + "\n")

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

def main():
    config = {
        'origin_path': "dataset/Dropout/Dropout_dataset_raw",
        'end_path': "dataset/Dropout/unitelma_dropout", 
        'dropout_threshold': 0.5,
        'time_bins': 50,
        'test_size': 0.2,
        'random_state': 42,
        'sparse_mode': True,
        'auto_optimize': True
    }
    
    processor = UnitelmaDatasetProcessor(**config)
    df_dictionary = processor.load_dataset()
    
    processor.analyze_activity_sparsity(df_dictionary)
    processor.analyze_original_labels(df_dictionary)
    processor.test_dropout_detection_methods(df_dictionary, sample_size=15)
    
    (student_info_dictionary, matrix, labels, survival_times, event_indicators, 
      discretized_times, bin_edges) = processor.process_dataset(df_dictionary)

    # Stratified split (if both events and censored exist)
    if np.sum(event_indicators) == 0 or np.sum(1-event_indicators) == 0:
        print("Warning: No events or no censored observations. Using random split.")
        stratify = None
    else:
        stratify = event_indicators

    indices = np.arange(len(matrix))
    train_idx, test_idx = train_test_split(
        indices, test_size=config['test_size'], 
        random_state=config['random_state'], stratify=stratify
    )
    
    X_train, X_test = matrix[train_idx], matrix[test_idx]
    survival_train, survival_test = survival_times[train_idx], survival_times[test_idx]
    event_train, event_test = event_indicators[train_idx], event_indicators[test_idx]
    discrete_train, discrete_test = discretized_times[train_idx], discretized_times[test_idx]
    
    print(f"Train: {X_train.shape}, Events: {np.sum(event_train)}/{len(event_train)}")
    print(f"Test: {X_test.shape}, Events: {np.sum(event_test)}/{len(event_test)}")
    
    processor.write_survival_split(X_train, survival_train, event_train, discrete_train, "TRAIN", True)
    processor.write_survival_split(X_test, survival_test, event_test, discrete_test, "TEST", True)
    

    metadata = {
        'time_bins': config['time_bins'],
        'bin_edges': bin_edges.tolist(),
        'max_time': float(np.max(survival_times)),
        'dropout_threshold': config['dropout_threshold'],
        'optimized_params': processor.save_current_params()
    }
    
    os.makedirs(config['end_path'], exist_ok=True)
    with open(os.path.join(config['end_path'], "survival_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Survival dataset created successfully!")

if __name__ == "__main__":
    main()
