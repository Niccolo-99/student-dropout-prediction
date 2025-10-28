# Survival Dataset Builders · Unitelma · KDDCup15 · XuetangX

These three scripts turn raw course logs into **multivariate UEA time-series (.ts)** for *dropout survival analysis* (time-to-event with censoring) and produce a metadata JSON. 

## How they work
1. *Load raw data*  
   Read dataset-specific CSVs (logs, course info, labels).
2. *Build daily trajectories*  
   For each student/enrollment, assemble a days x activities matrix (daily counts per event/action).
3. *Data analysis & auto-tuning (optional)*  
   Estimate *sparsity, **gaps* between active days, and *activity stats* to auto-set:
   max_normal_gap, smoothing_window, lookback_days, window_size,  
   min_activity_threshold, high/medium_sparsity_threshold, activity_drop_threshold.
4. *Dropout day detection*  
   Three methods (original / gap-based / window-based) plus a *hybrid* chooser based on sparsity.  
   *Early-dropout correction*: if the predicted day is too early but ≥30% of total activity occurs after it, the day is shifted to a dataset-specific minimum.
5. *Survival labels*  
   For each student: survival_time (days), event (1=dropout, 0=censored), discrete_bin (time bin).
6. *Time discretization*  
   Map times into time_bins (XuetangX can *auto-estimate* bins).
7. *Train/test split*  
   Stratified by event when both events and censored cases exist.
8. *Write files*  
   .ts with UEA header and @survivalAnalysis true + a .json with metadata.

**.ts line format**  
dim1_t0,dim1_t1,...:dim2_t0,dim2_t1,...:...:(survival_time:event:discrete_bin)

---

## Dataset-specific details

| Dataset     | Script/Class                  | Expected inputs                                                                                           | Key columns                      | Early-dropout min |
|-------------|-------------------------------|------------------------------------------------------------------------------------------------------------|----------------------------------|-------------------|
| *KDDCup15* | KDDCup15SurvivalDataset     | date.csv, object.csv, train/{enrollment,log,truth}_train.csv, test/{enrollment,log,truth}_test.csv | event, time, enrollment_id | *7* days        |
| *Unitelma* | UnitelmaDatasetProcessor    | CSVs under origin_path with: student_id, dropout, timeseries (matrix as string)                    | — (series already assembled)     | *14* days       |
| *XuetangX* | XuetangXSurvivalDataset     | course_info.csv, user_info.csv, prediction_log/{train_log,train_truth,test_log,test_truth}.csv       | action, time, enroll_id    | *10* days       |

---

## Produced outputs

- .ts (per dataset):  
  - KDDCup_Survival_{TRAIN,TEST}.ts  
  - Unitelma_Survival_{TRAIN,TEST}.ts  
  - XuetangX_Survival_{TRAIN,TEST}.ts
- Metadata .json:  
  - kddcup_survival_metadata.json, survival_metadata.json (Unitelma), xuetangx_survival_metadata.json  
  Each contains: time_bins, bin_edges, max_time, series dimensions/length, *optimized parameters*.

---

## Minimal usage examples

*KDDCup15*
```python
from KDDcup15_survival_builder import KDDCup15SurvivalDataset

proc = KDDCup15SurvivalDataset(
    dataset_root="dataset/KDDcup/kddcup15",
    dataset_out="dataset/KDDcup/kddcup15_survival",
    time_bins=4,
    auto_optimize=True
)
proc.create_survival_dataset(test_size=0.2, random_state=42)
