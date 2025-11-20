# Changes Required for Multivariate Time Series Support

This document outlines the files that need to be modified to support multivariate time series data in the Continual_TSDiff repository.

## Overview

Currently, the codebase assumes univariate time series data (single feature dimension). The following changes are required to support multivariate time series where each time step can have multiple features/variables.

## Key Files to Modify

### 1. `src/uncond_ts_diff/configs.py`

**Location:** `/export/home/anandr/diffusion/Continual_TSDiff/src/uncond_ts_diff/configs.py`

**Changes needed:**
- Change all `"input_dim": 1` to accept a parameter (e.g., `num_features`)
- Change all `"output_dim": 1` to accept a parameter (e.g., `num_features`)
- This affects all backbone configurations:
  - `residual_block_s4_backbone`
  - `residual_block_s4_backbone_smallv2`
  - `residual_block_s4_backbone_small`
  - `residual_block_s4_backbone_small_dropout01`
  - `residual_block_s4_backbone_small_dropout02`
  - `residual_block_s4_backbone_small_dropout03`
  - `residual_block_s4_backbone_large`

**Current code (lines 6-8):**
```python
residual_block_s4_backbone = {
    "input_dim": 1,
    "hidden_dim": 128,
    "output_dim": 1,
    ...
}
```

**Should become:**
```python
def create_backbone_config(num_features=1, hidden_dim=128, ...):
    return {
        "input_dim": num_features,
        "hidden_dim": hidden_dim,
        "output_dim": num_features,
        ...
    }
```

---

### 2. `src/uncond_ts_diff/model/diffusion/tsdiff.py`

**Location:** `/export/home/anandr/diffusion/Continual_TSDiff/src/uncond_ts_diff/model/diffusion/tsdiff.py`

**Changes needed:**

#### a. Add `num_features` parameter to `__init__` (around line 14-32)
- Add `num_features: int = 1` parameter
- Store it as `self.num_features`

#### b. Update `_extract_features` method (lines 77-133)
- **Line 92:** `x = torch.cat([scaled_context, scaled_future], dim=1)` - This assumes univariate
  - **Issue:** For multivariate, `scaled_context` and `scaled_future` may have shape `(batch, time, features)`
  - **Fix:** Check if data already has feature dimension or add dimension only for univariate
  
- **Line 129:** `x = x[:, :, None]` - This hardcodes univariate by adding dimension
  - **Issue:** For multivariate, data should already have feature dimension
  - **Fix:** 
    ```python
    if x.dim() == 2:  # Univariate case: (batch, time)
        x = x[:, :, None]  # -> (batch, time, 1)
    # For multivariate: x already has shape (batch, time, features)
    ```

- **Line 127:** `x = torch.cat([x[:, :, None], lags], dim=-1)` - Similar issue
  - **Fix:** Handle both univariate and multivariate cases

- **Line 133:** `return x, scale[:, :, None], features`
  - **Issue:** `scale[:, :, None]` assumes univariate
  - **Fix:** Ensure scale has correct shape for multivariate (may need `scale.unsqueeze(-1)` or keep as is depending on scaler)

#### c. Update `sample_n` method (lines 135-157)
- **Line 157:** `return samples[..., 0]` - This assumes single feature dimension
  - **Issue:** For multivariate, should return all features
  - **Fix:** 
    ```python
    if return_lags:
        return samples
    # For univariate, return first feature; for multivariate, return all
    if self.num_features == 1:
        return samples[..., 0]
    else:
        return samples
    ```

#### d. Update `__init__` backbone_parameters (lines 54-57)
- Currently adjusts `input_dim` and `output_dim` based on lags
- Need to account for `num_features`:
  ```python
  if use_lags:
      backbone_parameters = backbone_parameters.copy()
      backbone_parameters["dropout"] = dropout_rate
      # For multivariate: input_dim should be num_features + num_lags
      backbone_parameters["input_dim"] = num_features + len(self.lags_seq)
      backbone_parameters["output_dim"] = num_features + len(self.lags_seq)
  else:
      backbone_parameters["input_dim"] = num_features
      backbone_parameters["output_dim"] = num_features
  ```

---

### 3. `src/uncond_ts_diff/model/diffusion/tsdiff_cond.py`

**Location:** `/export/home/anandr/diffusion/Continual_TSDiff/src/uncond_ts_diff/model/diffusion/tsdiff_cond.py`

**Changes needed:**

#### a. Add `num_features` parameter to `__init__` (around line 16-34)

#### b. Update `_extract_features` method (lines 73-134)
- **Line 89:** `x = torch.cat([scaled_orig_context, scaled_future], dim=1)`
- **Line 90:** `observation_mask = torch.zeros_like(x, device=device)`
- **Lines 94-95:** Handle multivariate shape
- **Line 131:** `x[..., None]` - Similar to tsdiff.py, handle multivariate case
- **Line 133:** Ensure correct shape handling

#### c. Update `forward` method (lines 220-268)
- **Line 244-246:** `future_target=torch.zeros(past_target.shape[0], self.prediction_length, device=device)`
  - **Issue:** Should include feature dimension for multivariate
  - **Fix:** `torch.zeros(past_target.shape[0], self.prediction_length, self.num_features, device=device)`

- **Line 268:** `return pred[:, None, length - self.prediction_length :, 0]`
  - **Issue:** Indexing `[..., 0]` assumes univariate
  - **Fix:** Return all features for multivariate

---

### 4. Training Scripts

#### a. `bin/train_model.py`

**Location:** `/export/home/anandr/diffusion/Continual_TSDiff/bin/train_model.py`

**Changes needed:**
- **Line 36-48:** `create_model` function
  - Need to determine `num_features` from dataset
  - Pass `num_features` to model initialization
  - Update config to use dynamic `input_dim`/`output_dim`

**Add code to determine num_features:**
```python
def get_num_features(dataset):
    # Option 1: From metadata (if available)
    if hasattr(dataset.metadata, 'feat_static_cat') and hasattr(dataset.metadata, 'num_feat_dynamic_real'):
        # Check first training example
        example = next(iter(dataset.train))
        if example['target'].ndim == 2:
            return example['target'].shape[-1]  # (time, features)
        return 1  # Univariate
    # Option 2: Check actual data shape
    example = next(iter(dataset.train))
    target = example['target']
    if target.ndim == 2 and target.shape[1] > 1:
        return target.shape[1]  # Multivariate
    return 1  # Univariate
```

- **Line 132:** When creating model, get num_features and pass it:
  ```python
  num_features = get_num_features(dataset)
  model = create_model(config, num_features=num_features)
  ```

#### b. `bin/train_cond_model.py`

**Location:** `/export/home/anandr/diffusion/Continual_TSDiff/bin/train_cond_model.py`

**Similar changes as train_model.py:**
- Determine `num_features` from dataset
- Pass to model initialization

---

### 5. `src/uncond_ts_diff/utils.py`

**Location:** `/export/home/anandr/diffusion/Continual_TSDiff/src/uncond_ts_diff/utils.py`

**Changes needed:**
- **Line 151-154:** `AsNumpyArray(field=FieldName.TARGET, expected_ndim=1, ...)`
  - **Issue:** For multivariate, target has `ndim=2` (time, features)
  - **Fix:** Either make `expected_ndim` configurable or check if it handles both cases
  - **Note:** GluonTS `AsNumpyArray` may handle this automatically, but verify

### 6. `src/uncond_ts_diff/model/diffusion/_base.py`

**Location:** `/export/home/anandr/diffusion/Continual_TSDiff/src/uncond_ts_diff/model/diffusion/_base.py`

**Changes needed:**
- **Line 71:** `self.scaler = MeanScaler(dim=1, keepdim=True)`
  - **Issue:** May need to verify scaler works correctly with multivariate data
  - The GluonTS scalers should handle multivariate, but verify the `dim` parameter
  - For multivariate, scaler should scale per-feature (dim=1) or globally

---

### 7. Additional Considerations

#### a. Scaler Handling
- Verify `MeanScaler` and `NOPScaler` handle multivariate correctly
- For multivariate, scaling might need to be per-feature or global
- Check if `scale` output has shape `(batch, 1)` or `(batch, features)`

#### b. Lag Handling
- `lagged_sequence_values` from GluonTS should handle multivariate
- Verify that lags are computed per-feature or across features as intended

#### c. Evaluation Scripts
- Update evaluation scripts (`guidance_experiment.py`, `refinement_experiment.py`, etc.) to handle multivariate
- Metrics calculation may need adjustment for multivariate forecasts

---

## Testing Strategy

1. **Start with a simple multivariate dataset** (2-3 features)
2. **Test univariate backward compatibility** - ensure existing univariate configs still work
3. **Verify scaling** - check that normalization works correctly per-feature
4. **Check sample generation** - ensure `sample_n` produces correct shape
5. **Validate forecasts** - compare multivariate forecasts with ground truth

---

## Summary of Critical Changes

| File | Key Changes | Priority |
|------|-------------|----------|
| `configs.py` | Make `input_dim`/`output_dim` configurable | **High** |
| `tsdiff.py` | Handle multivariate shapes in `_extract_features` and `sample_n` | **High** |
| `tsdiff_cond.py` | Similar shape handling for conditional model | **High** |
| `train_model.py` | Infer and pass `num_features` | **High** |
| `train_cond_model.py` | Infer and pass `num_features` | **High** |
| `utils.py` | Check `AsNumpyArray` for multivariate | **Medium** |
| `_base.py` | Verify scaler handles multivariate | **Medium** |
| Evaluation scripts | Update for multivariate metrics | **Medium** |

---

## Notes

- GluonTS datasets typically provide targets as:
  - **Univariate:** `(time,)` or `(batch, time)`
  - **Multivariate:** `(time, features)` or `(batch, time, features)`
  
- The code needs to handle both cases gracefully, likely by checking tensor dimensions and adding feature dimension only when needed.

- Consider adding a `num_features` parameter to config files for explicit control, while also auto-detecting from data as fallback.

