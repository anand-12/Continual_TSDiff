#!/usr/bin/env python3
"""
TSDiff Continual Learning Comparison Plots
Clean script with time series selection capability
"""
import os

# Set CUDA path for apt-installed CUDA toolkit
os.environ['CUDA_PATH'] = '/usr'
os.environ['CUDA_HOME'] = '/usr'


import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from pathlib import Path
from tqdm.auto import tqdm
from typing import Tuple


from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.field_names import FieldName


import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    MaskInput,
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


GUIDANCE_MAP = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}



class TSDiffPlotter:
    """Simple TSDiff forecast plotter with time series selection"""
    
    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.device = torch.device(config.get("device", "cuda:1"))
        self.model, self.prediction_length = self._load_model(checkpoint_path)  # Unpack both values

        
    def _load_model(self, checkpoint_path: str) -> Tuple[TSDiff, int]:
        """Load TSDiff model from checkpoint and return prediction length"""
        logger.info(f"Loading: {Path(checkpoint_path).name}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract prediction_length from checkpoint if available
        pred_len = checkpoint.get('config', {}).get('prediction_length', 24)
        
        model = TSDiff(
            **getattr(diffusion_configs, self.config["diffusion_config"]),
            freq=self.config["freq"],
            use_features=self.config["use_features"],
            use_lags=self.config["use_lags"],
            normalization=self.config["normalization"],
            context_length=self.config["context_length"],
            prediction_length=pred_len,  # Use checkpoint's prediction_length
            lr=self.config["lr"],
            init_skip=self.config["init_skip"],
            dropout_rate=self.config.get("dropout_rate", 0.01),
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model, pred_len



    def generate_forecasts(self, dataset_name: str, start_index: int = 0, num_series: int = 1, num_samples: int = 100):
        """Generate forecasts using checkpoint's prediction_length"""
        # Use the model's prediction_length instead of config
        dataset = get_dataset(dataset_name, prediction_length=self.prediction_length, regenerate=False)
        
        # Auto-correct frequency
        actual_freq = str(dataset.metadata.freq)
        if actual_freq != self.config.get("freq", "H"):
            self.config["freq"] = actual_freq
        
        # Setup transformation - USE self.prediction_length
        transformation = create_transforms(
            num_feat_dynamic_real=0, num_feat_static_cat=0, num_feat_static_real=0,
            time_features=self.model.time_features,
            prediction_length=self.prediction_length,  # CHANGED
        )
        
        Guidance = GUIDANCE_MAP[self.config["sampler"]]
        sampler_kwargs = self.config.get("sampler_params", {})
        sampler = Guidance(
            model=self.model, 
            prediction_length=self.prediction_length,  # CHANGED
            num_samples=num_samples, 
            missing_scenario="none", 
            missing_values=0,
            **sampler_kwargs,
        )
        
        # Setup data pipeline
        transformed_testdata = transformation.apply(dataset.test, is_train=False)
        test_splitter = create_splitter(
            past_length=self.config["context_length"] + max(self.model.lags_seq),
            future_length=self.prediction_length,  # CHANGED
            mode="test",
        )
        masking_transform = MaskInput(
            FieldName.TARGET, FieldName.OBSERVED_VALUES,
            self.config["context_length"], "none", 0,
        )
        test_transform = test_splitter + masking_transform
        
        # Create predictor
        predictor = sampler.get_predictor(
            test_transform, batch_size=max(1, 1280 // num_samples), device=str(self.device),
        )
        
        # Select specific time series range
        selected_series = []
        try:
            testdata_list = list(transformed_testdata)
            end_index = min(start_index + num_series, len(testdata_list))
            
            if start_index >= len(testdata_list):
                logger.warning(f"Start index {start_index} >= dataset size {len(testdata_list)}. Using index 0.")
                start_index = 0
                end_index = min(num_series, len(testdata_list))
            
            selected_series = testdata_list[start_index:end_index]
            
        except MemoryError:
            logger.info(f"Large dataset detected, using itertools for indices {start_index}-{start_index+num_series-1}")
            selected_series = list(itertools.islice(transformed_testdata, start_index, start_index + num_series))
        
        if not selected_series:
            logger.warning(f"No series found at indices {start_index}-{start_index+num_series-1}. Using first series.")
            selected_series = [next(iter(transformed_testdata))]
        
        logger.info(f"Selected {len(selected_series)} time series starting from index {start_index}")
        
        # Generate forecasts for selected series
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=selected_series, predictor=predictor, num_samples=num_samples,
        )
        
        forecasts = list(tqdm(forecast_it, total=len(selected_series), desc="Generating forecasts"))
        tss = list(ts_it)
        
        return forecasts, tss, dataset.metadata.freq


    def plot_forecast(self, forecast, ts, freq, method_name: str, ax, is_leftmost: bool = False, ylim=None):
        """Plot single forecast on given axis"""
        # Handle time series data
        historical_data = ts
        forecast_start = forecast.start_date
        
        # Handle PeriodIndex
        if hasattr(ts, 'index') and isinstance(ts.index, pd.PeriodIndex):
            historical_index = ts.index.to_timestamp()
        else:
            historical_index = ts.index if hasattr(ts, 'index') else pd.date_range(
                end=forecast_start - pd.Timedelta(hours=1), periods=len(historical_data), freq=freq
            )
        
        # Handle forecast start
        if hasattr(forecast_start, 'to_timestamp'):
            forecast_start = forecast_start.to_timestamp()
        
        # Create forecast index
        forecast_index = pd.date_range(start=forecast_start, periods=len(forecast.mean), freq=freq)
        
        # Plot historical data (last portion)
        context_length = min(len(historical_data), self.config["context_length"])
        hist_start = max(0, len(historical_data) - context_length - 24)
        
        ax.plot(historical_index[hist_start:], historical_data[hist_start:], 
               'b-', linewidth=1.5, label='Ground Truth', alpha=0.8)
        
        # Plot forecast
        ax.plot(forecast_index, forecast.quantile(0.75), 'r-', linewidth=2, label='Median Forecast')
        
        # Plot prediction intervals
        if hasattr(forecast, 'quantile'):
            lower_90 = forecast.quantile(0.05)
            upper_90 = forecast.quantile(0.95)
            ax.fill_between(forecast_index, lower_90, upper_90, 
                           alpha=0.3, color='indianred', label='90% Interval')
        
        # Add forecast separator
        ax.axvline(x=forecast_start, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
        
        # Set consistent y-limits if provided
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Formatting
        # ax.set_title(f'{method_name}', fontsize=14)
        ax.xaxis.set_visible(False)
        ax.grid(True, alpha=0.3)
        
        # Only show y-axis and legend on leftmost subplot
        # if is_leftmost:
        #     ax.legend(fontsize=10, loc='best')
        # else:
        #     ax.set_yticklabels([])
        #     ax.tick_params(axis='y', which='both', left=False, labelleft=False)


def create_continual_learning_plots(start_series: int = 0, num_series: int = 1):
    """Create 5-method comparison plots with configurable series selection"""
    
    checkpoints = {
        # "2 hour": "/export/home/anandr/diffusion/Continual_TSDiff/logs_2hr/pedestrian_counts_checkpoint_best.pth",
        "50 hour": "/local/home0/anandr/diffusion/Continual_TSDiff/logs_50hr/pedestrian_counts_checkpoint_best.pth",
        # "24 hour": "/export/home/anandr/diffusion/Continual_TSDiff/logs_24hr/pedestrian_counts_checkpoint_best.pth",
        # "48 hour": "/export/home/anandr/diffusion/Continual_TSDiff/logs_48hr/pedestrian_counts_checkpoint_best.pth",
        # "96 hour": "/export/home/anandr/diffusion/Continual_TSDiff/logs_96hr/pedestrian_counts_checkpoint_best.pth",
        # "L2 Reg": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_score_l2/lambda_reg_2.0/task_1_kdd_cup_2018_without_missing/kdd_cup_2018_without_missing_checkpoint_best.pth",
    }
    
    config = yaml.safe_load(open("configs/eval_continual.yaml"))
    target_dataset = "pedestrian_counts"
    
    # Set seeds for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Store all forecast data for y-limit calculation
    all_forecast_data = {}
    all_y_values = []
    
    logger.info(f"First pass: Collecting data for {num_series} time series starting from index {start_series}...")
    
    # First pass: collect all data and y-values
    for method_idx, (method_name, checkpoint_path) in enumerate(checkpoints.items()):
        try:
            plotter = TSDiffPlotter(config, checkpoint_path)
            forecasts, tss, freq = plotter.generate_forecasts(
                target_dataset, start_index=start_series, num_series=num_series, num_samples=1000
            )
            
            # Store forecast data for second pass
            all_forecast_data[method_name] = {
                'forecasts': forecasts,
                'tss': tss,
                'freq': freq,
                'plotter': plotter
            }
            
            # Collect y-values for global scaling
            for series_idx in range(len(forecasts)):
                ts = tss[series_idx]
                forecast = forecasts[series_idx]
                
                # Historical data
                context_length = min(len(ts), config["context_length"])
                hist_start = max(0, len(ts) - context_length - 24)
                all_y_values.extend(ts[hist_start:])
                
                # Forecast data
                all_y_values.extend(forecast.mean)
                if hasattr(forecast, 'quantile'):
                    all_y_values.extend(forecast.quantile(0.05))
                    all_y_values.extend(forecast.quantile(0.95))
            
            logger.info(f"SUCCESS: {method_name} data collection completed")
            
        except Exception as e:
            logger.error(f"FAILED: {method_name} failed: {e}")
            continue
    
    # Calculate global y-limits with padding
    if all_y_values:
        y_min, y_max = min(all_y_values), max(all_y_values)
        y_padding = (y_max - y_min) * 0.05  # 5% padding
        global_ylim = (y_min - y_padding, y_max + y_padding)
        logger.info(f"Global Y-limits: {global_ylim}")
    else:
        global_ylim = None
        logger.warning("No y-values collected, using automatic scaling")
    
# Determine figure size
    figsize = (25, 5 * num_series) if num_series > 1 else (25, 5)
    num_methods = len(checkpoints)

    if num_series == 1 and num_methods == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = np.array([[ax]])
    elif num_series == 1:
        fig, axes = plt.subplots(1, num_methods, figsize=figsize, squeeze=False)
    elif num_methods == 1:
        fig, axes = plt.subplots(num_series, 1, figsize=figsize, squeeze=False)
    else:
        fig, axes = plt.subplots(num_series, num_methods, figsize=figsize, squeeze=False)

        logger.info("Second pass: Creating plots with synchronized y-limits...")
        
    # Second pass: plot with synchronized y-limits
    for method_idx, method_name in enumerate(checkpoints.keys()):
        if method_name not in all_forecast_data:
            # Handle failed methods
            if num_series == 1:
                ax = axes[method_idx]
                ax.text(0.5, 0.5, f"Error\n{method_name}", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
                if global_ylim:
                    ax.set_ylim(global_ylim)
            else:
                for series_idx in range(num_series):
                    ax = axes[series_idx, method_idx]
                    ax.text(0.5, 0.5, f"Error\n{method_name}", ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12, color='red')
                    if global_ylim:
                        ax.set_ylim(global_ylim)
            continue
        
        # Get stored data
        data = all_forecast_data[method_name]
        forecasts = data['forecasts']
        tss = data['tss']
        freq = data['freq']
        plotter = data['plotter']
        
        # Plot each time series
        for series_idx in range(len(forecasts)):
            if num_series == 1:
                ax = axes[method_idx]
            else:
                ax = axes[series_idx, method_idx]
            
            # Check if this is the leftmost subplot
            is_leftmost = (method_idx == 0)
            
            plotter.plot_forecast(forecasts[series_idx], tss[series_idx], freq, 
                                method_name, ax, is_leftmost=is_leftmost, ylim=global_ylim)
        
        logger.info(f"SUCCESS: {method_name} plots completed")
    
    plt.tight_layout()
    
    # Save plot
    if num_series == 1:
        output_file = f'continual_learning_comparison_series_{start_series}.png'
    else:
        output_file = f'continual_learning_comparison_series_{start_series}_to_{start_series+num_series-1}.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Comparison plot saved: {output_file}")


def main():
    """Main function with configurable parameters"""
    logger.info("Starting TSDiff continual learning comparison")
    
    # Configure which time series to plot
    create_continual_learning_plots(start_series=100, num_series=5)
    
    logger.info("Plotting completed successfully!")


if __name__ == "__main__":
    main()
