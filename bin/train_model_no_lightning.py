#!/usr/bin/env python3
"""
TSDiff Training Script for Continual Learning
Optimized for single-frequency (hourly) datasets
"""

import logging
import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F  # ADD THIS IMPORT

from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.dataset.split import OffsetSplitter, split
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository.datasets import get_dataset

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    filter_metrics,
    MaskInput,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global constants
GUIDANCE_MAP = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}

class ModelCheckpoint:
    """Handle model checkpointing with best model tracking"""
    
    def __init__(self, checkpoint_dir: Path, monitor: str = 'val_loss', mode: str = 'min'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
    def should_save(self, metrics: Dict[str, float]) -> bool:
        """Check if current model should be saved"""
        current_score = metrics.get(self.monitor, float('inf'))
        
        if self.mode == 'min' and current_score < self.best_score:
            self.best_score = current_score
            return True
        elif self.mode == 'max' and current_score > self.best_score:
            self.best_score = current_score
            return True
        return False
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    epoch: int, metrics: Dict[str, float], dataset_name: str, config: dict) -> Path:
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{dataset_name}_checkpoint_best.pth"
        last_checkpoint_path = self.checkpoint_dir / f"{dataset_name}_checkpoint_last.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score,
            'dataset_name': dataset_name,
            'config': config  # ADD THIS LINE
        }
            
        # Store the result to avoid calling should_save() twice
        is_best = self.should_save(metrics)
        
        # Save best checkpoint if score improved
        if is_best:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best checkpoint: {checkpoint_path}")
        
        # Always save last checkpoint
        torch.save(checkpoint, last_checkpoint_path)
        
        # Return the correct path
        return checkpoint_path if is_best else last_checkpoint_path


class EarlyStopping:
    """Early stopping with patience"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

class TSDiffTrainer:
    """Main trainer class for TSDiff continual learning"""
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lambda_reg = config.get("lambda_reg", 0.0)
        self.score_loss_type = config.get("score_loss_type", "l2")  # Add this line
        self.previous_model = None
        logger.info(f"Continual learning lambda_reg: {self.lambda_reg}")


        # Setup device
        self.device = torch.device(config["device"])
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.dataset = self._load_dataset()
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.transformation = self._create_transformation()
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
        # Setup training utilities
        self.checkpoint_manager = ModelCheckpoint(self.output_dir, monitor='val_loss')
        self.early_stopping = EarlyStopping(patience=20)
        
    def compute_weight_regularization_loss(self) -> torch.Tensor:
        """Compute L1/L2 weight regularization loss"""
        if self.l1_weight == 0.0 and self.l2_weight == 0.0:
            return torch.tensor(0.0, device=self.device, requires_grad=False)
        
        l1_loss = torch.tensor(0.0, device=self.device)
        l2_loss = torch.tensor(0.0, device=self.device)
        
        for param in self.model.parameters():
            if param.requires_grad:
                l1_loss += torch.sum(torch.abs(param))
                l2_loss += torch.sum(param ** 2)
        
        total_reg_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        return total_reg_loss
    def _load_dataset(self):
        """Load dataset with frequency auto-correction"""
        try:
            dataset = get_dataset(
                self.config["dataset"],
                prediction_length=2,  # Standardized for all datasets
                regenerate=True 
            )
            
            # Auto-correct frequency if needed
            actual_freq = str(dataset.metadata.freq)
            if actual_freq != self.config.get("freq", "H"):
                logger.warning(f"Frequency mismatch detected!")
                logger.warning(f"  Config freq: {self.config.get('freq', 'H')}")
                logger.warning(f"  Dataset freq: {actual_freq}")
                logger.warning(f"  Using dataset's native frequency: {actual_freq}")
                self.config["freq"] = actual_freq
            
            # Verify prediction length
            assert dataset.metadata.prediction_length == 2, \
                f"Expected prediction_length=2, got {dataset.metadata.prediction_length}"
            
            logger.info(f"Dataset {self.config['dataset']} loaded successfully")
            logger.info(f"  Frequency: {dataset.metadata.freq}")
            logger.info(f"  Prediction length: {dataset.metadata.prediction_length}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {self.config['dataset']}: {e}")
            raise
    
    def _create_model(self) -> TSDiff:
        """Create and initialize TSDiff model"""
        model = TSDiff(
            **getattr(diffusion_configs, self.config["diffusion_config"]),
            freq=self.config["freq"],
            use_features=self.config["use_features"],
            use_lags=self.config["use_lags"],
            normalization=self.config["normalization"],
            context_length=self.config["context_length"],
            prediction_length=self.config["prediction_length"],
            lr=self.config["lr"],
            init_skip=self.config["init_skip"],
            dropout_rate=self.config.get("dropout_rate", 0.01),  # ADD THIS LINE
        )
        model.to(self.device)
        logger.info(f"Model created with dropout_rate={self.config.get('dropout_rate', 0.01)}")
        return model

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        return optim.Adam(self.model.parameters(), lr=self.config["lr"])
    
    def _create_transformation(self):
        """Create data transformation pipeline"""
        return create_transforms(
            num_feat_dynamic_real=0,
            num_feat_static_cat=0,
            num_feat_static_real=0,
            time_features=self.model.time_features,
            prediction_length=self.config["prediction_length"],
        )
    
    def _setup_data_loaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Setup training and validation data loaders"""
        # Get training data
        training_data = self.dataset.train
        
        # Create training splitter
        training_splitter = create_splitter(
            past_length=self.config["context_length"] + max(self.model.lags_seq),
            future_length=self.config["prediction_length"],
            mode="train",
        )
        
        # Setup validation if requested
        val_loader = None
        if self.config.get("use_validation_set", False):
            val_data = training_data
            training_data, _ = split(val_data, offset=-self.config["prediction_length"])
            
            val_splitter = create_splitter(
                past_length=self.config["context_length"] + max(self.model.lags_seq),
                future_length=self.config["prediction_length"],
                mode="val",
            )
            
            val_loader = ValidationDataLoader(
                val_data,
                batch_size=self.config["batch_size"],
                stack_fn=batchify,
                transform=self.transformation + val_splitter,
            )
        
        # Create training data loader
        transformed_data = self.transformation.apply(training_data, is_train=True)
        train_loader = TrainDataLoader(
            Cached(transformed_data),
            batch_size=self.config["batch_size"],
            stack_fn=batchify,
            transform=training_splitter,
            num_batches_per_epoch=self.config["num_batches_per_epoch"],
            shuffle_buffer_length=10000,
        )
        
        return train_loader, val_loader
    
    def _move_batch_to_device(self, batch):
        """Recursively move batch data to device"""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {key: self._move_batch_to_device(value) for key, value in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch_to_device(item) for item in batch)
        return batch
    
    def _extract_loss(self, output) -> torch.Tensor:
        """Extract loss tensor from model output"""
        if isinstance(output, dict):
            for key in ['loss', 'train_loss', 'total_loss']:
                if key in output:
                    return output[key]
            available_keys = list(output.keys())
            raise KeyError(f"No loss found in output. Available keys: {available_keys}")
        else:
            return output

    def set_previous_model(self, checkpoint_path: str):
        """Load and freeze previous model for regularization"""
        print(f"DEBUG: set_previous_model called with path: {checkpoint_path}")
        print(f"DEBUG: lambda_reg = {self.lambda_reg}")
        print(f"DEBUG: Checkpoint exists: {os.path.exists(checkpoint_path) if checkpoint_path else 'None'}")
        
        if checkpoint_path is None or self.lambda_reg == 0.0:
            self.previous_model = None
            logger.info("No previous model set (lambda_reg=0 or no checkpoint)")
            print(f"DEBUG: Exiting early - checkpoint_path={checkpoint_path}, lambda_reg={self.lambda_reg}")
            return
            
        logger.info(f"Loading previous model from: {checkpoint_path}")
        try:
            # Check if file exists
            if not os.path.exists(checkpoint_path):
                print(f"DEBUG ERROR: Checkpoint file does not exist: {checkpoint_path}")
                self.previous_model = None
                return
                
            # Create and load previous model
            self.previous_model = TSDiff(
                **getattr(diffusion_configs, self.config["diffusion_config"]),
                freq=self.config["freq"],
                use_features=self.config["use_features"],
                use_lags=self.config["use_lags"],
                normalization=self.config["normalization"],
                context_length=self.config["context_length"],
                prediction_length=self.config["prediction_length"],
                lr=self.config["lr"],
                init_skip=self.config["init_skip"],
                dropout_rate=self.config.get("dropout_rate", 0.3),
            )
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"DEBUG: Checkpoint loaded, keys: {list(checkpoint.keys())}")
            
            self.previous_model.load_state_dict(checkpoint['model_state_dict'])
            self.previous_model.to(self.device)
            self.previous_model.eval()
            
            # Freeze previous model parameters
            for param in self.previous_model.parameters():
                param.requires_grad = False
            
            print(f"DEBUG SUCCESS: Previous model loaded successfully")
            print(f"DEBUG: Previous model is None: {self.previous_model is None}")
            logger.info("Previous model loaded and frozen successfully")
            
        except Exception as e:
            logger.error(f"Failed to load previous model: {e}")
            print(f"DEBUG ERROR: Exception during loading: {e}")
            self.previous_model = None
            raise


    def compute_score_regularization_loss(self, batch, loss_type="l2") -> torch.Tensor:
        """Compute score function regularization loss with L1 or L2 distance"""
        if self.previous_model is None or self.lambda_reg == 0.0:
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        try:
            # Extract features from batch
            if isinstance(batch, dict):
                x, _, features = self.model._extract_features(batch)
            else:
                x, _ = self.model.scaler(batch, torch.ones_like(batch))
                features = None

            # Sample random timesteps and noise
            batch_size = x.shape[0]
            t = torch.randint(0, self.model.timesteps, (batch_size,), device=self.device).long()
            noise = torch.randn_like(x, device=self.device)

            # Get noisy x_t using the forward diffusion process
            x_t = self.model.q_sample(x_start=x, t=t, noise=noise)

            # Get noise predictions from both models
            with torch.no_grad():
                noise_pred_old = self.previous_model.backbone(x_t, t, features)
            noise_pred_new = self.model.backbone(x_t, t, features)

            # Compute regularization loss with specified distance metric
            if loss_type == "l1":
                reg_loss = F.l1_loss(noise_pred_new, noise_pred_old, reduction='mean')
            elif loss_type == "l2":
                reg_loss = F.mse_loss(noise_pred_new, noise_pred_old, reduction='mean')
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            final_loss = self.lambda_reg * reg_loss
            return final_loss

        except Exception as e:
            logger.warning(f"Error computing regularization loss: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=False)


    def train_epoch(self) -> float:
        """Train for one epoch with score regularization only"""
        self.model.train()
        if self.previous_model is not None:
            self.previous_model.eval()

        total_loss = 0
        total_standard_loss = 0
        total_score_reg_loss = 0
        num_batches = 0

        # Update description based on regularization
        reg_methods = []
        if self.lambda_reg > 0: 
            reg_methods.append(f"Score-{self.score_loss_type.upper()}(λ={self.lambda_reg})")
        
        dropout_rate = self.config.get("dropout_rate", 0.0)
        if dropout_rate > 0: 
            reg_methods.append(f"Dropout({dropout_rate})")

        desc = "Training" if not reg_methods else f"Training [{'+'.join(reg_methods)}]"
        pbar = tqdm(self.train_loader, desc=desc)

        for batch_idx, batch in enumerate(pbar):
            batch = self._move_batch_to_device(batch)
            self.optimizer.zero_grad()

            # Standard diffusion loss
            try:
                output = self.model.training_step(batch)
            except TypeError:
                try:
                    output = self.model.training_step(batch, batch_idx)
                except TypeError:
                    output = self.model.training_step(batch, batch_idx, dataloader_idx=0)

            standard_loss = self._extract_loss(output)

            # Score regularization loss (with L1 or L2 distance)
            score_reg_loss = self.compute_score_regularization_loss(batch, self.score_loss_type)

            # Combined loss (remove weight regularization)
            total_loss_batch = standard_loss + score_reg_loss

            total_loss_batch.backward()

            # Gradient clipping
            if self.config.get("gradient_clip_val"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["gradient_clip_val"]
                )

            self.optimizer.step()

            # Update metrics
            total_loss += total_loss_batch.item()
            total_standard_loss += standard_loss.item()
            total_score_reg_loss += score_reg_loss.item()
            num_batches += 1

            # Update progress bar
            pbar_dict = {'std': f"{standard_loss.item():.4f}"}
            if self.lambda_reg > 0:
                pbar_dict[f'score_{self.score_loss_type}'] = f"{score_reg_loss.item():.4f}"
            pbar_dict['total'] = f"{total_loss_batch.item():.4f}"
            pbar.set_postfix(pbar_dict)

        # Log detailed metrics
        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_standard_loss = total_standard_loss / num_batches if num_batches > 0 else 0.0
        avg_score_reg_loss = total_score_reg_loss / num_batches if num_batches > 0 else 0.0

        logger.info(f"Epoch metrics - Standard: {avg_standard_loss:.6f}, "
                    f"Score_{self.score_loss_type.upper()}: {avg_score_reg_loss:.6f}, "
                    f"Total: {avg_total_loss:.6f}")

        return avg_total_loss


    # def train_epoch(self) -> float:
    #     """Train for one epoch"""
    #     self.model.train()
    #     total_loss = 0
    #     num_batches = 0
        
    #     pbar = tqdm(self.train_loader, desc="Training")
    #     for batch_idx, batch in enumerate(pbar):
    #         # Move batch to device
    #         batch = self._move_batch_to_device(batch)
            
    #         # Forward pass
    #         self.optimizer.zero_grad()
            
    #         try:
    #             output = self.model.training_step(batch)
    #         except TypeError:
    #             try:
    #                 output = self.model.training_step(batch, batch_idx)
    #             except TypeError:
    #                 output = self.model.training_step(batch, batch_idx, dataloader_idx=0)
            
    #         # Extract loss and backward pass
    #         loss = self._extract_loss(output)
    #         loss.backward()
            
    #         # Gradient clipping
    #         if self.config.get("gradient_clip_val"):
    #             torch.nn.utils.clip_grad_norm_(
    #                 self.model.parameters(), 
    #                 self.config["gradient_clip_val"]
    #             )
            
    #         self.optimizer.step()
            
    #         # Update metrics
    #         total_loss += loss.item()
    #         num_batches += 1
    #         pbar.set_postfix({'loss': loss.item()})
        
    #     return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate_epoch(self) -> Optional[float]:
        """Validate for one epoch"""
        if self.val_loader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch_idx, batch in enumerate(pbar):
                batch = self._move_batch_to_device(batch)
                
                # Try validation_step or fallback to training_step
                try:
                    output = self.model.validation_step(batch)
                except (TypeError, AttributeError):
                    try:
                        output = self.model.validation_step(batch, batch_idx)
                    except (TypeError, AttributeError):
                        try:
                            output = self.model.training_step(batch)
                        except TypeError:
                            output = self.model.training_step(batch, batch_idx)
                
                loss = self._extract_loss(output)
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'val_loss': loss.item()})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model and optimizer from checkpoint"""
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state dict
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            logger.info(f"Resuming from epoch {start_epoch}")
            return start_epoch
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.warning("Starting training from scratch")
            return 0
    
    def evaluate_model(self, num_samples: int = 100) -> list:
        """Evaluate model using guidance"""
        logger.info(f"Starting evaluation with {num_samples} samples")
        
        try:
            # Setup guidance sampler
            Guidance = GUIDANCE_MAP[self.config["sampler"]]
            sampler_kwargs = self.config.get("sampler_params", {})
            
            sampler = Guidance(
                model=self.model,
                prediction_length=self.config["prediction_length"],
                num_samples=num_samples,
                missing_scenario="none",
                missing_values=0,
                **sampler_kwargs,
            )
            
            # Setup test data transformation
            transformed_testdata = self.transformation.apply(self.dataset.test, is_train=False)
            test_splitter = create_splitter(
                past_length=self.config["context_length"] + max(self.model.lags_seq),
                future_length=self.config["prediction_length"],
                mode="test",
            )
            
            masking_transform = MaskInput(
                FieldName.TARGET,
                FieldName.OBSERVED_VALUES,
                self.config["context_length"],
                "none",
                0,
            )
            test_transform = test_splitter + masking_transform
            
            # Create predictor
            predictor = sampler.get_predictor(
                test_transform,
                batch_size=max(1, 1280 // num_samples),
                device=str(self.device),
            )
            
            # Generate predictions and evaluate
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=transformed_testdata,
                predictor=predictor,
                num_samples=num_samples,
            )
            
            forecasts = list(tqdm(forecast_it, total=len(transformed_testdata), desc="Generating forecasts"))
            tss = list(ts_it)
            
            # Compute metrics
            evaluator = Evaluator()
            metrics, _ = evaluator(tss, forecasts)
            filtered_metrics = filter_metrics(metrics)
            
            logger.info("Evaluation completed successfully")
            return [{"missing_scenario": "none", "missing_values": 0, **filtered_metrics}]
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return [{"error": str(e)}]
    
    def train(self, resume_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Main training loop"""
        logger.info(f"Starting training for {self.config['max_epochs']} epochs")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Load checkpoint if specified
        start_epoch = 0
        if resume_checkpoint:
            start_epoch = self.load_checkpoint(resume_checkpoint)
        
        # Training loop
        training_history = {'train_loss': [], 'val_loss': [], 'epochs': []}
        best_checkpoint_path = None
        
        try:
            for epoch in range(start_epoch, self.config["max_epochs"]):
                epoch_start_time = time.time()
                
                # Train epoch
                train_loss = self.train_epoch()
                
                # Validate epoch
                val_loss = None
                if self.val_loader is not None and epoch % self.config.get("eval_every", 10) == 0:
                    val_loss = self.validate_epoch()
                
                # Prepare metrics
                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss if val_loss is not None else train_loss
                }
                
                # Log progress
                epoch_time = time.time() - epoch_start_time
                log_msg = f"Epoch {epoch+1:03d}/{self.config['max_epochs']:03d} - "
                log_msg += f"train_loss: {train_loss:.6f}"
                if val_loss is not None:
                    log_msg += f" - val_loss: {val_loss:.6f}"
                log_msg += f" - time: {epoch_time:.2f}s"
                logger.info(log_msg)
                
                # Update training history
                training_history['epochs'].append(epoch + 1)
                training_history['train_loss'].append(train_loss)
                training_history['val_loss'].append(val_loss if val_loss is not None else train_loss)
                
                # Save checkpoint
                # Save checkpoint
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, metrics, self.config['dataset'], self.config  # Add self.config
                )



                # Update best checkpoint if this was the best
                if checkpoint_path.name.endswith('_checkpoint_best.pth'):
                    best_checkpoint_path = checkpoint_path
                
                # Early stopping
                monitor_loss = val_loss if val_loss is not None else train_loss
                if self.early_stopping(monitor_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        logger.info("Training completed")
        
        # Load best model for evaluation
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            logger.info(f"Loading best checkpoint for evaluation: {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        evaluation_metrics = None
        if self.config.get("do_final_eval", True):
            logger.info("Starting final evaluation")
            evaluation_metrics = self.evaluate_model()
            logger.info("Final evaluation completed")
        
        # Save results
        results = {
            "config": self.config,
            "training_history": training_history,
            "metrics": evaluation_metrics,
            "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path else None
        }
        
        results_path = self.output_dir / "results.yaml"
        with open(results_path, "w") as f:
            yaml.dump(results, f)
        
        logger.info(f"Results saved to: {results_path}")
        return results

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train TSDiff model")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--out_dir", type=str, default="./logs_2hr", help="Output directory")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                       help="Path to checkpoint file to resume from")
    return parser.parse_args()

def load_config(config_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Add command line arguments to config
    if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint: 
        config["resume_from_checkpoint"] = args.resume_from_checkpoint

    return config


def main():
    """Main training function"""
    try:
        # Parse arguments and load config
        args = parse_arguments()
        config = load_config(args.config, args)
        
        # Create trainer and run training
        trainer = TSDiffTrainer(config, args.out_dir)
        results = trainer.train(resume_checkpoint=config.get("resume_from_checkpoint"))
        
        logger.info("Training pipeline completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
