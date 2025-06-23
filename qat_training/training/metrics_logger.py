"""
Metrics logging and monitoring for QAT training
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsLogger(TrainerCallback):
    """Custom callback for logging training metrics and progress"""
    
    def __init__(self, output_dir: str):
        """
        Initialize metrics logger
        
        Args:
            output_dir: Directory to save logs and plots
        """
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, "logs")
        self.plots_dir = os.path.join(output_dir, "plots")
        
        # Create directories
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Training metrics storage
        self.training_logs = []
        self.eval_logs = []
        self.step_times = []
        
        # Training state
        self.start_time = None
        self.last_log_time = None
        
        # Setup logging files
        self.setup_logging_files()
    
    def setup_logging_files(self):
        """Setup logging files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Training log file
        self.train_log_file = os.path.join(self.logs_dir, f"training_{timestamp}.jsonl")
        
        # Metrics summary file
        self.metrics_file = os.path.join(self.logs_dir, f"metrics_{timestamp}.json")
        
        # Progress log file
        self.progress_file = os.path.join(self.logs_dir, f"progress_{timestamp}.txt")
        
        # Initialize progress file
        with open(self.progress_file, 'w') as f:
            f.write(f"QAT Training Progress Log - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training"""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Log training start
        start_info = {
            "event": "training_start",
            "timestamp": datetime.now().isoformat(),
            "total_steps": state.max_steps,
            "num_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
        
        self._log_event(start_info)
        
        # Progress log
        with open(self.progress_file, 'a') as f:
            f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total steps: {state.max_steps}\n")
            f.write(f"Epochs: {args.num_train_epochs}\n")
            f.write(f"Batch size: {args.per_device_train_batch_size}\n")
            f.write(f"Learning rate: {args.learning_rate}\n\n")
        
        logger.info("Training started - metrics logging initialized")
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """Called when logging metrics"""
        if logs is None:
            return
        
        current_time = time.time()
        step_time = current_time - self.last_log_time if self.last_log_time else 0
        self.last_log_time = current_time
        
        # Prepare log entry
        log_entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": current_time - self.start_time,
            "step_time": step_time,
            **logs
        }
        
        # Distinguish between training and evaluation logs
        if "eval_loss" in logs:
            self.eval_logs.append(log_entry)
            self._log_evaluation_progress(log_entry)
        else:
            self.training_logs.append(log_entry)
            self._log_training_progress(log_entry)
        
        # Save log entry
        self._log_event(log_entry)
        
        # Update plots periodically
        if state.global_step % (args.logging_steps * 5) == 0:
            self._update_plots()
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each epoch"""
        current_time = time.time()
        epoch_info = {
            "event": "epoch_end",
            "epoch": state.epoch,
            "step": state.global_step,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": current_time - self.start_time,
        }
        
        self._log_event(epoch_info)
        
        # Progress log
        with open(self.progress_file, 'a') as f:
            f.write(f"Epoch {state.epoch} completed at step {state.global_step}\n")
            
        # Update plots
        self._update_plots()
        
        logger.info(f"Epoch {state.epoch} completed")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training"""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        end_info = {
            "event": "training_end",
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "total_steps": state.global_step,
            "final_epoch": state.epoch,
        }
        
        self._log_event(end_info)
        
        # Progress log
        with open(self.progress_file, 'a') as f:
            f.write(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total time: {total_time/3600:.2f} hours\n")
            f.write(f"Total steps: {state.global_step}\n")
            f.write(f"Final epoch: {state.epoch}\n")
        
        # Generate final plots and summary
        self._generate_final_plots()
        self._generate_training_summary()
        
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
    
    def _log_event(self, event: Dict[str, Any]):
        """Log event to JSONL file"""
        with open(self.train_log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def _log_training_progress(self, log_entry: Dict[str, Any]):
        """Log training progress to console and file"""
        step = log_entry.get("step", 0)
        loss = log_entry.get("loss", 0)
        lr = log_entry.get("learning_rate", 0)
        epoch = log_entry.get("epoch", 0)
        elapsed = log_entry.get("elapsed_time", 0)
        
        # Console output
        logger.info(f"Step {step:>6} | Epoch {epoch:>6.2f} | Loss: {loss:>8.4f} | LR: {lr:>10.2e} | Time: {elapsed/60:>6.1f}m")
        
        # Progress file
        with open(self.progress_file, 'a') as f:
            f.write(f"Step {step:>6} | Epoch {epoch:>6.2f} | Loss: {loss:>8.4f} | LR: {lr:>10.2e} | Time: {elapsed/60:>6.1f}m\n")
    
    def _log_evaluation_progress(self, log_entry: Dict[str, Any]):
        """Log evaluation progress"""
        step = log_entry.get("step", 0)
        eval_loss = log_entry.get("eval_loss", 0)
        epoch = log_entry.get("epoch", 0)
        
        # Console output
        logger.info(f"EVAL  {step:>6} | Epoch {epoch:>6.2f} | Eval Loss: {eval_loss:>8.4f}")
        
        # Progress file  
        with open(self.progress_file, 'a') as f:
            f.write(f"EVAL  {step:>6} | Epoch {epoch:>6.2f} | Eval Loss: {eval_loss:>8.4f}\n")
    
    def _update_plots(self):
        """Update training plots"""
        try:
            if len(self.training_logs) < 2:
                return
            
            # Extract data for plotting
            steps = [log["step"] for log in self.training_logs]
            losses = [log.get("loss", 0) for log in self.training_logs]
            learning_rates = [log.get("learning_rate", 0) for log in self.training_logs]
            
            # Create plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Training loss
            ax1.plot(steps, losses, 'b-', alpha=0.7)
            ax1.set_title("Training Loss")
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Loss")
            ax1.grid(True, alpha=0.3)
            
            # Learning rate
            ax2.plot(steps, learning_rates, 'r-', alpha=0.7)
            ax2.set_title("Learning Rate")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Learning Rate")
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            # Evaluation loss (if available)
            if self.eval_logs:
                eval_steps = [log["step"] for log in self.eval_logs]
                eval_losses = [log.get("eval_loss", 0) for log in self.eval_logs]
                ax3.plot(eval_steps, eval_losses, 'g-', alpha=0.7, marker='o')
                ax3.set_title("Evaluation Loss")
                ax3.set_xlabel("Step")
                ax3.set_ylabel("Eval Loss")
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, "No evaluation data", ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title("Evaluation Loss")
            
            # Training speed
            if len(self.training_logs) > 1:
                times = [log["elapsed_time"] for log in self.training_logs]
                speed = [(steps[i] - steps[i-1]) / (times[i] - times[i-1]) 
                        for i in range(1, len(steps)) if times[i] > times[i-1]]
                speed_steps = steps[1:len(speed)+1]
                ax4.plot(speed_steps, speed, 'purple', alpha=0.7)
                ax4.set_title("Training Speed (steps/sec)")
                ax4.set_xlabel("Step") 
                ax4.set_ylabel("Steps/sec")
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title("Training Speed")
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, "training_progress.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to update plots: {e}")
    
    def _generate_final_plots(self):
        """Generate comprehensive final plots"""
        try:
            if not self.training_logs:
                return
            
            # Create comprehensive final plot
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # Extract all data
            steps = [log["step"] for log in self.training_logs]
            losses = [log.get("loss", 0) for log in self.training_logs]
            learning_rates = [log.get("learning_rate", 0) for log in self.training_logs]
            epochs = [log.get("epoch", 0) for log in self.training_logs]
            times = [log["elapsed_time"] for log in self.training_logs]
            
            # Plot 1: Training Loss
            axes[0, 0].plot(steps, losses, 'b-', alpha=0.8, linewidth=2)
            axes[0, 0].set_title("Training Loss", fontsize=14)
            axes[0, 0].set_xlabel("Step")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Learning Rate Schedule
            axes[0, 1].plot(steps, learning_rates, 'r-', alpha=0.8, linewidth=2)
            axes[0, 1].set_title("Learning Rate Schedule", fontsize=14)
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Learning Rate")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
            
            # Plot 3: Loss vs Epoch
            axes[0, 2].plot(epochs, losses, 'g-', alpha=0.8, linewidth=2)
            axes[0, 2].set_title("Loss vs Epoch", fontsize=14)
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("Loss")
            axes[0, 2].grid(True, alpha=0.3)
            
            # Plot 4: Training Speed
            if len(times) > 1:
                speed = [(steps[i] - steps[i-1]) / (times[i] - times[i-1]) 
                        for i in range(1, len(steps)) if times[i] > times[i-1]]
                speed_steps = steps[1:len(speed)+1]
                axes[1, 0].plot(speed_steps, speed, 'purple', alpha=0.8, linewidth=2)
                axes[1, 0].set_title("Training Speed", fontsize=14)
                axes[1, 0].set_xlabel("Step")
                axes[1, 0].set_ylabel("Steps/sec")
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Evaluation metrics (if available)
            if self.eval_logs:
                eval_steps = [log["step"] for log in self.eval_logs]
                eval_losses = [log.get("eval_loss", 0) for log in self.eval_logs]
                axes[1, 1].plot(eval_steps, eval_losses, 'orange', alpha=0.8, linewidth=2, marker='o')
                axes[1, 1].set_title("Evaluation Loss", fontsize=14)
                axes[1, 1].set_xlabel("Step")
                axes[1, 1].set_ylabel("Eval Loss")
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, "No evaluation data", ha='center', va='center', 
                              transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title("Evaluation Loss", fontsize=14)
            
            # Plot 6: Loss smoothed (moving average)
            if len(losses) > 10:
                window = min(50, len(losses) // 10)
                smoothed = [sum(losses[max(0, i-window):i+1]) / min(i+1, window) 
                          for i in range(len(losses))]
                axes[1, 2].plot(steps, losses, 'b-', alpha=0.3, label='Raw')
                axes[1, 2].plot(steps, smoothed, 'b-', alpha=0.8, linewidth=2, label='Smoothed')
                axes[1, 2].set_title("Loss (Smoothed)", fontsize=14)
                axes[1, 2].set_xlabel("Step")
                axes[1, 2].set_ylabel("Loss")
                axes[1, 2].grid(True, alpha=0.3)
                axes[1, 2].legend()
            
            plt.tight_layout()
            
            # Save final plot
            final_plot_path = os.path.join(self.plots_dir, "final_training_summary.png")
            plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Final plots saved to: {final_plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate final plots: {e}")
    
    def _generate_training_summary(self):
        """Generate training summary report"""
        try:
            summary = {
                "training_completed": True,
                "timestamp": datetime.now().isoformat(),
                "total_training_logs": len(self.training_logs),
                "total_eval_logs": len(self.eval_logs),
            }
            
            if self.training_logs:
                final_log = self.training_logs[-1]
                initial_log = self.training_logs[0]
                
                summary.update({
                    "final_step": final_log.get("step", 0),
                    "final_epoch": final_log.get("epoch", 0),
                    "final_loss": final_log.get("loss", 0),
                    "initial_loss": initial_log.get("loss", 0),
                    "loss_improvement": initial_log.get("loss", 0) - final_log.get("loss", 0),
                    "total_training_time": final_log.get("elapsed_time", 0),
                })
                
                # Loss statistics
                losses = [log.get("loss", 0) for log in self.training_logs]
                summary["loss_statistics"] = {
                    "min": min(losses),
                    "max": max(losses),
                    "mean": sum(losses) / len(losses),
                    "final": losses[-1]
                }
            
            if self.eval_logs:
                final_eval = self.eval_logs[-1]
                initial_eval = self.eval_logs[0]
                
                summary.update({
                    "final_eval_loss": final_eval.get("eval_loss", 0),
                    "initial_eval_loss": initial_eval.get("eval_loss", 0),
                    "eval_loss_improvement": initial_eval.get("eval_loss", 0) - final_eval.get("eval_loss", 0),
                })
            
            # Save summary
            with open(self.metrics_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Training summary saved to: {self.metrics_file}")
            
        except Exception as e:
            logger.warning(f"Failed to generate training summary: {e}")
    
    def save_final_metrics(self, train_result):
        """Save final training metrics"""
        final_metrics = {
            "training_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
            "total_flos": train_result.metrics.get("total_flos", 0),
        }
        
        # Save to file
        final_metrics_file = os.path.join(self.logs_dir, "final_metrics.json")
        with open(final_metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(f"Final metrics saved to: {final_metrics_file}")