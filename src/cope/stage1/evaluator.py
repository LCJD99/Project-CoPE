"""
Evaluation utilities and callbacks for training
"""

import json
import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from collections import defaultdict


class EvaluationCallback(TrainerCallback):
    """
    Custom callback for evaluation during training.
    Monitors loss, perplexity, and token prediction accuracy.
    """
    
    def __init__(self, tokenizer, log_interval: int = 100):
        """
        Initialize callback.
        
        Args:
            tokenizer: Tokenizer instance
            log_interval: Logging interval in steps
        """
        self.tokenizer = tokenizer
        self.log_interval = log_interval
        self.metrics_history = defaultdict(list)
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Called when logging occurs.
        
        Computes and logs additional metrics:
        - Perplexity
        - Loss trends
        """
        if logs is None:
            return
        
        # Calculate perplexity from loss
        if 'loss' in logs:
            perplexity = np.exp(logs['loss'])
            logs['perplexity'] = perplexity
            self.metrics_history['perplexity'].append(perplexity)
            self.metrics_history['loss'].append(logs['loss'])
        
        # Log learning rate
        if 'learning_rate' in logs:
            self.metrics_history['learning_rate'].append(logs['learning_rate'])
        
        # Print summary every log_interval steps
        if state.global_step % self.log_interval == 0:
            self._print_metrics(state.global_step, logs)
    
    def _print_metrics(self, step: int, logs: Dict[str, float]):
        """Print formatted metrics"""
        print(f"\n--- Step {step} ---")
        
        if 'loss' in logs:
            print(f"Loss: {logs['loss']:.4f}")
        
        if 'perplexity' in logs:
            print(f"Perplexity: {logs['perplexity']:.2f}")
        
        if 'learning_rate' in logs:
            print(f"Learning Rate: {logs['learning_rate']:.2e}")
        
        if 'grad_norm' in logs:
            print(f"Gradient Norm: {logs['grad_norm']:.4f}")
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Called at the end of training.
        Save metrics history.
        """
        metrics_file = f"{args.output_dir}/metrics_history.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(dict(self.metrics_history), f, indent=2)
        
        print(f"\nMetrics history saved to {metrics_file}")
        
        # Print final summary
        if self.metrics_history['loss']:
            final_loss = self.metrics_history['loss'][-1]
            final_perplexity = self.metrics_history['perplexity'][-1]
            
            print("\n" + "=" * 60)
            print("Training Summary")
            print("=" * 60)
            print(f"Final Loss: {final_loss:.4f}")
            print(f"Final Perplexity: {final_perplexity:.2f}")
            
            if len(self.metrics_history['loss']) > 1:
                initial_loss = self.metrics_history['loss'][0]
                improvement = ((initial_loss - final_loss) / initial_loss) * 100
                print(f"Loss Improvement: {improvement:.2f}%")
            
            print("=" * 60)


def compute_token_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    virtual_token_ids: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute token prediction accuracy.
    
    Args:
        predictions: Model predictions [batch_size, seq_len, vocab_size]
        labels: True labels [batch_size, seq_len]
        tokenizer: Tokenizer instance
        virtual_token_ids: List of virtual token IDs to focus on
    
    Returns:
        Dictionary with accuracy metrics
    """
    # Get predicted token IDs
    pred_ids = torch.argmax(predictions, dim=-1)
    
    # Mask out padding and ignored tokens (-100)
    mask = labels != -100
    
    # Overall accuracy
    correct = (pred_ids == labels) & mask
    total_tokens = mask.sum().item()
    correct_tokens = correct.sum().item()
    
    overall_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    metrics = {
        'overall_accuracy': overall_accuracy,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens
    }
    
    # Virtual token specific accuracy
    if virtual_token_ids is not None:
        virtual_mask = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in virtual_token_ids:
            virtual_mask |= (labels == token_id)
        
        virtual_mask = virtual_mask & mask
        virtual_correct = correct & virtual_mask
        
        total_virtual = virtual_mask.sum().item()
        correct_virtual = virtual_correct.sum().item()
        
        virtual_accuracy = correct_virtual / total_virtual if total_virtual > 0 else 0.0
        
        metrics.update({
            'virtual_token_accuracy': virtual_accuracy,
            'total_virtual_tokens': total_virtual,
            'correct_virtual_tokens': correct_virtual
        })
    
    return metrics


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss value
    
    Returns:
        Perplexity value
    """
    return np.exp(loss)


class MetricsTracker:
    """Track and aggregate metrics during training"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values"""
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
    
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric"""
        values = self.metrics[metric_name]
        
        if not values:
            return 0.0
        
        if last_n is not None:
            values = values[-last_n:]
        
        return np.mean(values)
    
    def get_all_averages(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get averages of all metrics"""
        return {
            name: self.get_average(name, last_n)
            for name in self.metrics.keys()
        }
    
    def save(self, filepath: str):
        """Save metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
    
    def load(self, filepath: str):
        """Load metrics from file"""
        with open(filepath, 'r') as f:
            loaded = json.load(f)
            self.metrics = defaultdict(list, loaded)
