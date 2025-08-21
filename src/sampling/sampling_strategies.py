"""
Sampling strategies for imbalanced fraud detection dataset.

This module implements various sampling techniques to address class imbalance,
including oversampling, undersampling, and hybrid approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils import check_random_state

# Sampling techniques
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

import logging

logger = logging.getLogger(__name__)


class SamplingStrategyFactory:
    """Factory class for creating different sampling strategies."""
    
    @staticmethod
    def create_sampler(strategy: str, random_state: int = 42, **kwargs) -> BaseEstimator:
        """
        Create a sampling strategy instance.
        
        Args:
            strategy: Name of the sampling strategy
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for the sampler
            
        Returns:
            Configured sampler instance
            
        Raises:
            ValueError: If strategy is not supported
        """
        samplers = {
            # Oversampling techniques
            'smote': SMOTE(random_state=random_state, **kwargs),
            'adasyn': ADASYN(random_state=random_state, **kwargs),
            'borderline_smote': BorderlineSMOTE(random_state=random_state, **kwargs),
            'svm_smote': SVMSMOTE(random_state=random_state, **kwargs),
            
            # Undersampling techniques
            'random_under': RandomUnderSampler(random_state=random_state, **kwargs),
            'edited_nn': EditedNearestNeighbours(**kwargs),
            'tomek_links': TomekLinks(**kwargs),
            
            # Hybrid techniques
            'smote_enn': SMOTEENN(random_state=random_state, **kwargs),
            'smote_tomek': SMOTETomek(random_state=random_state, **kwargs),
            
            # No sampling (baseline)
            'none': None
        }
        
        if strategy not in samplers:
            raise ValueError(f"Unsupported sampling strategy: {strategy}. "
                           f"Available strategies: {list(samplers.keys())}")
        
        logger.info(f"Created {strategy} sampler with parameters: {kwargs}")
        return samplers[strategy]


class AdaptiveSamplingRatio:
    """Utility class for calculating adaptive sampling ratios based on dataset characteristics."""
    
    @staticmethod
    def calculate_optimal_ratio(y: np.ndarray, strategy: str = 'balanced') -> Dict[int, int]:
        """
        Calculate optimal sampling ratio based on class distribution.
        
        Args:
            y: Target variable
            strategy: Sampling strategy ('balanced', 'moderate', 'conservative')
            
        Returns:
            Dictionary with class ratios
        """
        from collections import Counter
        
        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)
        
        minority_count = class_counts[minority_class]
        majority_count = class_counts[majority_class]
        
        if strategy == 'balanced':
            # Perfect balance
            target_minority = majority_count
        elif strategy == 'moderate':
            # 1:3 ratio (minority:majority)
            target_minority = majority_count // 3
        elif strategy == 'conservative':
            # 1:5 ratio (minority:majority)
            target_minority = majority_count // 5
        else:
            target_minority = minority_count
        
        return {
            minority_class: target_minority,
            majority_class: majority_count
        }


class SamplingPipelineBuilder:
    """Builder class for creating sampling pipelines that prevent data leakage."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessing_steps = []
        self.sampling_step = None
        self.model_step = None
    
    def add_preprocessing(self, name: str, transformer: BaseEstimator) -> 'SamplingPipelineBuilder':
        """Add preprocessing step to pipeline."""
        self.preprocessing_steps.append((name, transformer))
        return self
    
    def add_sampling(self, strategy: str, **kwargs) -> 'SamplingPipelineBuilder':
        """Add sampling step to pipeline."""
        sampler = SamplingStrategyFactory.create_sampler(
            strategy, random_state=self.random_state, **kwargs
        )
        if sampler is not None:
            self.sampling_step = ('sampling', sampler)
        return self
    
    def add_model(self, name: str, model: BaseEstimator) -> 'SamplingPipelineBuilder':
        """Add model step to pipeline."""
        self.model_step = (name, model)
        return self
    
    def build(self) -> ImbPipeline:
        """Build the complete pipeline."""
        steps = []
        
        # Add preprocessing steps
        steps.extend(self.preprocessing_steps)
        
        # Add sampling step (if specified)
        if self.sampling_step is not None:
            steps.append(self.sampling_step)
        
        # Add model step
        if self.model_step is not None:
            steps.append(self.model_step)
        
        logger.info(f"Built pipeline with {len(steps)} steps")
        return ImbPipeline(steps)


class SamplingExperimentTracker:
    """Tracks sampling experiments and their effects on class distribution."""
    
    def __init__(self):
        self.experiments = []
    
    def track_experiment(self, 
                        strategy: str,
                        original_distribution: Dict[int, int],
                        sampled_distribution: Dict[int, int],
                        performance_metrics: Dict[str, float],
                        parameters: Dict[str, Any]) -> None:
        """Track a sampling experiment."""
        
        # Calculate sampling statistics
        original_total = sum(original_distribution.values())
        sampled_total = sum(sampled_distribution.values())
        
        minority_class = min(original_distribution, key=original_distribution.get)
        majority_class = max(original_distribution, key=original_distribution.get)
        
        original_ratio = original_distribution[minority_class] / original_distribution[majority_class]
        sampled_ratio = sampled_distribution[minority_class] / sampled_distribution[majority_class]
        
        experiment = {
            'strategy': strategy,
            'parameters': parameters,
            'original_distribution': original_distribution,
            'sampled_distribution': sampled_distribution,
            'original_total_samples': original_total,
            'sampled_total_samples': sampled_total,
            'original_minority_ratio': original_ratio,
            'sampled_minority_ratio': sampled_ratio,
            'sampling_effect': sampled_total / original_total,
            'ratio_improvement': sampled_ratio / original_ratio,
            'performance_metrics': performance_metrics
        }
        
        self.experiments.append(experiment)
        
        logger.info(f"Tracked experiment: {strategy} - "
                   f"Ratio changed from {original_ratio:.4f} to {sampled_ratio:.4f}")
    
    def get_best_strategy(self, metric: str = 'f1_score') -> Dict[str, Any]:
        """Get the best performing sampling strategy."""
        if not self.experiments:
            raise ValueError("No experiments tracked yet")
        
        best_experiment = max(self.experiments, 
                            key=lambda x: x['performance_metrics'].get(metric, 0))
        
        return best_experiment
    
    def get_experiment_summary(self) -> pd.DataFrame:
        """Get summary of all experiments as DataFrame."""
        if not self.experiments:
            return pd.DataFrame()
        
        summary_data = []
        for exp in self.experiments:
            row = {
                'strategy': exp['strategy'],
                'original_ratio': exp['original_minority_ratio'],
                'sampled_ratio': exp['sampled_minority_ratio'],
                'ratio_improvement': exp['ratio_improvement'],
                'sampling_effect': exp['sampling_effect'],
                'sampled_total': exp['sampled_total_samples']
            }
            
            # Add performance metrics
            row.update(exp['performance_metrics'])
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


# Sampling strategy configurations for experimentation
SAMPLING_CONFIGURATIONS = {
    'smote_default': {
        'strategy': 'smote',
        'k_neighbors': 5,
        'sampling_strategy': 'auto'
    },
    'smote_conservative': {
        'strategy': 'smote',
        'k_neighbors': 3,
        'sampling_strategy': 0.5  # 1:2 ratio
    },
    'smote_balanced': {
        'strategy': 'smote',
        'k_neighbors': 5,
        'sampling_strategy': 'auto'  # 1:1 ratio
    },
    'adasyn_default': {
        'strategy': 'adasyn',
        'n_neighbors': 5,
        'sampling_strategy': 'auto'
    },
    'adasyn_conservative': {
        'strategy': 'adasyn',
        'n_neighbors': 3,
        'sampling_strategy': 0.3  # 1:3 ratio
    },
    'borderline_smote': {
        'strategy': 'borderline_smote',
        'k_neighbors': 5,
        'kind': 'borderline-1'
    },
    'random_under_balanced': {
        'strategy': 'random_under',
        'sampling_strategy': 'auto'
    },
    'random_under_conservative': {
        'strategy': 'random_under',
        'sampling_strategy': 0.5  # Keep more majority samples
    },
    'smote_tomek': {
        'strategy': 'smote_tomek',
        'sampling_strategy': 'auto'
    },
    'smote_enn': {
        'strategy': 'smote_enn',
        'sampling_strategy': 'auto'
    },
    'no_sampling': {
        'strategy': 'none'
    }
}


def create_sampling_pipeline(preprocessing_pipeline, 
                           sampling_config: str,
                           model,
                           random_state: int = 42) -> ImbPipeline:
    """
    Create a complete pipeline with preprocessing, sampling, and model.
    
    Args:
        preprocessing_pipeline: Sklearn preprocessing pipeline
        sampling_config: Name of sampling configuration
        model: ML model instance
        random_state: Random state for reproducibility
        
    Returns:
        Complete imbalanced-learn pipeline
    """
    if sampling_config not in SAMPLING_CONFIGURATIONS:
        raise ValueError(f"Unknown sampling configuration: {sampling_config}")
    
    config = SAMPLING_CONFIGURATIONS[sampling_config].copy()
    strategy = config.pop('strategy')
    
    builder = SamplingPipelineBuilder(random_state=random_state)
    
    # Add preprocessing
    builder.add_preprocessing('preprocessing', preprocessing_pipeline)
    
    # Add sampling
    builder.add_sampling(strategy, **config)
    
    # Add model
    builder.add_model('classifier', model)
    
    return builder.build()


if __name__ == "__main__":
    # Example usage and testing
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=10000, n_features=20, n_informative=15,
        n_redundant=5, n_clusters_per_class=1, weights=[0.95, 0.05],
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create preprocessing pipeline
    preprocessing = StandardScaler()
    
    # Create model
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Create sampling pipeline
    pipeline = create_sampling_pipeline(
        preprocessing_pipeline=preprocessing,
        sampling_config='smote_balanced',
        model=model
    )
    
    # Train and evaluate
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Track experiment
    tracker = SamplingExperimentTracker()
    from collections import Counter
    
    original_dist = Counter(y_train)
    # For demonstration, we'll use the same distribution
    # In practice, you'd get this from the sampled data
    sampled_dist = {0: 4000, 1: 4000}  # Example balanced distribution
    
    tracker.track_experiment(
        strategy='smote_balanced',
        original_distribution=original_dist,
        sampled_distribution=sampled_dist,
        performance_metrics={'f1_score': 0.85, 'precision': 0.82, 'recall': 0.88},
        parameters={'k_neighbors': 5, 'sampling_strategy': 'auto'}
    )
    
    print("\nExperiment Summary:")
    print(tracker.get_experiment_summary())
