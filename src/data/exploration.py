"""
Aegis Fraud Detector - Exploratory Data Analysis Module

This module contains reusable functions for comprehensive exploratory data analysis
of the IEEE-CIS Fraud Detection dataset. All analyses performed in notebooks
must use functions from this module to ensure reproducibility and version control.

Author: Aegis Development Team
Date: August 2025
Sprint: 1.1 - Rigorous EDA Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FraudDataExplorer:
    """
    Comprehensive fraud detection dataset explorer with rigorous statistical analysis.
    
    This class provides methods for:
    - Dataset overview and basic statistics
    - Missing value analysis
    - Target variable investigation
    - Feature distribution analysis
    - Correlation analysis
    - Temporal pattern detection
    - Anomaly detection
    """
    
    def __init__(self, data_path: str = "data/01_raw"):
        """
        Initialize the fraud data explorer.
        
        Args:
            data_path: Path to the raw data directory
        """
        self.data_path = Path(data_path)
        self.train_transaction = None
        self.train_identity = None
        self.test_transaction = None
        self.test_identity = None
        self.merged_train = None
        
        # EDA findings storage
        self.findings = {
            'dataset_overview': {},
            'missing_values': {},
            'target_analysis': {},
            'feature_distributions': {},
            'correlations': {},
            'temporal_patterns': {},
            'anomalies': {}
        }
    
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all IEEE-CIS fraud detection datasets.
        
        Returns:
            Dictionary containing all loaded datasets
        """
        logger.info("Loading IEEE-CIS Fraud Detection datasets...")
        
        try:
            # Load transaction datasets
            self.train_transaction = pd.read_csv(self.data_path / "train_transaction.csv")
            self.train_identity = pd.read_csv(self.data_path / "train_identity.csv")
            self.test_transaction = pd.read_csv(self.data_path / "test_transaction.csv")
            self.test_identity = pd.read_csv(self.data_path / "test_identity.csv")
            
            # Merge training datasets
            self.merged_train = self.train_transaction.merge(
                self.train_identity, 
                on='TransactionID', 
                how='left'
            )
            
            datasets = {
                'train_transaction': self.train_transaction,
                'train_identity': self.train_identity,
                'test_transaction': self.test_transaction,
                'test_identity': self.test_identity,
                'merged_train': self.merged_train
            }
            
            logger.info("✅ All datasets loaded successfully")
            return datasets
            
        except Exception as e:
            logger.error(f"❌ Error loading datasets: {str(e)}")
            raise
    
    def analyze_dataset_overview(self) -> Dict[str, Any]:
        """
        Perform comprehensive dataset overview analysis.
        
        Returns:
            Dictionary containing dataset overview findings
        """
        logger.info("🔍 Analyzing dataset overview...")
        
        if self.merged_train is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")
        
        overview = {}
        
        # Basic dataset information
        overview['shape'] = {
            'train_transaction': self.train_transaction.shape,
            'train_identity': self.train_identity.shape,
            'test_transaction': self.test_transaction.shape,
            'test_identity': self.test_identity.shape,
            'merged_train': self.merged_train.shape
        }
        
        # Data types analysis
        overview['dtypes'] = {
            'transaction_dtypes': self.train_transaction.dtypes.value_counts().to_dict(),
            'identity_dtypes': self.train_identity.dtypes.value_counts().to_dict(),
            'merged_dtypes': self.merged_train.dtypes.value_counts().to_dict()
        }
        
        # Memory usage
        overview['memory_usage'] = {
            'train_transaction_mb': self.train_transaction.memory_usage(deep=True).sum() / 1024**2,
            'train_identity_mb': self.train_identity.memory_usage(deep=True).sum() / 1024**2,
            'merged_train_mb': self.merged_train.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Feature categories
        transaction_features = self.train_transaction.columns.tolist()
        identity_features = self.train_identity.columns.tolist()
        
        # Categorize transaction features
        feature_categories = {
            'target': ['isFraud'],
            'transaction_id': ['TransactionID'],
            'transaction_amount': ['TransactionAmt'],
            'transaction_datetime': ['TransactionDT'],
            'card_features': [col for col in transaction_features if 'card' in col.lower()],
            'address_features': [col for col in transaction_features if 'addr' in col.lower()],
            'distance_features': [col for col in transaction_features if 'dist' in col.lower()],
            'email_features': [col for col in transaction_features if 'email' in col.lower()],
            'categorical_features': [col for col in transaction_features if col.startswith(('P_', 'M_', 'R_'))],
            'continuous_features': [col for col in transaction_features if col.startswith(('C', 'D', 'V'))],
            'identity_features': identity_features
        }
        
        overview['feature_categories'] = feature_categories
        overview['feature_counts'] = {k: len(v) for k, v in feature_categories.items()}
        
        self.findings['dataset_overview'] = overview
        
        logger.info("✅ Dataset overview analysis completed")
        return overview
    
    def analyze_missing_values(self) -> Dict[str, Any]:
        """
        Comprehensive missing value analysis across all datasets.
        
        Returns:
            Dictionary containing missing value analysis results
        """
        logger.info("🔍 Analyzing missing values patterns...")
        
        missing_analysis = {}
        
        datasets = {
            'train_transaction': self.train_transaction,
            'train_identity': self.train_identity,
            'merged_train': self.merged_train
        }
        
        for name, df in datasets.items():
            # Calculate missing value statistics
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100
            
            # Create missing value summary
            missing_summary = pd.DataFrame({
                'column': missing_counts.index,
                'missing_count': missing_counts.values,
                'missing_percentage': missing_percentages.values,
                'dtype': df.dtypes.values
            }).sort_values('missing_percentage', ascending=False)
            
            # Categorize missing value patterns
            missing_analysis[name] = {
                'total_features': len(df.columns),
                'features_with_missing': (missing_counts > 0).sum(),
                'complete_features': (missing_counts == 0).sum(),
                'heavily_missing_features': (missing_percentages > 90).sum(),
                'moderately_missing_features': ((missing_percentages > 50) & (missing_percentages <= 90)).sum(),
                'lightly_missing_features': ((missing_percentages > 0) & (missing_percentages <= 50)).sum(),
                'missing_summary': missing_summary,
                'missing_value_heatmap_data': df.isnull().astype(int)
            }
        
        self.findings['missing_values'] = missing_analysis
        
        logger.info("✅ Missing value analysis completed")
        return missing_analysis
    
    def analyze_target_variable(self) -> Dict[str, Any]:
        """
        Deep analysis of the fraud target variable including temporal patterns.
        
        Returns:
            Dictionary containing comprehensive target variable analysis
        """
        logger.info("🎯 Analyzing target variable (isFraud)...")
        
        if 'isFraud' not in self.merged_train.columns:
            raise ValueError("Target variable 'isFraud' not found in dataset")
        
        target_analysis = {}
        
        # Basic target distribution
        fraud_counts = self.merged_train['isFraud'].value_counts()
        fraud_percentages = self.merged_train['isFraud'].value_counts(normalize=True) * 100
        
        target_analysis['basic_distribution'] = {
            'total_transactions': len(self.merged_train),
            'fraud_transactions': fraud_counts.get(1, 0),
            'legitimate_transactions': fraud_counts.get(0, 0),
            'fraud_rate': fraud_percentages.get(1, 0),
            'class_imbalance_ratio': fraud_counts.get(0, 0) / fraud_counts.get(1, 1)  # Avoid division by zero
        }
        
        # Temporal analysis
        if 'TransactionDT' in self.merged_train.columns:
            # Convert TransactionDT to more interpretable time features
            # TransactionDT appears to be seconds from a reference point
            self.merged_train['hour'] = (self.merged_train['TransactionDT'] / 3600) % 24
            self.merged_train['day'] = (self.merged_train['TransactionDT'] / (3600 * 24)) % 7
            self.merged_train['week'] = (self.merged_train['TransactionDT'] / (3600 * 24 * 7))
            
            # Hourly fraud patterns
            hourly_fraud = self.merged_train.groupby('hour')['isFraud'].agg(['count', 'sum', 'mean']).reset_index()
            hourly_fraud.columns = ['hour', 'total_transactions', 'fraud_count', 'fraud_rate']
            
            # Daily fraud patterns
            daily_fraud = self.merged_train.groupby('day')['isFraud'].agg(['count', 'sum', 'mean']).reset_index()
            daily_fraud.columns = ['day', 'total_transactions', 'fraud_count', 'fraud_rate']
            
            # Weekly fraud trends
            weekly_fraud = self.merged_train.groupby('week')['isFraud'].agg(['count', 'sum', 'mean']).reset_index()
            weekly_fraud.columns = ['week', 'total_transactions', 'fraud_count', 'fraud_rate']
            
            target_analysis['temporal_patterns'] = {
                'hourly_patterns': hourly_fraud,
                'daily_patterns': daily_fraud,
                'weekly_trends': weekly_fraud,
                'peak_fraud_hour': hourly_fraud.loc[hourly_fraud['fraud_rate'].idxmax(), 'hour'],
                'peak_fraud_day': daily_fraud.loc[daily_fraud['fraud_rate'].idxmax(), 'day'],
                'fraud_rate_variance_hourly': hourly_fraud['fraud_rate'].var(),
                'fraud_rate_variance_daily': daily_fraud['fraud_rate'].var()
            }
        
        # Fraud amount analysis
        if 'TransactionAmt' in self.merged_train.columns:
            fraud_amounts = self.merged_train[self.merged_train['isFraud'] == 1]['TransactionAmt']
            legit_amounts = self.merged_train[self.merged_train['isFraud'] == 0]['TransactionAmt']
            
            target_analysis['amount_patterns'] = {
                'fraud_amount_stats': fraud_amounts.describe().to_dict(),
                'legit_amount_stats': legit_amounts.describe().to_dict(),
                'median_fraud_amount': fraud_amounts.median(),
                'median_legit_amount': legit_amounts.median(),
                'fraud_amount_skewness': fraud_amounts.skew(),
                'legit_amount_skewness': legit_amounts.skew(),
                'amount_correlation_with_fraud': self.merged_train[['TransactionAmt', 'isFraud']].corr().iloc[0, 1]
            }
        
        self.findings['target_analysis'] = target_analysis
        
        logger.info("✅ Target variable analysis completed")
        return target_analysis
    
    def analyze_feature_distributions(self, sample_size: int = 10000) -> Dict[str, Any]:
        """
        Analyze distributions of key features with statistical tests.
        
        Args:
            sample_size: Number of samples to use for analysis (for performance)
            
        Returns:
            Dictionary containing feature distribution analysis
        """
        logger.info("📊 Analyzing feature distributions...")
        
        # Sample data for performance
        if len(self.merged_train) > sample_size:
            sample_df = self.merged_train.sample(n=sample_size, random_state=42)
        else:
            sample_df = self.merged_train.copy()
        
        distribution_analysis = {}
        
        # Numerical features analysis
        numerical_features = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [col for col in numerical_features if col not in ['TransactionID', 'isFraud']]
        
        for feature in numerical_features[:20]:  # Analyze top 20 numerical features
            if sample_df[feature].notna().sum() > 100:  # Only analyze features with sufficient data
                feature_stats = {
                    'mean': sample_df[feature].mean(),
                    'median': sample_df[feature].median(),
                    'std': sample_df[feature].std(),
                    'skewness': sample_df[feature].skew(),
                    'kurtosis': sample_df[feature].kurtosis(),
                    'unique_values': sample_df[feature].nunique(),
                    'missing_percentage': (sample_df[feature].isnull().sum() / len(sample_df)) * 100,
                    'outlier_percentage': self._detect_outliers_percentage(sample_df[feature])
                }
                
                # Distribution by fraud status
                if 'isFraud' in sample_df.columns:
                    fraud_dist = sample_df[sample_df['isFraud'] == 1][feature].describe()
                    legit_dist = sample_df[sample_df['isFraud'] == 0][feature].describe()
                    
                    feature_stats['fraud_distribution'] = fraud_dist.to_dict()
                    feature_stats['legit_distribution'] = legit_dist.to_dict()
                
                distribution_analysis[feature] = feature_stats
        
        # Categorical features analysis
        categorical_features = sample_df.select_dtypes(include=['object']).columns.tolist()
        
        categorical_analysis = {}
        for feature in categorical_features[:10]:  # Analyze top 10 categorical features
            if sample_df[feature].notna().sum() > 50:
                cat_stats = {
                    'unique_values': sample_df[feature].nunique(),
                    'most_frequent': sample_df[feature].mode().iloc[0] if len(sample_df[feature].mode()) > 0 else None,
                    'missing_percentage': (sample_df[feature].isnull().sum() / len(sample_df)) * 100,
                    'value_counts': sample_df[feature].value_counts().head(10).to_dict()
                }
                
                categorical_analysis[feature] = cat_stats
        
        distribution_analysis['categorical_analysis'] = categorical_analysis
        
        self.findings['feature_distributions'] = distribution_analysis
        
        logger.info("✅ Feature distribution analysis completed")
        return distribution_analysis
    
    def analyze_correlations(self, method: str = 'pearson') -> Dict[str, Any]:
        """
        Comprehensive correlation analysis between features and with target.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary containing correlation analysis results
        """
        logger.info(f"🔗 Analyzing correlations using {method} method...")
        
        # Select numerical features only
        numerical_df = self.merged_train.select_dtypes(include=[np.number])
        
        # Remove features with too many missing values
        threshold = 0.5  # Remove features with >50% missing values
        numerical_df = numerical_df.loc[:, numerical_df.isnull().mean() < threshold]
        
        correlation_analysis = {}
        
        # Overall correlation matrix
        correlation_matrix = numerical_df.corr(method=method)
        correlation_analysis['correlation_matrix'] = correlation_matrix
        
        # Correlations with target variable
        if 'isFraud' in correlation_matrix.columns:
            target_correlations = correlation_matrix['isFraud'].abs().sort_values(ascending=False)
            target_correlations = target_correlations[target_correlations.index != 'isFraud']
            
            correlation_analysis['target_correlations'] = {
                'top_positive': target_correlations.head(10).to_dict(),
                'top_negative': correlation_matrix['isFraud'].sort_values().head(10).to_dict(),
                'strongest_correlations': target_correlations.head(20).to_dict()
            }
        
        # High correlation pairs (potential multicollinearity)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        correlation_analysis['high_correlations'] = high_corr_pairs
        correlation_analysis['multicollinearity_candidates'] = len(high_corr_pairs)
        
        self.findings['correlations'] = correlation_analysis
        
        logger.info("✅ Correlation analysis completed")
        return correlation_analysis
    
    def _detect_outliers_percentage(self, series: pd.Series) -> float:
        """
        Detect outliers using IQR method and return percentage.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            Percentage of outliers
        """
        if series.notna().sum() < 10:
            return 0.0
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_condition = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
        outlier_percentage = (outlier_condition.sum() / len(series)) * 100
        
        return outlier_percentage
    
    def generate_eda_visualizations(self) -> Dict[str, Any]:
        """
        Generate comprehensive visualizations for EDA findings.
        
        Returns:
            Dictionary containing visualization configurations
        """
        logger.info("📈 Generating EDA visualizations...")
        
        visualizations = {}
        
        # Target distribution visualization
        if 'isFraud' in self.merged_train.columns:
            fraud_counts = self.merged_train['isFraud'].value_counts()
            
            visualizations['target_distribution'] = {
                'type': 'bar',
                'data': fraud_counts.to_dict(),
                'title': 'Target Variable Distribution (Fraud vs Legitimate)',
                'labels': {'0': 'Legitimate', '1': 'Fraud'}
            }
        
        # Transaction amount distribution by fraud status
        if 'TransactionAmt' in self.merged_train.columns:
            visualizations['amount_distribution'] = {
                'type': 'histogram',
                'data': {
                    'fraud': self.merged_train[self.merged_train['isFraud'] == 1]['TransactionAmt'].values,
                    'legitimate': self.merged_train[self.merged_train['isFraud'] == 0]['TransactionAmt'].values
                },
                'title': 'Transaction Amount Distribution by Fraud Status'
            }
        
        # Temporal patterns visualization
        if hasattr(self, 'findings') and 'target_analysis' in self.findings:
            temporal_data = self.findings['target_analysis'].get('temporal_patterns', {})
            if 'hourly_patterns' in temporal_data:
                visualizations['hourly_fraud_pattern'] = {
                    'type': 'line',
                    'data': temporal_data['hourly_patterns'],
                    'title': 'Fraud Rate by Hour of Day'
                }
        
        # Missing value heatmap configuration
        if 'missing_values' in self.findings:
            missing_data = self.findings['missing_values'].get('merged_train', {})
            if 'missing_value_heatmap_data' in missing_data:
                visualizations['missing_value_heatmap'] = {
                    'type': 'heatmap',
                    'data': missing_data['missing_value_heatmap_data'],
                    'title': 'Missing Value Patterns Across Features'
                }
        
        # Correlation heatmap
        if 'correlations' in self.findings:
            corr_matrix = self.findings['correlations'].get('correlation_matrix')
            if corr_matrix is not None:
                visualizations['correlation_heatmap'] = {
                    'type': 'heatmap',
                    'data': corr_matrix,
                    'title': 'Feature Correlation Matrix'
                }
        
        self.findings['visualizations'] = visualizations
        
        logger.info("✅ Visualization configurations generated")
        return visualizations
    
    def generate_eda_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive EDA report with all findings.
        
        Returns:
            Complete EDA report dictionary
        """
        logger.info("📋 Generating comprehensive EDA report...")
        
        report = {
            'metadata': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'dataset_version': 'IEEE-CIS Fraud Detection',
                'analysis_version': 'Sprint 1.1 - Rigorous EDA',
                'total_features_analyzed': len(self.merged_train.columns) if self.merged_train is not None else 0
            },
            'executive_summary': self._generate_executive_summary(),
            'detailed_findings': self.findings,
            'recommendations': self._generate_recommendations()
        }
        
        logger.info("✅ EDA report generated successfully")
        return report
    
    def _generate_executive_summary(self) -> Dict[str, str]:
        """Generate executive summary of key findings."""
        summary = {}
        
        if 'dataset_overview' in self.findings:
            overview = self.findings['dataset_overview']
            summary['dataset_size'] = f"Training set: {overview.get('shape', {}).get('merged_train', 'Unknown')} samples"
        
        if 'target_analysis' in self.findings:
            target = self.findings['target_analysis']
            fraud_rate = target.get('basic_distribution', {}).get('fraud_rate', 0)
            summary['fraud_prevalence'] = f"Fraud rate: {fraud_rate:.2f}% (highly imbalanced dataset)"
        
        if 'missing_values' in self.findings:
            missing = self.findings['missing_values'].get('merged_train', {})
            missing_features = missing.get('features_with_missing', 0)
            total_features = missing.get('total_features', 0)
            summary['data_quality'] = f"Missing values in {missing_features}/{total_features} features"
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = [
            "Implement robust handling of missing values with domain-specific imputation strategies",
            "Address severe class imbalance using sampling techniques or cost-sensitive learning",
            "Investigate temporal patterns for feature engineering opportunities",
            "Consider feature selection based on correlation analysis to reduce dimensionality",
            "Develop outlier detection strategies for transaction amounts",
            "Create engineered features from categorical variables with high fraud signal"
        ]
        
        return recommendations


def create_eda_plotting_functions():
    """
    Create reusable plotting functions for EDA visualizations.
    
    Returns:
        Dictionary of plotting functions
    """
    
    def plot_target_distribution(fraud_counts: pd.Series, save_path: Optional[str] = None):
        """Plot target variable distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        fraud_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
        ax1.set_title('Fraud vs Legitimate Transactions (Count)')
        ax1.set_xlabel('Transaction Type')
        ax1.set_ylabel('Count')
        ax1.set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
        
        # Percentage plot
        fraud_percentages = (fraud_counts / fraud_counts.sum()) * 100
        fraud_percentages.plot(kind='pie', ax=ax2, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
        ax2.set_title('Fraud vs Legitimate Transactions (%)')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_patterns(hourly_data: pd.DataFrame, daily_data: pd.DataFrame, save_path: Optional[str] = None):
        """Plot temporal fraud patterns."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Hourly patterns
        ax1.plot(hourly_data['hour'], hourly_data['fraud_rate'], marker='o', linewidth=2, markersize=6)
        ax1.set_title('Fraud Rate by Hour of Day')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Fraud Rate')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        
        # Daily patterns
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax2.bar(daily_data['day'], daily_data['fraud_rate'], color='coral', alpha=0.7)
        ax2.set_title('Fraud Rate by Day of Week')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Fraud Rate')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(day_labels)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_missing_value_matrix(missing_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot missing value patterns."""
        plt.figure(figsize=(20, 12))
        
        # Sample columns for visualization (too many to show all)
        sample_cols = missing_df.columns[:50] if len(missing_df.columns) > 50 else missing_df.columns
        sample_data = missing_df[sample_cols]
        
        sns.heatmap(sample_data.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title('Missing Value Patterns (Sample of Features)')
        plt.xlabel('Features')
        plt.ylabel('Samples')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, save_path: Optional[str] = None):
        """Plot correlation heatmap for key features."""
        # Select subset of features for visualization
        if 'isFraud' in correlation_matrix.columns:
            target_corrs = correlation_matrix['isFraud'].abs().sort_values(ascending=False)
            top_features = target_corrs.head(20).index.tolist()
            subset_corr = correlation_matrix.loc[top_features, top_features]
        else:
            subset_corr = correlation_matrix.iloc[:20, :20]
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(subset_corr, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, linewidths=0.5, fmt='.2f')
        plt.title('Correlation Heatmap (Top Features)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_amount_distribution(fraud_amounts: pd.Series, legit_amounts: pd.Series, save_path: Optional[str] = None):
        """Plot transaction amount distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Log scale histograms
        ax1.hist(np.log1p(legit_amounts.dropna()), bins=50, alpha=0.7, label='Legitimate', color='skyblue')
        ax1.hist(np.log1p(fraud_amounts.dropna()), bins=50, alpha=0.7, label='Fraud', color='salmon')
        ax1.set_title('Transaction Amount Distribution (Log Scale)')
        ax1.set_xlabel('Log(Transaction Amount + 1)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Box plot comparison
        data_for_box = [legit_amounts.dropna(), fraud_amounts.dropna()]
        ax2.boxplot(data_for_box, labels=['Legitimate', 'Fraud'])
        ax2.set_title('Transaction Amount Box Plot')
        ax2.set_ylabel('Transaction Amount')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    return {
        'plot_target_distribution': plot_target_distribution,
        'plot_temporal_patterns': plot_temporal_patterns,
        'plot_missing_value_matrix': plot_missing_value_matrix,
        'plot_correlation_heatmap': plot_correlation_heatmap,
        'plot_amount_distribution': plot_amount_distribution
    }


# Export main class and functions
__all__ = ['FraudDataExplorer', 'create_eda_plotting_functions']
