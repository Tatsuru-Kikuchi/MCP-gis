#!/usr/bin/env python3
"""
Enhanced Dynamic Analysis Script with Machine Learning Predictions
This script implements the dynamic temporal analysis and ML prediction framework
described in the academic manuscript.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDynamicAnalysis:
    """
    Enhanced dynamic analysis framework with ML predictions
    """
    
    def __init__(self, data_dir="data", results_dir="results", viz_dir="visualizations"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.viz_dir = Path(viz_dir)
        
        # Create directories
        for dir_path in [self.data_dir, self.results_dir, self.viz_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for dynamic analysis
        self.temporal_results_dir = self.results_dir / "temporal"
        self.predictions_dir = self.results_dir / "predictions"
        self.models_dir = self.predictions_dir / "models"
        self.dynamic_viz_dir = self.viz_dir / "dynamic"
        
        for dir_path in [self.temporal_results_dir, self.predictions_dir, 
                        self.models_dir, self.dynamic_viz_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def generate_comprehensive_data(self):
        """
        Generate comprehensive temporal data for dynamic analysis
        """
        logger.info("Generating comprehensive temporal data...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Parameters
        n_wards = 23  # Tokyo wards
        n_industries = 6
        n_years = 24  # 2000-2023
        
        # Ward and industry names
        wards = [f"Ward_{i+1}" for i in range(n_wards)]
        industries = ['IT', 'Finance', 'Professional', 'Manufacturing', 'Retail', 'Healthcare']
        years = list(range(2000, 2024))
        
        # Create comprehensive panel data
        data = []
        
        for ward_idx, ward in enumerate(wards):
            for industry_idx, industry in enumerate(industries):
                for year_idx, year in enumerate(years):
                    
                    # Demographic variables (time-varying)
                    base_population = 50000 + 20000 * ward_idx
                    population_growth = -0.005 if year > 2010 else 0.01  # Decline after 2010
                    population = base_population * (1 + population_growth) ** (year - 2000)
                    
                    # Age structure (aging over time)
                    elderly_share = 0.15 + 0.008 * (year - 2000) + np.random.normal(0, 0.02)
                    young_share = 0.25 - 0.006 * (year - 2000) + np.random.normal(0, 0.02)
                    working_age_share = 1 - elderly_share - young_share
                    
                    # Economic shocks (realistic timing)
                    shock_2008 = -0.15 if year in [2008, 2009] else 0  # Financial crisis
                    shock_2020 = -0.12 if year in [2020, 2021] else 0  # COVID-19
                    shock_2011 = -0.08 if year == 2011 else 0          # Earthquake
                    total_shock = shock_2008 + shock_2020 + shock_2011
                    
                    # AI adoption (accelerating after 2015)
                    if year < 2015:
                        ai_adoption = 0.02 + 0.005 * industry_idx
                    else:
                        ai_growth = 0.15 if industry in ['IT', 'Finance'] else 0.08
                        ai_adoption = 0.02 + 0.005 * industry_idx + ai_growth * (year - 2015)
                    
                    ai_adoption = min(0.8, ai_adoption)  # Cap at 80%
                    
                    # Infrastructure development
                    infrastructure = 0.3 + 0.02 * ward_idx + 0.015 * (year - 2000) + np.random.normal(0, 0.05)
                    infrastructure = max(0, min(1, infrastructure))
                    
                    # Education level (improving over time)
                    education = 0.35 + 0.01 * ward_idx + 0.008 * (year - 2000) + np.random.normal(0, 0.03)
                    education = max(0, min(1, education))
                    
                    # Employment (dependent variable 1)
                    base_employment = 1000 + 500 * industry_idx + 300 * ward_idx
                    employment_trend = 0.02 - 0.003 * (year - 2000)  # Slowing growth
                    ai_employment_effect = 0.1 * ai_adoption if industry in ['IT', 'Finance'] else -0.05 * ai_adoption
                    demographic_effect = -0.1 * elderly_share + 0.08 * young_share
                    
                    employment = base_employment * (
                        1 + employment_trend * (year - 2000) +
                        ai_employment_effect +
                        demographic_effect +
                        total_shock +
                        np.random.normal(0, 0.1)
                    )
                    employment = max(100, employment)
                    
                    # Concentration index (dependent variable 2)
                    base_concentration = 0.4 + 0.1 * industry_idx
                    
                    # Time-varying agglomeration effects
                    demographic_concentration_effect = -0.2 * elderly_share + 0.15 * young_share
                    ai_concentration_effect = 0.3 * ai_adoption if industry in ['IT', 'Finance', 'Professional'] else 0.1 * ai_adoption
                    infrastructure_effect = 0.2 * infrastructure
                    
                    concentration = base_concentration + (
                        demographic_concentration_effect +
                        ai_concentration_effect +
                        infrastructure_effect +
                        total_shock * 0.5 +
                        np.random.normal(0, 0.08)
                    )
                    concentration = max(0.1, min(1.0, concentration))
                    
                    # Productivity (dependent variable 3)
                    base_productivity = 8.0 + 0.3 * industry_idx
                    productivity_trend = 0.015 * (year - 2000)
                    ai_productivity_effect = 0.25 * ai_adoption
                    agglomeration_productivity_effect = 0.4 * concentration
                    education_effect = 0.3 * education
                    
                    productivity = base_productivity + (
                        productivity_trend +
                        ai_productivity_effect +
                        agglomeration_productivity_effect +
                        education_effect +
                        total_shock * 0.3 +
                        np.random.normal(0, 0.12)
                    )
                    
                    # Additional variables for feature engineering
                    distance_to_center = ward_idx * 2.5  # km from center
                    market_potential = sum([employment * np.exp(-0.1 * abs(w - ward_idx)) 
                                          for w in range(n_wards)])
                    
                    data.append({
                        'ward': ward,
                        'industry': industry,
                        'year': year,
                        'ward_id': ward_idx,
                        'industry_id': industry_idx,
                        'population': population,
                        'elderly_share': elderly_share,
                        'young_share': young_share,
                        'working_age_share': working_age_share,
                        'ai_adoption': ai_adoption,
                        'infrastructure': infrastructure,
                        'education': education,
                        'employment': employment,
                        'concentration_index': concentration,
                        'productivity': productivity,
                        'shock_2008': shock_2008,
                        'shock_2020': shock_2020,
                        'shock_2011': shock_2011,
                        'total_shock': total_shock,
                        'distance_to_center': distance_to_center,
                        'market_potential': market_potential / 1000000,  # Scale
                        'time_trend': year - 2000,
                        'time_trend_squared': (year - 2000) ** 2
                    })
        
        self.df = pd.DataFrame(data)
        
        # Save the comprehensive data
        self.df.to_csv(self.data_dir / "dynamic_analysis_data.csv", index=False)
        logger.info(f"Generated comprehensive data with {len(self.df)} observations")
        
        return self.df
    
    def temporal_agglomeration_analysis(self):
        """
        Analyze time-varying agglomeration patterns
        """
        logger.info("Running temporal agglomeration analysis...")
        
        # Calculate rolling window concentration indices
        temporal_results = []
        
        # 5-year rolling windows
        window_size = 5
        
        for industry in self.df['industry'].unique():
            industry_data = self.df[self.df['industry'] == industry]
            
            for start_year in range(2000, 2020):  # Windows: 2000-2004, 2001-2005, ..., 2019-2023
                end_year = start_year + window_size - 1
                window_data = industry_data[
                    (industry_data['year'] >= start_year) & 
                    (industry_data['year'] <= end_year)
                ]
                
                if len(window_data) > 0:
                    # Calculate concentration measures for this window
                    employment_by_ward = window_data.groupby('ward')['employment'].sum()
                    total_employment = employment_by_ward.sum()
                    
                    if total_employment > 0:
                        # Gini coefficient calculation
                        employment_shares = employment_by_ward / total_employment
                        employment_shares = employment_shares.sort_values()
                        n = len(employment_shares)
                        gini = (2 * np.sum((np.arange(1, n+1) * employment_shares)) / 
                               (n * employment_shares.sum()) - (n + 1) / n)
                        
                        # Herfindahl-Hirschman Index
                        hhi = np.sum(employment_shares ** 2)
                        
                        # Average metrics for this window
                        avg_concentration = window_data['concentration_index'].mean()
                        avg_ai_adoption = window_data['ai_adoption'].mean()
                        avg_elderly_share = window_data['elderly_share'].mean()
                        avg_shock = window_data['total_shock'].mean()
                        
                        temporal_results.append({
                            'industry': industry,
                            'window_start': start_year,
                            'window_end': end_year,
                            'window_center': start_year + 2,  # Middle year of window
                            'gini_coefficient': gini,
                            'hhi': hhi,
                            'avg_concentration_index': avg_concentration,
                            'avg_ai_adoption': avg_ai_adoption,
                            'avg_elderly_share': avg_elderly_share,
                            'avg_shock': avg_shock,
                            'total_employment': total_employment
                        })
        
        temporal_df = pd.DataFrame(temporal_results)
        temporal_df.to_csv(self.temporal_results_dir / 'temporal_concentration_indices.csv', index=False)
        
        logger.info("Temporal agglomeration analysis completed")
        return temporal_df
    
    def demographic_transition_analysis(self):
        """
        Analyze demographic transition effects on agglomeration
        """
        logger.info("Analyzing demographic transition effects...")
        
        # Aggregate by year and industry
        demo_effects = []
        
        for year in self.df['year'].unique():
            year_data = self.df[self.df['year'] == year]
            
            for industry in self.df['industry'].unique():
                industry_year_data = year_data[year_data['industry'] == industry]
                
                if len(industry_year_data) > 0:
                    demo_effects.append({
                        'year': year,
                        'industry': industry,
                        'avg_elderly_share': industry_year_data['elderly_share'].mean(),
                        'avg_young_share': industry_year_data['young_share'].mean(),
                        'avg_concentration': industry_year_data['concentration_index'].mean(),
                        'avg_employment': industry_year_data['employment'].mean(),
                        'avg_productivity': industry_year_data['productivity'].mean(),
                        'total_population': industry_year_data['population'].sum(),
                        'concentration_std': industry_year_data['concentration_index'].std()
                    })
        
        demo_df = pd.DataFrame(demo_effects)
        demo_df.to_csv(self.temporal_results_dir / 'demographic_transition_effects.csv', index=False)
        
        logger.info("Demographic transition analysis completed")
        return demo_df
    
    def prepare_ml_features(self):
        """
        Prepare features for machine learning models
        """
        logger.info("Preparing machine learning features...")
        
        # Create lagged variables
        df_ml = self.df.copy()
        df_ml = df_ml.sort_values(['ward', 'industry', 'year'])
        
        # Create lags for key variables
        lag_vars = ['employment', 'concentration_index', 'ai_adoption', 'elderly_share']
        
        for var in lag_vars:
            for lag in [1, 2, 3]:
                df_ml[f'{var}_lag{lag}'] = df_ml.groupby(['ward', 'industry'])[var].shift(lag)
        
        # Moving averages
        for var in ['employment', 'concentration_index']:
            df_ml[f'{var}_ma3'] = df_ml.groupby(['ward', 'industry'])[var].rolling(3).mean().reset_index(drop=True)
            df_ml[f'{var}_ma5'] = df_ml.groupby(['ward', 'industry'])[var].rolling(5).mean().reset_index(drop=True)
        
        # Growth rates
        for var in ['employment', 'concentration_index', 'ai_adoption']:
            df_ml[f'{var}_growth'] = df_ml.groupby(['ward', 'industry'])[var].pct_change()
        
        # Interaction terms
        df_ml['ai_x_education'] = df_ml['ai_adoption'] * df_ml['education']
        df_ml['ai_x_infrastructure'] = df_ml['ai_adoption'] * df_ml['infrastructure']
        df_ml['elderly_x_ai'] = df_ml['elderly_share'] * df_ml['ai_adoption']
        
        # Industry and ward dummies
        industry_dummies = pd.get_dummies(df_ml['industry'], prefix='ind')
        ward_dummies = pd.get_dummies(df_ml['ward'], prefix='ward')
        
        # Combine all features
        feature_columns = [
            'time_trend', 'time_trend_squared', 'ai_adoption', 'infrastructure', 
            'education', 'elderly_share', 'young_share', 'distance_to_center',
            'market_potential', 'total_shock'
        ]
        
        # Add lagged variables (exclude NaN rows)
        for var in lag_vars:
            for lag in [1, 2, 3]:
                feature_columns.append(f'{var}_lag{lag}')
        
        # Add moving averages
        feature_columns.extend(['employment_ma3', 'concentration_index_ma3'])
        
        # Add growth rates
        feature_columns.extend(['employment_growth', 'concentration_index_growth', 'ai_adoption_growth'])
        
        # Add interactions
        feature_columns.extend(['ai_x_education', 'ai_x_infrastructure', 'elderly_x_ai'])
        
        # Combine with dummies
        df_features = pd.concat([
            df_ml[['year', 'ward', 'industry', 'employment', 'concentration_index', 'productivity'] + feature_columns],
            industry_dummies, 
            ward_dummies
        ], axis=1)
        
        # Remove rows with NaN (due to lags)
        df_features = df_features.dropna()
        
        self.df_ml = df_features
        self.df_ml.to_csv(self.predictions_dir / 'ml_features.csv', index=False)
        
        logger.info(f"ML features prepared: {df_features.shape[0]} observations, {df_features.shape[1]} features")
        return df_features
    
    def train_ml_models(self):
        """
        Train machine learning models for predictions
        """
        logger.info("Training machine learning models...")
        
        # Prepare data
        target_vars = ['employment', 'concentration_index', 'productivity']
        
        # Feature columns (exclude target variables and identifiers)
        feature_cols = [col for col in self.df_ml.columns 
                       if col not in ['year', 'ward', 'industry'] + target_vars]
        
        X = self.df_ml[feature_cols]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # Save scaler
        joblib.dump(scaler, self.models_dir / 'feature_scaler.joblib')
        
        models = {}
        performance = {}
        
        for target in target_vars:
            logger.info(f"Training models for {target}...")
            
            y = self.df_ml[target]
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Initialize models
            models[target] = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
            }
            
            performance[target] = {}
            
            # Train and evaluate each model
            for model_name, model in models[target].items():
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
                
                # Train on full data
                model.fit(X_scaled, y)
                
                # Make predictions
                y_pred = model.predict(X_scaled)
                
                # Calculate metrics
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                
                performance[target][model_name] = {
                    'r2_score': r2,
                    'mae': mae,
                    'mse': mse,
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std()
                }
                
                # Save model
                joblib.dump(model, self.models_dir / f'{target}_{model_name}.joblib')
                
                logger.info(f"{target} - {model_name}: R² = {r2:.3f}, MAE = {mae:.3f}")
        
        # Save performance metrics
        perf_data = []
        for target, target_perf in performance.items():
            for model_name, metrics in target_perf.items():
                perf_data.append({
                    'target_variable': target,
                    'model': model_name,
                    **metrics
                })
        
        perf_df = pd.DataFrame(perf_data)
        perf_df.to_csv(self.predictions_dir / 'model_performance.csv', index=False)
        
        self.models = models
        self.performance = performance
        
        logger.info("Machine learning models trained successfully")
        return models, performance
    
    def generate_scenarios(self):
        """
        Generate future scenarios for predictions
        """
        logger.info("Generating future scenarios...")
        
        # Scenario parameters
        demographic_scenarios = {
            'pessimistic': {'fertility_rate': 1.1, 'immigration_rate': 0.001, 'life_expectancy_increase': 0.1},
            'baseline': {'fertility_rate': 1.3, 'immigration_rate': 0.002, 'life_expectancy_increase': 0.2},
            'optimistic': {'fertility_rate': 1.6, 'immigration_rate': 0.005, 'life_expectancy_increase': 0.3}
        }
        
        ai_scenarios = {
            'conservative': {'annual_adoption_rate': 0.02, 'productivity_boost': 0.03},
            'moderate': {'annual_adoption_rate': 0.05, 'productivity_boost': 0.08},
            'aggressive': {'annual_adoption_rate': 0.10, 'productivity_boost': 0.15}
        }
        
        economic_scenarios = {
            'stable': {'shock_probability': 0.05, 'shock_intensity': 0.10, 'growth_rate': 0.02},
            'volatile': {'shock_probability': 0.15, 'shock_intensity': 0.30, 'growth_rate': 0.015},
            'crisis': {'shock_probability': 0.25, 'shock_intensity': 0.50, 'growth_rate': 0.005}
        }
        
        # Generate all combinations (3 × 3 × 3 = 27 scenarios)
        scenarios = []
        
        for demo_name, demo_params in demographic_scenarios.items():
            for ai_name, ai_params in ai_scenarios.items():
                for econ_name, econ_params in economic_scenarios.items():
                    scenarios.append({
                        'scenario_name': f"{demo_name}_{ai_name}_{econ_name}",
                        'demographic': demo_name,
                        'ai_adoption': ai_name,
                        'economic': econ_name,
                        **demo_params,
                        **ai_params,
                        **econ_params
                    })
        
        self.scenarios = scenarios
        logger.info(f"Generated {len(scenarios)} scenarios")
        return scenarios
    
    def make_predictions(self, prediction_years=range(2024, 2051)):
        """
        Make predictions for future years under different scenarios
        """
        logger.info("Making predictions for future scenarios...")
        
        # Load trained models and scaler
        scaler = joblib.load(self.models_dir / 'feature_scaler.joblib')
        
        all_predictions = []
        
        for scenario in self.scenarios:
            logger.info(f"Predicting scenario: {scenario['scenario_name']}")
            
            scenario_predictions = []
            
            # Use 2023 as base year
            base_data = self.df_ml[self.df_ml['year'] == 2023].copy()
            
            for pred_year in prediction_years:
                year_predictions = []
                
                for _, row in base_data.iterrows():
                    # Update time-dependent variables
                    years_ahead = pred_year - 2023
                    
                    # Demographic evolution
                    if scenario['demographic'] == 'pessimistic':
                        elderly_growth = 0.015  # 1.5% per year
                        young_decline = 0.008
                    elif scenario['demographic'] == 'baseline':
                        elderly_growth = 0.012
                        young_decline = 0.006
                    else:  # optimistic
                        elderly_growth = 0.008
                        young_decline = 0.003
                    
                    new_elderly_share = min(0.5, row['elderly_share'] + elderly_growth * years_ahead)
                    new_young_share = max(0.1, row['young_share'] - young_decline * years_ahead)
                    new_working_age = 1 - new_elderly_share - new_young_share
                    
                    # AI adoption evolution
                    ai_growth = scenario['annual_adoption_rate']
                    new_ai_adoption = min(0.9, row['ai_adoption'] + ai_growth * years_ahead)
                    
                    # Infrastructure and education (gradual improvement)
                    new_infrastructure = min(1.0, row['infrastructure'] + 0.01 * years_ahead)
                    new_education = min(1.0, row['education'] + 0.005 * years_ahead)
                    
                    # Economic shocks (probabilistic)
                    np.random.seed(42 + pred_year)  # Consistent across scenarios
                    shock_occurs = np.random.random() < scenario['shock_probability']
                    economic_shock = -scenario['shock_intensity'] if shock_occurs else 0
                    
                    # Update features for prediction
                    updated_features = row.copy()
                    updated_features['time_trend'] = pred_year - 2000
                    updated_features['time_trend_squared'] = (pred_year - 2000) ** 2
                    updated_features['elderly_share'] = new_elderly_share
                    updated_features['young_share'] = new_young_share
                    updated_features['ai_adoption'] = new_ai_adoption
                    updated_features['infrastructure'] = new_infrastructure
                    updated_features['education'] = new_education
                    updated_features['total_shock'] = economic_shock
                    
                    # Update interaction terms
                    updated_features['ai_x_education'] = new_ai_adoption * new_education
                    updated_features['ai_x_infrastructure'] = new_ai_adoption * new_infrastructure
                    updated_features['elderly_x_ai'] = new_elderly_share * new_ai_adoption
                    
                    # Prepare feature vector
                    feature_cols = [col for col in updated_features.index 
                                   if col not in ['year', 'ward', 'industry', 'employment', 
                                                 'concentration_index', 'productivity']]
                    
                    X_pred = updated_features[feature_cols].values.reshape(1, -1)
                    X_pred_scaled = scaler.transform(X_pred)
                    
                    # Make predictions using best models
                    predictions = {}
                    
                    for target in ['employment', 'concentration_index', 'productivity']:
                        # Use best performing model for each target
                        best_model_name = max(self.performance[target].keys(), 
                                            key=lambda x: self.performance[target][x]['r2_score'])
                        model = joblib.load(self.models_dir / f'{target}_{best_model_name}.joblib')
                        
                        pred_value = model.predict(X_pred_scaled)[0]
                        predictions[target] = pred_value
                    
                    year_predictions.append({
                        'scenario': scenario['scenario_name'],
                        'year': pred_year,
                        'ward': row['ward'],
                        'industry': row['industry'],
                        'predicted_employment': predictions['employment'],
                        'predicted_concentration': predictions['concentration_index'],
                        'predicted_productivity': predictions['productivity'],
                        'elderly_share': new_elderly_share,
                        'young_share': new_young_share,
                        'ai_adoption': new_ai_adoption,
                        'infrastructure': new_infrastructure,
                        'education': new_education,
                        'economic_shock': economic_shock
                    })
                
                scenario_predictions.extend(year_predictions)
            
            all_predictions.extend(scenario_predictions)
        
        # Convert to DataFrame and save
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(self.predictions_dir / 'all_scenario_predictions.csv', index=False)
        
        # Create summary by scenario
        scenario_summary = predictions_df.groupby(['scenario', 'year']).agg({
            'predicted_employment': 'mean',
            'predicted_concentration': 'mean',
            'predicted_productivity': 'mean',
            'ai_adoption': 'mean',
            'elderly_share': 'mean'
        }).reset_index()
        
        scenario_summary.to_csv(self.predictions_dir / 'scenario_summary.csv', index=False)
        
        logger.info("Predictions completed for all scenarios")
        return predictions_df, scenario_summary
    
    def create_dynamic_visualizations(self, temporal_df, demo_df, predictions_df):
        """
        Create comprehensive visualizations for dynamic analysis
        """
        logger.info("Creating dynamic analysis visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Temporal concentration trends
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Concentration evolution by industry
        for industry in temporal_df['industry'].unique():
            industry_data = temporal_df[temporal_df['industry'] == industry]
            ax1.plot(industry_data['window_center'], industry_data['gini_coefficient'], 
                    'o-', label=industry, linewidth=2, markersize=4)
        
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Gini Coefficient')
        ax1.set_title('Industry Concentration Evolution (5-year rolling windows)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: AI adoption vs concentration
        recent_temporal = temporal_df[temporal_df['window_center'] >= 2018]
        scatter = ax2.scatter(recent_temporal['avg_ai_adoption'], 
                            recent_temporal['gini_coefficient'],
                            c=recent_temporal['window_center'], 
                            s=60, alpha=0.7, cmap='viridis')
        ax2.set_xlabel('Average AI Adoption')
        ax2.set_ylabel('Gini Coefficient')
        ax2.set_title('AI Adoption vs Concentration (2018-2023)')
        plt.colorbar(scatter, ax=ax2, label='Year')
        
        # Plot 3: Demographic transition effects
        for industry in ['IT', 'Finance', 'Healthcare']:
            industry_demo = demo_df[demo_df['industry'] == industry]
            ax3.plot(industry_demo['year'], industry_demo['avg_concentration'], 
                    'o-', label=industry, linewidth=2, markersize=4)
        
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Average Concentration Index')
        ax3.set_title('Demographic Impact on Concentration')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Aging vs concentration correlation
        correlation_data = []
        for industry in demo_df['industry'].unique():
            industry_data = demo_df[demo_df['industry'] == industry]
            corr = np.corrcoef(industry_data['avg_elderly_share'], 
                             industry_data['avg_concentration'])[0, 1]
            correlation_data.append({'industry': industry, 'correlation': corr})
        
        corr_df = pd.DataFrame(correlation_data)
        bars = ax4.bar(corr_df['industry'], corr_df['correlation'], 
                      color=['red' if x < 0 else 'green' for x in corr_df['correlation']],
                      alpha=0.7)
        ax4.set_ylabel('Correlation Coefficient')
        ax4.set_title('Aging-Concentration Correlation by Industry')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.dynamic_viz_dir / 'temporal_analysis_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scenario predictions comparison
        if predictions_df is not None and len(predictions_df) > 0:
            self.create_scenario_visualizations(predictions_df)
        
        # 3. Model performance visualization
        self.create_model_performance_visualization()
        
        logger.info("Dynamic visualizations created successfully")
    
    def create_scenario_visualizations(self, predictions_df):
        """
        Create scenario comparison visualizations
        """
        # Aggregate predictions by scenario and year
        scenario_agg = predictions_df.groupby(['scenario', 'year']).agg({
            'predicted_employment': 'mean',
            'predicted_concentration': 'mean', 
            'predicted_productivity': 'mean'
        }).reset_index()
        
        # Select key scenarios for visualization
        key_scenarios = [
            'pessimistic_conservative_crisis',
            'baseline_moderate_stable', 
            'optimistic_aggressive_stable'
        ]
        
        scenario_subset = scenario_agg[scenario_agg['scenario'].isin(key_scenarios)]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot employment predictions
        for scenario in key_scenarios:
            scenario_data = scenario_subset[scenario_subset['scenario'] == scenario]
            ax1.plot(scenario_data['year'], scenario_data['predicted_employment'], 
                    'o-', label=scenario.replace('_', '-'), linewidth=2)
        
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Predicted Employment')
        ax1.set_title('Employment Projections by Scenario')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot concentration predictions
        for scenario in key_scenarios:
            scenario_data = scenario_subset[scenario_subset['scenario'] == scenario]
            ax2.plot(scenario_data['year'], scenario_data['predicted_concentration'], 
                    'o-', label=scenario.replace('_', '-'), linewidth=2)
        
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Predicted Concentration Index')
        ax2.set_title('Concentration Projections by Scenario')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot productivity predictions
        for scenario in key_scenarios:
            scenario_data = scenario_subset[scenario_subset['scenario'] == scenario]
            ax3.plot(scenario_data['year'], scenario_data['predicted_productivity'], 
                    'o-', label=scenario.replace('_', '-'), linewidth=2)
        
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Predicted Productivity')
        ax3.set_title('Productivity Projections by Scenario')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dynamic_viz_dir / 'scenario_predictions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_model_performance_visualization(self):
        """
        Create model performance comparison visualization
        """
        if hasattr(self, 'performance'):
            perf_data = []
            for target, target_perf in self.performance.items():
                for model_name, metrics in target_perf.items():
                    perf_data.append({
                        'target': target,
                        'model': model_name,
                        'r2_score': metrics['r2_score'],
                        'mae': metrics['mae']
                    })
            
            perf_df = pd.DataFrame(perf_data)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # R² scores by model and target
            perf_pivot_r2 = perf_df.pivot(index='model', columns='target', values='r2_score')
            perf_pivot_r2.plot(kind='bar', ax=ax1, alpha=0.8)
            ax1.set_title('Model Performance: R² Scores')
            ax1.set_ylabel('R² Score')
            ax1.legend(title='Target Variable')
            plt.setp(ax1.get_xticklabels(), rotation=45)
            
            # MAE by model and target
            perf_pivot_mae = perf_df.pivot(index='model', columns='target', values='mae')
            perf_pivot_mae.plot(kind='bar', ax=ax2, alpha=0.8)
            ax2.set_title('Model Performance: Mean Absolute Error')
            ax2.set_ylabel('MAE')
            ax2.legend(title='Target Variable')
            plt.setp(ax2.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.dynamic_viz_dir / 'model_performance_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_full_dynamic_analysis(self):
        """
        Run the complete dynamic analysis pipeline
        """
        logger.info("Starting comprehensive dynamic analysis...")
        
        # Generate comprehensive data
        self.generate_comprehensive_data()
        
        # Run temporal analysis
        temporal_df = self.temporal_agglomeration_analysis()
        demo_df = self.demographic_transition_analysis()
        
        # Prepare ML features and train models
        self.prepare_ml_features()
        self.train_ml_models()
        
        # Generate scenarios and make predictions
        self.generate_scenarios()
        predictions_df, scenario_summary = self.make_predictions()
        
        # Create visualizations
        self.create_dynamic_visualizations(temporal_df, demo_df, predictions_df)
        
        logger.info("Dynamic analysis completed successfully!")
        
        return {
            'temporal_results': temporal_df,
            'demographic_effects': demo_df,
            'predictions': predictions_df,
            'scenario_summary': scenario_summary,
            'model_performance': self.performance if hasattr(self, 'performance') else None
        }

def main():
    """
    Main execution function
    """
    print("=" * 70)
    print("ENHANCED DYNAMIC ANALYSIS WITH ML PREDICTIONS")
    print("Tokyo Agglomeration Analysis Framework")
    print("=" * 70)
    
    # Initialize framework
    analyzer = EnhancedDynamicAnalysis()
    
    # Run complete analysis
    results = analyzer.run_full_dynamic_analysis()
    
    # Print summary
    print("\nANALYSIS COMPLETED SUCCESSFULLY!")
    print("-" * 50)
    print("Generated Components:")
    print("• Comprehensive temporal data (2000-2023)")
    print("• Time-varying concentration analysis")
    print("• Demographic transition effects")
    print("• Machine learning models (RF, GB, NN)")
    print("• 27 future scenarios (2024-2050)")
    print("• Comprehensive visualizations")
    
    if 'model_performance' in results and results['model_performance']:
        print("\nModel Performance Summary:")
        print("-" * 30)
        for target, models in results['model_performance'].items():
            best_model = max(models.keys(), key=lambda x: models[x]['r2_score'])
            best_r2 = models[best_model]['r2_score']
            print(f"{target:20} | Best: {best_model:15} | R² = {best_r2:.3f}")
    
    print(f"\nOutput Locations:")
    print(f"• Data: data/dynamic_analysis_data.csv")
    print(f"• Results: results/temporal/ and results/predictions/")
    print(f"• Models: results/predictions/models/")
    print(f"• Visualizations: visualizations/dynamic/")
    
    print("\n" + "=" * 70)
    print("Dynamic analysis framework ready for manuscript!")
    print("=" * 70)

if __name__ == "__main__":
    main()