#!/usr/bin/env python3
"""
Theoretical-Empirical Validation Framework

This module validates the theoretical predictions from the AI-driven spatial 
distribution model against empirical patterns and provides advanced econometric
tests for the theoretical mechanisms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import networkx as nx
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TheoreticalEmpiricalValidator:
    """
    Advanced framework for validating theoretical predictions against empirical data
    """
    
    def __init__(self, theoretical_model=None, data_dir="data", results_dir="results"):
        self.theoretical_model = theoretical_model
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create validation results directory
        self.validation_dir = self.results_dir / "theoretical_validation"
        self.validation_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize theoretical predictions
        self.theoretical_predictions = {}
        self.empirical_tests = {}
        
        logger.info("Theoretical-Empirical Validation Framework initialized")
    
    def generate_theoretical_predictions(self):
        """
        Generate specific quantitative predictions from the theoretical model
        """
        logger.info("Generating theoretical predictions...")
        
        self.theoretical_predictions = {
            'ai_concentration_hypothesis': {
                'prediction': 'AI adoption concentrates in high-infrastructure, high-human-capital locations',
                'testable_implication': 'Correlation between initial infrastructure/HC and AI adoption > 0.6',
                'statistical_test': 'correlation_test',
                'threshold': 0.6,
                'direction': 'positive'
            },
            
            'heterogeneous_returns_hypothesis': {
                'prediction': 'Productivity gains from AI vary significantly across locations',
                'testable_implication': 'Coefficient of variation in AI productivity effects > 0.3',
                'statistical_test': 'heterogeneity_test',
                'threshold': 0.3,
                'direction': 'greater_than'
            },
            
            'network_amplification_hypothesis': {
                'prediction': 'Locations in high-AI networks experience amplified productivity gains',
                'testable_implication': 'Network AI exposure coefficient > own AI coefficient in productivity regression',
                'statistical_test': 'network_effects_test',
                'threshold': 1.0,  # Network effect relative to own effect
                'direction': 'greater_than'
            },
            
            'dynamic_divergence_hypothesis': {
                'prediction': 'Early AI adoption differences amplify over time',
                'testable_implication': 'Variance in AI adoption increases over time with acceleration',
                'statistical_test': 'divergence_test',
                'threshold': 0.05,  # Annual increase in variance
                'direction': 'positive'
            },
            
            'virtual_agglomeration_hypothesis': {
                'prediction': 'AI reduces importance of physical proximity for knowledge work',
                'testable_implication': 'Distance coefficient becomes less negative over time in AI-intensive industries',
                'statistical_test': 'distance_decay_test',
                'threshold': 0.1,  # 10% reduction in distance sensitivity per year
                'direction': 'positive'
            },
            
            'complementarity_hypothesis': {
                'prediction': 'AI and human capital are complements in production',
                'testable_implication': 'AI × Human Capital interaction term > 0 in productivity regression',
                'statistical_test': 'complementarity_test',
                'threshold': 0.0,
                'direction': 'positive'
            }
        }
        
        return self.theoretical_predictions
    
    def test_ai_concentration_hypothesis(self, empirical_data):
        """
        Test whether AI adoption concentrates in high-infrastructure locations
        """
        logger.info("Testing AI concentration hypothesis...")
        
        # Calculate correlations between initial conditions and AI adoption
        correlations = {}
        
        # Infrastructure correlation
        infra_corr = np.corrcoef(empirical_data['infrastructure'], empirical_data['ai_adoption'])[0, 1]
        correlations['infrastructure'] = infra_corr
        
        # Human capital correlation
        hc_corr = np.corrcoef(empirical_data['human_capital'], empirical_data['ai_adoption'])[0, 1]
        correlations['human_capital'] = hc_corr
        
        # Education correlation
        education_corr = np.corrcoef(empirical_data['education'], empirical_data['ai_adoption'])[0, 1]
        correlations['education'] = education_corr
        
        # Combined index
        combined_index = (empirical_data['infrastructure'] + empirical_data['human_capital'] + 
                         empirical_data['education']) / 3
        combined_corr = np.corrcoef(combined_index, empirical_data['ai_adoption'])[0, 1]
        correlations['combined_index'] = combined_corr
        
        # Statistical significance test
        n = len(empirical_data)
        t_stat = combined_corr * np.sqrt((n - 2) / (1 - combined_corr**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        test_result = {
            'hypothesis': 'ai_concentration_hypothesis',
            'correlations': correlations,
            'main_correlation': combined_corr,
            'threshold': self.theoretical_predictions['ai_concentration_hypothesis']['threshold'],
            'prediction_supported': combined_corr > 0.6,
            'statistical_significance': p_value < 0.05,
            'p_value': p_value,
            't_statistic': t_stat
        }
        
        return test_result
    
    def test_heterogeneous_returns_hypothesis(self, empirical_data):
        """
        Test whether AI productivity effects are heterogeneous across locations
        """
        logger.info("Testing heterogeneous returns hypothesis...")
        
        # Estimate location-specific AI productivity effects
        location_effects = []
        
        unique_locations = empirical_data['location'].unique()
        
        for location in unique_locations:
            location_data = empirical_data[empirical_data['location'] == location]
            
            if len(location_data) > 5:  # Sufficient observations
                # Estimate productivity-AI relationship for this location
                X = location_data[['ai_adoption', 'human_capital', 'infrastructure']].values
                y = location_data['productivity'].values
                
                if len(X) > 0 and len(y) > 0 and X.shape[0] == y.shape[0]:
                    try:
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        ai_effect = model.coef_[0]  # AI coefficient
                        location_effects.append(ai_effect)
                    except Exception as e:
                        logger.warning(f"Could not fit model for location {location}: {e}")
                        continue
        
        if len(location_effects) == 0:
            logger.warning("No location effects could be estimated")
            location_effects = [0.0]  # Fallback to avoid empty array
        
        location_effects = np.array(location_effects)
        
        # Calculate coefficient of variation
        mean_effect = np.mean(location_effects)
        std_effect = np.std(location_effects)
        cv = std_effect / abs(mean_effect) if mean_effect != 0 else np.inf
        
        # Test for heterogeneity using F-test approach
        # Null hypothesis: all location effects are equal
        n_locations = len(location_effects)
        if n_locations > 1:
            f_stat = np.var(location_effects) / (np.mean(location_effects)**2 / n_locations) if np.mean(location_effects) != 0 else 0
            p_value_hetero = 1 - stats.f.cdf(f_stat, n_locations-1, n_locations-1) if f_stat > 0 else 1.0
        else:
            f_stat = 0
            p_value_hetero = 1.0
        
        test_result = {
            'hypothesis': 'heterogeneous_returns_hypothesis',
            'coefficient_of_variation': cv,
            'location_effects': location_effects.tolist(),
            'mean_effect': mean_effect,
            'std_effect': std_effect,
            'threshold': self.theoretical_predictions['heterogeneous_returns_hypothesis']['threshold'],
            'prediction_supported': cv > 0.3,
            'statistical_significance': p_value_hetero < 0.05,
            'p_value': p_value_hetero,
            'f_statistic': f_stat,
            'n_locations_analyzed': n_locations
        }
        
        return test_result
    
    def test_network_amplification_hypothesis(self, empirical_data):
        """
        Test whether network AI exposure amplifies productivity gains
        """
        logger.info("Testing network amplification hypothesis...")
        
        # Create spatial network based on distance
        unique_locations = empirical_data['location'].unique()
        n_locations = len(unique_locations)
        
        # Generate spatial coordinates (for demonstration)
        np.random.seed(42)
        coordinates = {loc: (np.random.uniform(0, 100), np.random.uniform(0, 100)) 
                      for loc in unique_locations}
        
        # Calculate network AI exposure for each location
        network_ai_exposure = {}
        
        for loc in unique_locations:
            exposure = 0
            loc_data = empirical_data[empirical_data['location'] == loc]
            
            if len(loc_data) > 0:
                own_coords = coordinates[loc]
                
                for other_loc in unique_locations:
                    if other_loc != loc:
                        other_coords = coordinates[other_loc]
                        distance = np.sqrt((own_coords[0] - other_coords[0])**2 + 
                                         (own_coords[1] - other_coords[1])**2)
                        
                        # Weight by inverse distance
                        weight = 1 / (1 + distance/10)
                        
                        other_data = empirical_data[empirical_data['location'] == other_loc]
                        if len(other_data) > 0:
                            other_ai = other_data['ai_adoption'].mean()
                            exposure += weight * other_ai
                
                network_ai_exposure[loc] = exposure
        
        # Add network exposure to data
        empirical_data_enhanced = empirical_data.copy()
        empirical_data_enhanced['network_ai_exposure'] = empirical_data_enhanced['location'].map(network_ai_exposure)
        
        # Estimate productivity regression with own AI and network AI
        X = empirical_data_enhanced[['ai_adoption', 'network_ai_exposure', 'human_capital', 'infrastructure']]
        y = empirical_data_enhanced['productivity']
        
        model = LinearRegression()
        model.fit(X, y)
        
        own_ai_coeff = model.coef_[0]
        network_ai_coeff = model.coef_[1]
        
        # Test whether network effect exceeds own effect
        amplification_ratio = network_ai_coeff / own_ai_coeff if own_ai_coeff != 0 else 0
        
        # Statistical test for difference
        # Simplified approach - in practice would use robust standard errors
        predictions = model.predict(X)
        residuals = y - predictions
        mse = np.mean(residuals**2)
        
        test_result = {
            'hypothesis': 'network_amplification_hypothesis',
            'own_ai_coefficient': own_ai_coeff,
            'network_ai_coefficient': network_ai_coeff,
            'amplification_ratio': amplification_ratio,
            'threshold': self.theoretical_predictions['network_amplification_hypothesis']['threshold'],
            'prediction_supported': amplification_ratio > 1.0,
            'model_r2': r2_score(y, predictions),
            'network_exposure_data': network_ai_exposure
        }
        
        return test_result
    
    def test_dynamic_divergence_hypothesis(self, empirical_data):
        """
        Test whether AI adoption differences amplify over time
        """
        logger.info("Testing dynamic divergence hypothesis...")
        
        # Calculate variance in AI adoption over time
        time_periods = sorted(empirical_data['year'].unique())
        variance_over_time = []
        
        for year in time_periods:
            year_data = empirical_data[empirical_data['year'] == year]
            ai_variance = np.var(year_data['ai_adoption'])
            variance_over_time.append({'year': year, 'ai_variance': ai_variance})
        
        variance_df = pd.DataFrame(variance_over_time)
        
        # Test for increasing variance (linear trend)
        X = variance_df['year'].values.reshape(-1, 1)
        y = variance_df['ai_variance'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        trend_coefficient = model.coef_[0]
        r2 = r2_score(y, model.predict(X))
        
        # Test for acceleration (quadratic trend)
        X_quad = np.column_stack([variance_df['year'], variance_df['year']**2])
        model_quad = LinearRegression()
        model_quad.fit(X_quad, y)
        
        acceleration_coeff = model_quad.coef_[1]
        r2_quad = r2_score(y, model_quad.predict(X_quad))
        
        test_result = {
            'hypothesis': 'dynamic_divergence_hypothesis',
            'variance_over_time': variance_df,
            'linear_trend_coefficient': trend_coefficient,
            'acceleration_coefficient': acceleration_coeff,
            'linear_r2': r2,
            'quadratic_r2': r2_quad,
            'threshold': self.theoretical_predictions['dynamic_divergence_hypothesis']['threshold'],
            'prediction_supported': trend_coefficient > 0.05,
            'acceleration_present': acceleration_coeff > 0
        }
        
        return test_result
    
    def test_virtual_agglomeration_hypothesis(self, empirical_data):
        """
        Test whether AI reduces importance of physical proximity
        """
        logger.info("Testing virtual agglomeration hypothesis...")
        
        # For each year, estimate distance coefficient in productivity regression
        time_periods = sorted(empirical_data['year'].unique())
        distance_coefficients = []
        
        for year in time_periods:
            year_data = empirical_data[empirical_data['year'] == year]
            
            if len(year_data) > 10:  # Sufficient observations
                # Add distance from center (assuming location 0 is center)
                year_data = year_data.copy()
                year_data['distance_from_center'] = year_data['location'] * 2.5  # Simplified
                
                # Estimate productivity regression with distance
                X = year_data[['ai_adoption', 'distance_from_center', 'human_capital', 'infrastructure']]
                y = year_data['productivity']
                
                model = LinearRegression()
                model.fit(X, y)
                
                distance_coeff = model.coef_[1]  # Distance coefficient
                distance_coefficients.append({'year': year, 'distance_coefficient': distance_coeff})
        
        distance_df = pd.DataFrame(distance_coefficients)
        
        # Test whether distance coefficient becomes less negative over time
        X = distance_df['year'].values.reshape(-1, 1)
        y = distance_df['distance_coefficient'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        time_trend = model.coef_[0]  # Should be positive if distance matters less over time
        
        test_result = {
            'hypothesis': 'virtual_agglomeration_hypothesis',
            'distance_coefficients_over_time': distance_df,
            'time_trend': time_trend,
            'threshold': self.theoretical_predictions['virtual_agglomeration_hypothesis']['threshold'],
            'prediction_supported': time_trend > 0.1,
            'initial_distance_effect': distance_df['distance_coefficient'].iloc[0] if len(distance_df) > 0 else None,
            'final_distance_effect': distance_df['distance_coefficient'].iloc[-1] if len(distance_df) > 0 else None
        }
        
        return test_result
    
    def test_complementarity_hypothesis(self, empirical_data):
        """
        Test whether AI and human capital are complements
        """
        logger.info("Testing AI-human capital complementarity hypothesis...")
        
        # Create interaction term
        empirical_data_enhanced = empirical_data.copy()
        empirical_data_enhanced['ai_hc_interaction'] = (empirical_data['ai_adoption'] * 
                                                       empirical_data['human_capital'])
        
        # Estimate production function with interaction
        X = empirical_data_enhanced[['ai_adoption', 'human_capital', 'ai_hc_interaction', 'infrastructure']]
        y = empirical_data_enhanced['productivity']
        
        model = LinearRegression()
        model.fit(X, y)
        
        ai_coeff = model.coef_[0]
        hc_coeff = model.coef_[1]
        interaction_coeff = model.coef_[2]
        
        # Test statistical significance of interaction
        predictions = model.predict(X)
        residuals = y - predictions
        mse = np.mean(residuals**2)
        n = len(y)
        
        # Simplified standard error calculation
        X_np = X.values
        var_covar = mse * np.linalg.inv(X_np.T @ X_np)
        interaction_se = np.sqrt(var_covar[2, 2])
        
        t_stat = interaction_coeff / interaction_se if interaction_se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - X.shape[1]))
        
        test_result = {
            'hypothesis': 'complementarity_hypothesis',
            'ai_coefficient': ai_coeff,
            'human_capital_coefficient': hc_coeff,
            'interaction_coefficient': interaction_coeff,
            'interaction_standard_error': interaction_se,
            'interaction_t_statistic': t_stat,
            'interaction_p_value': p_value,
            'threshold': self.theoretical_predictions['complementarity_hypothesis']['threshold'],
            'prediction_supported': interaction_coeff > 0,
            'statistical_significance': p_value < 0.05,
            'model_r2': r2_score(y, predictions)
        }
        
        return test_result
    
    def structural_model_estimation(self, empirical_data):
        """
        Estimate the structural parameters of the theoretical model
        """
        logger.info("Estimating structural model parameters...")
        
        def objective_function(params, data):
            """
            Objective function for structural estimation
            """
            # Extract parameters
            alpha_ai, beta_learning, gamma_network, delta_digital, phi_distance = params
            
            # Ensure parameters are positive
            if any(p < 0 for p in params) or any(p > 2 for p in params):
                return 1e10  # Large penalty for invalid parameters
            
            # Calculate model predictions
            predicted_productivity = np.zeros(len(data))
            
            for i, row in data.iterrows():
                # Base productivity
                base_prod = row['human_capital'] * row['infrastructure']
                
                # AI effect
                ai_effect = 1 + alpha_ai * row['ai_adoption']
                
                # Learning spillovers (simplified)
                learning_effect = 1 + beta_learning * row['ai_adoption'] * 0.1  # Simplified
                
                # Network effects (simplified)
                network_effect = 1 + gamma_network * row['ai_adoption'] * 0.1
                
                # Digital infrastructure effect
                digital_effect = 1 + delta_digital * row['infrastructure'] * row['ai_adoption']
                
                predicted_productivity[i] = base_prod * ai_effect * learning_effect * network_effect * digital_effect
            
            # Calculate objective (minimize sum of squared errors)
            actual_productivity = data['productivity'].values
            sse = np.sum((actual_productivity - predicted_productivity)**2)
            
            return sse
        
        # Initial parameter guess
        initial_params = [0.25, 0.15, 0.10, 0.20, 0.08]  # From theoretical model
        
        # Bounds for parameters
        bounds = [(0.1, 0.5), (0.05, 0.3), (0.05, 0.2), (0.1, 0.4), (0.05, 0.15)]
        
        # Optimization
        result = minimize(objective_function, initial_params, args=(empirical_data,), 
                         bounds=bounds, method='L-BFGS-B')
        
        estimated_params = {
            'alpha_ai': result.x[0],
            'beta_learning': result.x[1],
            'gamma_network': result.x[2],
            'delta_digital': result.x[3],
            'phi_distance': result.x[4],
            'objective_value': result.fun,
            'optimization_success': result.success
        }
        
        return estimated_params
    
    def model_selection_tests(self, empirical_data):
        """
        Compare theoretical model against alternative specifications
        """
        logger.info("Running model selection tests...")
        
        # Model 1: Traditional agglomeration (no AI)
        X1 = empirical_data[['human_capital', 'infrastructure']]
        y = empirical_data['productivity']
        
        model1 = LinearRegression()
        model1.fit(X1, y)
        r2_traditional = r2_score(y, model1.predict(X1))
        
        # Model 2: Simple AI addition
        X2 = empirical_data[['human_capital', 'infrastructure', 'ai_adoption']]
        
        model2 = LinearRegression()
        model2.fit(X2, y)
        r2_simple_ai = r2_score(y, model2.predict(X2))
        
        # Model 3: AI with interactions (our theoretical model)
        empirical_data_enhanced = empirical_data.copy()
        empirical_data_enhanced['ai_hc_interaction'] = (empirical_data['ai_adoption'] * 
                                                       empirical_data['human_capital'])
        empirical_data_enhanced['ai_infra_interaction'] = (empirical_data['ai_adoption'] * 
                                                          empirical_data['infrastructure'])
        
        X3 = empirical_data_enhanced[['human_capital', 'infrastructure', 'ai_adoption',
                                     'ai_hc_interaction', 'ai_infra_interaction']]
        
        model3 = LinearRegression()
        model3.fit(X3, y)
        r2_theoretical = r2_score(y, model3.predict(X3))
        
        # Model 4: Full non-linear (Random Forest baseline)
        X4 = empirical_data[['human_capital', 'infrastructure', 'ai_adoption']]
        
        model4 = RandomForestRegressor(n_estimators=100, random_state=42)
        model4.fit(X4, y)
        r2_nonlinear = r2_score(y, model4.predict(X4))
        
        # Information criteria (AIC approximation)
        n = len(y)
        
        def calculate_aic(r2, k, n):
            if r2 >= 1:
                return np.inf
            log_likelihood = -n/2 * np.log(1 - r2)
            return 2*k - 2*log_likelihood
        
        aic_traditional = calculate_aic(r2_traditional, X1.shape[1], n)
        aic_simple_ai = calculate_aic(r2_simple_ai, X2.shape[1], n)
        aic_theoretical = calculate_aic(r2_theoretical, X3.shape[1], n)
        
        model_comparison = {
            'traditional_agglomeration': {
                'r2': r2_traditional,
                'aic': aic_traditional,
                'variables': X1.shape[1]
            },
            'simple_ai_model': {
                'r2': r2_simple_ai,
                'aic': aic_simple_ai,
                'variables': X2.shape[1]
            },
            'theoretical_model': {
                'r2': r2_theoretical,
                'aic': aic_theoretical,
                'variables': X3.shape[1]
            },
            'nonlinear_baseline': {
                'r2': r2_nonlinear,
                'aic': None,  # Not applicable for RF
                'variables': X4.shape[1]
            }
        }
        
        # Best model by AIC
        aic_values = {k: v['aic'] for k, v in model_comparison.items() if v['aic'] is not None}
        best_model_aic = min(aic_values, key=aic_values.get)
        
        # Best model by R²
        r2_values = {k: v['r2'] for k, v in model_comparison.items()}
        best_model_r2 = max(r2_values, key=r2_values.get)
        
        model_comparison['best_model_aic'] = best_model_aic
        model_comparison['best_model_r2'] = best_model_r2
        model_comparison['theoretical_model_preferred'] = best_model_aic == 'theoretical_model'
        
        return model_comparison
    
    def generate_empirical_data(self, n_observations=1000):
        """
        Generate realistic empirical data for testing (when real data not available)
        """
        logger.info("Generating synthetic empirical data for validation...")
        
        np.random.seed(42)
        
        data = []
        locations = range(23)  # Tokyo wards
        years = range(2000, 2024)
        
        for year in years:
            for location in locations:
                # Base characteristics
                infrastructure = 0.3 + 0.03 * location + 0.01 * (year - 2000) + np.random.normal(0, 0.1)
                infrastructure = max(0.1, min(1.0, infrastructure))
                
                human_capital = 0.4 + 0.02 * location + 0.005 * (year - 2000) + np.random.normal(0, 0.08)
                human_capital = max(0.1, min(1.0, human_capital))
                
                education = 0.35 + 0.025 * location + 0.008 * (year - 2000) + np.random.normal(0, 0.07)
                education = max(0.1, min(1.0, education))
                
                # AI adoption (accelerating after 2015, location-dependent)
                if year < 2015:
                    ai_base = 0.05 + 0.01 * infrastructure + 0.01 * human_capital
                else:
                    ai_growth = 0.08 * (year - 2015) * (infrastructure + human_capital)
                    ai_base = 0.05 + 0.01 * infrastructure + 0.01 * human_capital + ai_growth
                
                ai_adoption = ai_base + np.random.normal(0, 0.03)
                ai_adoption = max(0.01, min(0.9, ai_adoption))
                
                # Productivity (following theoretical model)
                base_productivity = human_capital * infrastructure
                ai_effect = 1 + 0.25 * ai_adoption  # alpha_ai = 0.25
                complementarity_effect = 1 + 0.12 * ai_adoption * human_capital  # theta_complement
                infrastructure_effect = 1 + 0.20 * infrastructure * ai_adoption  # delta_digital
                
                productivity = (base_productivity * ai_effect * complementarity_effect * 
                              infrastructure_effect + np.random.normal(0, 0.1))
                productivity = max(0.1, productivity)
                
                data.append({
                    'year': year,
                    'location': location,
                    'infrastructure': infrastructure,
                    'human_capital': human_capital,
                    'education': education,
                    'ai_adoption': ai_adoption,
                    'productivity': productivity
                })
        
        return pd.DataFrame(data)
    
    def run_comprehensive_validation(self, empirical_data=None):
        """
        Run comprehensive validation of theoretical predictions
        """
        logger.info("Running comprehensive theoretical validation...")
        
        # Generate data if not provided
        if empirical_data is None:
            empirical_data = self.generate_empirical_data()
        
        # Generate theoretical predictions
        self.generate_theoretical_predictions()
        
        # Run all hypothesis tests
        validation_results = {}
        
        validation_results['ai_concentration'] = self.test_ai_concentration_hypothesis(empirical_data)
        validation_results['heterogeneous_returns'] = self.test_heterogeneous_returns_hypothesis(empirical_data)
        validation_results['network_amplification'] = self.test_network_amplification_hypothesis(empirical_data)
        validation_results['dynamic_divergence'] = self.test_dynamic_divergence_hypothesis(empirical_data)
        validation_results['virtual_agglomeration'] = self.test_virtual_agglomeration_hypothesis(empirical_data)
        validation_results['complementarity'] = self.test_complementarity_hypothesis(empirical_data)
        
        # Structural estimation
        validation_results['structural_parameters'] = self.structural_model_estimation(empirical_data)
        
        # Model selection
        validation_results['model_comparison'] = self.model_selection_tests(empirical_data)
        
        # Overall validation summary
        supported_hypotheses = sum(1 for test in validation_results.values() 
                                 if isinstance(test, dict) and test.get('prediction_supported', False))
        total_hypotheses = 6
        
        validation_results['summary'] = {
            'total_hypotheses_tested': total_hypotheses,
            'hypotheses_supported': supported_hypotheses,
            'support_rate': supported_hypotheses / total_hypotheses,
            'theoretical_model_preferred': validation_results['model_comparison']['theoretical_model_preferred'],
            'overall_validation_success': (supported_hypotheses / total_hypotheses) > 0.5
        }
        
        # Save results
        self.save_validation_results(validation_results)
        
        logger.info("Comprehensive validation completed")
        return validation_results
    
    def save_validation_results(self, validation_results):
        """
        Save validation results to files
        """
        # Save summary results
        summary_data = []
        for hypothesis, result in validation_results.items():
            if isinstance(result, dict) and 'hypothesis' in result:
                summary_data.append({
                    'hypothesis': result['hypothesis'],
                    'prediction_supported': result.get('prediction_supported', False),
                    'statistical_significance': result.get('statistical_significance', False),
                    'p_value': result.get('p_value', None),
                    'main_statistic': result.get('main_correlation', result.get('coefficient_of_variation', 
                                                result.get('amplification_ratio', None)))
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.validation_dir / 'hypothesis_test_summary.csv', index=False)
        
        # Save detailed results (simplified for CSV)
        detailed_results = {}
        for key, result in validation_results.items():
            if isinstance(result, dict):
                # Convert complex objects to simple representations
                simplified_result = {}
                for k, v in result.items():
                    if isinstance(v, (int, float, bool, str)) or v is None:
                        simplified_result[k] = v
                    elif isinstance(v, np.ndarray):
                        simplified_result[k] = v.tolist() if v.size < 100 else 'large_array'
                    elif isinstance(v, dict):
                        simplified_result[k] = str(v)
                    else:
                        simplified_result[k] = str(v)
                detailed_results[key] = simplified_result
        
        # Save as JSON for better structure preservation
        import json
        with open(self.validation_dir / 'detailed_validation_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        logger.info("Validation results saved")
    
    def create_validation_visualizations(self, validation_results):
        """
        Create comprehensive validation visualizations
        """
        logger.info("Creating validation visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        # 1. Hypothesis Support Summary
        ax = axes[0]
        hypotheses = ['AI Concentration', 'Heterogeneous Returns', 'Network Amplification',
                     'Dynamic Divergence', 'Virtual Agglomeration', 'Complementarity']
        
        support_status = []
        for key in ['ai_concentration', 'heterogeneous_returns', 'network_amplification',
                   'dynamic_divergence', 'virtual_agglomeration', 'complementarity']:
            if key in validation_results:
                support_status.append(validation_results[key].get('prediction_supported', False))
            else:
                support_status.append(False)
        
        colors = ['green' if supported else 'red' for supported in support_status]
        bars = ax.bar(range(len(hypotheses)), [1 if s else 0 for s in support_status], 
                     color=colors, alpha=0.7)
        ax.set_xticks(range(len(hypotheses)))
        ax.set_xticklabels(hypotheses, rotation=45, ha='right')
        ax.set_ylabel('Hypothesis Supported')
        ax.set_title('Theoretical Hypothesis Validation Results')
        ax.set_ylim(0, 1.2)
        
        # Add labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   'Supported' if support_status[i] else 'Not Supported',
                   ha='center', va='bottom', fontsize=8)
        
        # 2. Model Comparison
        ax = axes[1]
        if 'model_comparison' in validation_results:
            models = ['Traditional', 'Simple AI', 'Theoretical', 'Nonlinear']
            r2_values = [
                validation_results['model_comparison']['traditional_agglomeration']['r2'],
                validation_results['model_comparison']['simple_ai_model']['r2'],
                validation_results['model_comparison']['theoretical_model']['r2'],
                validation_results['model_comparison']['nonlinear_baseline']['r2']
            ]
            
            bars = ax.bar(models, r2_values, alpha=0.7)
            ax.set_ylabel('R² Score')
            ax.set_title('Model Performance Comparison')
            
            # Highlight best model
            best_idx = np.argmax(r2_values)
            bars[best_idx].set_color('gold')
            
            # Add value labels
            for bar, r2 in zip(bars, r2_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{r2:.3f}', ha='center', va='bottom')
        
        # 3. Structural Parameters Comparison
        ax = axes[2]
        if 'structural_parameters' in validation_results:
            theoretical_params = [0.25, 0.15, 0.10, 0.20, 0.08]  # From model
            estimated_params = [
                validation_results['structural_parameters']['alpha_ai'],
                validation_results['structural_parameters']['beta_learning'],
                validation_results['structural_parameters']['gamma_network'],
                validation_results['structural_parameters']['delta_digital'],
                validation_results['structural_parameters']['phi_distance']
            ]
            
            param_names = ['α_AI', 'β_learning', 'γ_network', 'δ_digital', 'φ_distance']
            
            x = np.arange(len(param_names))
            width = 0.35
            
            ax.bar(x - width/2, theoretical_params, width, label='Theoretical', alpha=0.7)
            ax.bar(x + width/2, estimated_params, width, label='Estimated', alpha=0.7)
            
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Parameter Values')
            ax.set_title('Theoretical vs Estimated Parameters')
            ax.set_xticks(x)
            ax.set_xticklabels(param_names)
            ax.legend()
        
        # 4-9. Individual test results visualizations
        test_plots = [
            ('ai_concentration', 'AI Concentration Test'),
            ('heterogeneous_returns', 'Heterogeneous Returns Test'),
            ('network_amplification', 'Network Effects Test'),
            ('dynamic_divergence', 'Dynamic Divergence Test'),
            ('virtual_agglomeration', 'Virtual Agglomeration Test'),
            ('complementarity', 'Complementarity Test')
        ]
        
        for i, (test_key, title) in enumerate(test_plots, 3):
            ax = axes[i]
            
            if test_key in validation_results:
                result = validation_results[test_key]
                
                # Create appropriate visualization based on test type
                if test_key == 'ai_concentration':
                    # Correlation plot
                    correlations = result.get('correlations', {})
                    if correlations:
                        bars = ax.bar(correlations.keys(), correlations.values(), alpha=0.7)
                        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Threshold')
                        ax.set_ylabel('Correlation')
                        ax.set_title(title)
                        ax.legend()
                        plt.setp(ax.get_xticklabels(), rotation=45)
                
                elif test_key == 'dynamic_divergence':
                    # Time series of variance
                    if 'variance_over_time' in result:
                        variance_data = result['variance_over_time']
                        ax.plot(variance_data['year'], variance_data['ai_variance'], 'o-')
                        ax.set_xlabel('Year')
                        ax.set_ylabel('AI Adoption Variance')
                        ax.set_title(title)
                
                else:
                    # Generic visualization showing main statistic vs threshold
                    main_stat = result.get('main_correlation', 
                                         result.get('coefficient_of_variation',
                                                   result.get('amplification_ratio', 0)))
                    threshold = result.get('threshold', 0)
                    
                    bars = ax.bar(['Observed', 'Threshold'], [main_stat, threshold], 
                                 color=['blue', 'red'], alpha=0.7)
                    ax.set_ylabel('Value')
                    ax.set_title(title)
                    
                    # Add value labels
                    for bar, val in zip(bars, [main_stat, threshold]):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.05,
                               f'{val:.3f}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'Test not completed', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(self.validation_dir / 'comprehensive_validation_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Validation visualizations created")

def main():
    """
    Main execution function for theoretical-empirical validation
    """
    print("=" * 80)
    print("THEORETICAL-EMPIRICAL VALIDATION FRAMEWORK")
    print("Advanced Testing of AI-Driven Spatial Distribution Theory")
    print("=" * 80)
    
    # Initialize validator
    validator = TheoreticalEmpiricalValidator()
    
    # Run comprehensive validation
    validation_results = validator.run_comprehensive_validation()
    
    # Create visualizations
    validator.create_validation_visualizations(validation_results)
    
    # Print summary
    print("\nVALIDATION COMPLETED!")
    print("-" * 50)
    
    summary = validation_results['summary']
    print(f"Hypotheses tested: {summary['total_hypotheses_tested']}")
    print(f"Hypotheses supported: {summary['hypotheses_supported']}")
    print(f"Support rate: {summary['support_rate']:.1%}")
    print(f"Theoretical model preferred: {summary['theoretical_model_preferred']}")
    print(f"Overall validation success: {summary['overall_validation_success']}")
    
    print("\nIndividual Hypothesis Results:")
    for hypothesis in ['ai_concentration', 'heterogeneous_returns', 'network_amplification',
                      'dynamic_divergence', 'virtual_agglomeration', 'complementarity']:
        if hypothesis in validation_results:
            result = validation_results[hypothesis]
            status = "✅ SUPPORTED" if result.get('prediction_supported', False) else "❌ NOT SUPPORTED"
            significance = "📊 SIGNIFICANT" if result.get('statistical_significance', False) else "📊 NOT SIGNIFICANT"
            print(f"  {hypothesis.replace('_', ' ').title()}: {status} | {significance}")
    
    if 'structural_parameters' in validation_results:
        print(f"\nStructural Parameter Estimation:")
        params = validation_results['structural_parameters']
        print(f"  α_AI (AI productivity): {params['alpha_ai']:.3f}")
        print(f"  β_learning (spillovers): {params['beta_learning']:.3f}")
        print(f"  γ_network (network effects): {params['gamma_network']:.3f}")
        print(f"  δ_digital (infrastructure): {params['delta_digital']:.3f}")
        print(f"  φ_distance (distance decay): {params['phi_distance']:.3f}")
    
    print(f"\nResults saved to: results/theoretical_validation/")
    print("=" * 80)
    
    return validation_results

if __name__ == "__main__":
    results = main()
