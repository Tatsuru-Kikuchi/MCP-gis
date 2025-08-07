"""
Advanced Identification Strategy for Spatial AI Effects
Addresses JUE Editor Critique #2: Identification challenges with spatial spillovers and GE effects

This module implements:
1. Pre-treatment trend analysis
2. Spatial placebo tests
3. Decomposition of local shocks vs. agglomeration forces
4. General equilibrium adjustment methods
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SpatialIdentification:
    """
    Advanced identification strategies for spatial economic analysis
    Addresses editor's concerns about distinguishing agglomeration from correlated shocks
    """
    
    def __init__(self, data, geometry, time_periods):
        """
        Initialize with panel data and spatial structure
        
        Args:
            data: DataFrame with panel structure (location x time)
            geometry: GeoDataFrame with spatial units
            time_periods: List of time periods
        """
        self.data = data
        self.geometry = geometry
        self.time_periods = time_periods
        self.spatial_weights = self._create_spatial_weights()
        
    def _create_spatial_weights(self):
        """Create spatial weights matrix"""
        coords = np.array([[geom.centroid.x, geom.centroid.y] 
                          for geom in self.geometry.geometry])
        dist_matrix = distance_matrix(coords, coords)
        
        # Inverse distance weights with cutoff
        W = np.zeros_like(dist_matrix)
        cutoff = np.percentile(dist_matrix[dist_matrix > 0], 25)
        
        for i in range(len(dist_matrix)):
            for j in range(len(dist_matrix)):
                if i != j and dist_matrix[i, j] <= cutoff:
                    W[i, j] = 1 / dist_matrix[i, j]
        
        # Row normalize
        W = W / W.sum(axis=1, keepdims=True)
        return W
    
    def test_pre_trends(self, treatment_time, outcome_var, treatment_var):
        """
        Test for differential pre-treatment trends
        Critical for identifying causal effects vs. correlated shocks
        """
        pre_periods = [t for t in self.time_periods if t < treatment_time]
        
        if len(pre_periods) < 3:
            raise ValueError("Need at least 3 pre-treatment periods for trend analysis")
        
        results = {
            'parallel_trends_test': None,
            'location_specific_trends': {},
            'spatial_correlation_trends': []
        }
        
        # Test 1: Overall parallel trends
        treated_units = self.data[self.data[treatment_var] == 1]['location_id'].unique()
        control_units = self.data[self.data[treatment_var] == 0]['location_id'].unique()
        
        treated_trends = []
        control_trends = []
        
        for t in pre_periods:
            treated_mean = self.data[
                (self.data['time'] == t) & 
                (self.data['location_id'].isin(treated_units))
            ][outcome_var].mean()
            
            control_mean = self.data[
                (self.data['time'] == t) & 
                (self.data['location_id'].isin(control_units))
            ][outcome_var].mean()
            
            treated_trends.append(treated_mean)
            control_trends.append(control_mean)
        
        # Test if trends are parallel (interaction term should be zero)
        from scipy.stats import linregress
        time_index = np.arange(len(pre_periods))
        
        treated_slope = linregress(time_index, treated_trends).slope
        control_slope = linregress(time_index, control_trends).slope
        
        # Bootstrap test for difference in slopes
        slope_diff = treated_slope - control_slope
        bootstrap_diffs = []
        
        for _ in range(1000):
            # Resample locations
            boot_treated = np.random.choice(treated_units, len(treated_units), replace=True)
            boot_control = np.random.choice(control_units, len(control_units), replace=True)
            
            boot_treated_trend = []
            boot_control_trend = []
            
            for t in pre_periods:
                boot_treated_mean = self.data[
                    (self.data['time'] == t) & 
                    (self.data['location_id'].isin(boot_treated))
                ][outcome_var].mean()
                
                boot_control_mean = self.data[
                    (self.data['time'] == t) & 
                    (self.data['location_id'].isin(boot_control))
                ][outcome_var].mean()
                
                boot_treated_trend.append(boot_treated_mean)
                boot_control_trend.append(boot_control_mean)
            
            boot_treated_slope = linregress(time_index, boot_treated_trend).slope
            boot_control_slope = linregress(time_index, boot_control_trend).slope
            bootstrap_diffs.append(boot_treated_slope - boot_control_slope)
        
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(slope_diff))
        
        results['parallel_trends_test'] = {
            'treated_slope': treated_slope,
            'control_slope': control_slope,
            'difference': slope_diff,
            'p_value': p_value,
            'reject_parallel': p_value < 0.05
        }
        
        # Test 2: Spatial correlation of trends
        for t_idx in range(1, len(pre_periods)):
            growth = []
            for loc in self.data['location_id'].unique():
                y_t = self.data[
                    (self.data['time'] == pre_periods[t_idx]) & 
                    (self.data['location_id'] == loc)
                ][outcome_var].values
                
                y_t_1 = self.data[
                    (self.data['time'] == pre_periods[t_idx-1]) & 
                    (self.data['location_id'] == loc)
                ][outcome_var].values
                
                if len(y_t) > 0 and len(y_t_1) > 0:
                    growth.append(y_t[0] - y_t_1[0])
                else:
                    growth.append(np.nan)
            
            # Moran's I for growth rates
            growth = np.array(growth)
            valid_idx = ~np.isnan(growth)
            
            if valid_idx.sum() > 0:
                I = self._morans_i(growth[valid_idx], 
                                  self.spatial_weights[valid_idx][:, valid_idx])
                results['spatial_correlation_trends'].append({
                    'period': f"{pre_periods[t_idx-1]}-{pre_periods[t_idx]}",
                    'morans_i': I['I'],
                    'p_value': I['p_value']
                })
        
        return results
    
    def spatial_placebo_test(self, treatment_var, outcome_var, n_placebos=100):
        """
        Spatial placebo test: randomly reassign treatment spatially
        Tests if observed effects could arise from random spatial patterns
        """
        actual_effect = self._estimate_spatial_effect(treatment_var, outcome_var)
        
        placebo_effects = []
        n_treated = self.data[treatment_var].sum()
        all_locations = self.data['location_id'].unique()
        
        for _ in range(n_placebos):
            # Randomly assign treatment to same number of locations
            placebo_treated = np.random.choice(all_locations, 
                                              size=int(n_treated), 
                                              replace=False)
            
            # Create placebo treatment variable
            self.data['placebo_treatment'] = self.data['location_id'].isin(placebo_treated).astype(int)
            
            # Estimate placebo effect
            placebo_effect = self._estimate_spatial_effect('placebo_treatment', outcome_var)
            placebo_effects.append(placebo_effect)
        
        # Clean up
        del self.data['placebo_treatment']
        
        # Calculate p-value
        p_value = np.mean(np.abs(placebo_effects) >= np.abs(actual_effect))
        
        return {
            'actual_effect': actual_effect,
            'placebo_distribution': placebo_effects,
            'p_value': p_value,
            'percentile': stats.percentileofscore(placebo_effects, actual_effect)
        }
    
    def decompose_local_shocks_vs_spillovers(self, outcome_var, treatment_var, 
                                            shock_vars=None):
        """
        Decompose effects into local shocks vs. spatial spillovers
        Addresses editor's concern about distinguishing mechanisms
        """
        if shock_vars is None:
            shock_vars = ['demand_shock', 'cost_shock', 'productivity_shock']
        
        results = {
            'total_effect': None,
            'direct_local': None,
            'spatial_spillover': None,
            'correlated_shocks': None,
            'decomposition': {}
        }
        
        # Step 1: Estimate total effect
        total_effect = self._estimate_spatial_effect(treatment_var, outcome_var)
        results['total_effect'] = total_effect
        
        # Step 2: Control for local shocks
        available_shocks = [s for s in shock_vars if s in self.data.columns]
        
        if available_shocks:
            # Partial out local shocks
            from sklearn.linear_model import LinearRegression
            
            X_shocks = self.data[available_shocks].values
            y = self.data[outcome_var].values
            treatment = self.data[treatment_var].values
            
            # Residualize outcome and treatment
            reg_y = LinearRegression().fit(X_shocks, y)
            y_resid = y - reg_y.predict(X_shocks)
            
            reg_t = LinearRegression().fit(X_shocks, treatment)
            t_resid = treatment - reg_t.predict(X_shocks)
            
            # Effect after controlling for shocks
            reg_controlled = LinearRegression().fit(t_resid.reshape(-1, 1), y_resid)
            direct_effect = reg_controlled.coef_[0]
            
            results['direct_local'] = direct_effect
            results['correlated_shocks'] = total_effect - direct_effect
        
        # Step 3: Spatial spillover decomposition
        # Create spatial lag of treatment
        location_treatment = self.data.groupby('location_id')[treatment_var].mean()
        n_locs = len(location_treatment)
        treatment_vector = location_treatment.values
        
        # Spatial lag
        spatial_lag = self.spatial_weights @ treatment_vector
        
        # Estimate with both own and neighbor treatment
        X = np.column_stack([treatment_vector, spatial_lag])
        y_by_loc = self.data.groupby('location_id')[outcome_var].mean().values
        
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.1).fit(X, y_by_loc)
        
        results['decomposition'] = {
            'own_effect': model.coef_[0],
            'spillover_effect': model.coef_[1],
            'ratio_spillover': model.coef_[1] / (model.coef_[0] + model.coef_[1])
            if (model.coef_[0] + model.coef_[1]) != 0 else 0
        }
        
        return results
    
    def general_equilibrium_adjustment(self, partial_effect, market_vars):
        """
        Adjust partial equilibrium estimates for GE effects
        Addresses editor's concern about GE rendering DID estimates uninterpretable
        """
        # Implement market clearing conditions
        results = {
            'partial_equilibrium': partial_effect,
            'general_equilibrium': None,
            'adjustment_factor': None,
            'market_feedback': {}
        }
        
        # Step 1: Estimate market feedback effects
        if 'price' in market_vars and 'quantity' in market_vars:
            # Estimate price elasticity
            price = self.data[market_vars['price']].values
            quantity = self.data[market_vars['quantity']].values
            
            # Log-log regression for elasticity
            log_p = np.log(price + 1e-10)
            log_q = np.log(quantity + 1e-10)
            
            elasticity = np.cov(log_q, log_p)[0, 1] / np.var(log_p)
            
            results['market_feedback']['elasticity'] = elasticity
            
            # GE adjustment based on market size
            market_share = quantity / quantity.sum()
            avg_market_share = market_share.mean()
            
            # Larger market share → larger GE effects
            ge_multiplier = 1 / (1 - avg_market_share * elasticity)
            
            results['general_equilibrium'] = partial_effect * ge_multiplier
            results['adjustment_factor'] = ge_multiplier
        
        # Step 2: Spatial GE effects (ripple effects)
        if self.spatial_weights is not None:
            # Calculate spatial multiplier
            # (I - ρW)^{-1} where ρ is spatial autocorrelation
            I = np.eye(len(self.spatial_weights))
            
            # Estimate spatial autocorrelation
            y = self.data.groupby('location_id')[market_vars.get('outcome', 'productivity')].mean().values
            rho = self._estimate_spatial_autocorrelation(y)
            
            # Spatial multiplier matrix
            try:
                spatial_multiplier = np.linalg.inv(I - rho * self.spatial_weights)
                avg_multiplier = spatial_multiplier.mean()
                
                results['market_feedback']['spatial_multiplier'] = avg_multiplier
                results['general_equilibrium'] = partial_effect * avg_multiplier
                
            except np.linalg.LinAlgError:
                results['market_feedback']['spatial_multiplier'] = 1.0
        
        return results
    
    def heterogeneous_exposure_iv(self, treatment_var, outcome_var, instrument_var):
        """
        Use heterogeneous exposure to AI as instrument
        Addresses endogeneity of AI adoption
        """
        from sklearn.linear_model import LinearRegression
        from scipy import stats
        
        # First stage: Treatment on instrument
        X_iv = self.data[instrument_var].values.reshape(-1, 1)
        treatment = self.data[treatment_var].values
        
        first_stage = LinearRegression().fit(X_iv, treatment)
        treatment_hat = first_stage.predict(X_iv)
        
        # Test instrument strength
        f_stat = (first_stage.score(X_iv, treatment) * len(treatment)) / (1 - first_stage.score(X_iv, treatment))
        
        # Second stage: Outcome on predicted treatment
        outcome = self.data[outcome_var].values
        second_stage = LinearRegression().fit(treatment_hat.reshape(-1, 1), outcome)
        
        iv_effect = second_stage.coef_[0]
        
        # Bootstrap standard errors
        n_boot = 1000
        boot_effects = []
        
        for _ in range(n_boot):
            idx = np.random.choice(len(self.data), len(self.data), replace=True)
            
            # First stage with bootstrap sample
            X_boot = X_iv[idx]
            t_boot = treatment[idx]
            y_boot = outcome[idx]
            
            fs_boot = LinearRegression().fit(X_boot, t_boot)
            t_hat_boot = fs_boot.predict(X_boot)
            
            # Second stage
            ss_boot = LinearRegression().fit(t_hat_boot.reshape(-1, 1), y_boot)
            boot_effects.append(ss_boot.coef_[0])
        
        se = np.std(boot_effects)
        ci = np.percentile(boot_effects, [2.5, 97.5])
        
        return {
            'iv_estimate': iv_effect,
            'standard_error': se,
            'confidence_interval': ci,
            'first_stage_f': f_stat,
            'weak_instrument': f_stat < 10,
            'reduced_form': LinearRegression().fit(X_iv, outcome).coef_[0]
        }
    
    def synthetic_control_spatial(self, treated_location, outcome_var, 
                                 pre_periods, post_periods):
        """
        Spatial synthetic control method
        Creates synthetic version of treated location using spatial neighbors
        """
        from scipy.optimize import minimize
        
        # Get pre-treatment outcomes for treated unit
        treated_pre = []
        for t in pre_periods:
            val = self.data[
                (self.data['location_id'] == treated_location) & 
                (self.data['time'] == t)
            ][outcome_var].values
            if len(val) > 0:
                treated_pre.append(val[0])
        
        treated_pre = np.array(treated_pre)
        
        # Get potential control units (spatial neighbors)
        control_locations = [loc for loc in self.data['location_id'].unique() 
                           if loc != treated_location]
        
        # Pre-treatment outcomes for controls
        control_pre = []
        for loc in control_locations:
            loc_pre = []
            for t in pre_periods:
                val = self.data[
                    (self.data['location_id'] == loc) & 
                    (self.data['time'] == t)
                ][outcome_var].values
                if len(val) > 0:
                    loc_pre.append(val[0])
            if len(loc_pre) == len(pre_periods):
                control_pre.append(loc_pre)
        
        control_pre = np.array(control_pre).T  # Time x Units
        
        # Find optimal weights
        def loss(w):
            synthetic = control_pre @ w
            return np.sum((treated_pre - synthetic) ** 2)
        
        # Constraints: weights sum to 1, all non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(control_pre.shape[1])]
        
        result = minimize(loss, 
                         np.ones(control_pre.shape[1]) / control_pre.shape[1],
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        
        optimal_weights = result.x
        
        # Calculate treatment effect in post-period
        effects = []
        for t in post_periods:
            # Actual outcome
            actual = self.data[
                (self.data['location_id'] == treated_location) & 
                (self.data['time'] == t)
            ][outcome_var].values
            
            if len(actual) == 0:
                continue
                
            # Synthetic outcome
            synthetic = 0
            for i, loc in enumerate(control_locations[:len(optimal_weights)]):
                loc_outcome = self.data[
                    (self.data['location_id'] == loc) & 
                    (self.data['time'] == t)
                ][outcome_var].values
                
                if len(loc_outcome) > 0:
                    synthetic += optimal_weights[i] * loc_outcome[0]
            
            effects.append(actual[0] - synthetic)
        
        return {
            'weights': optimal_weights,
            'pre_period_fit': result.fun,
            'treatment_effects': effects,
            'average_effect': np.mean(effects) if effects else None
        }
    
    def _estimate_spatial_effect(self, treatment_var, outcome_var):
        """Helper: Estimate basic spatial effect"""
        treated = self.data[self.data[treatment_var] == 1][outcome_var].mean()
        control = self.data[self.data[treatment_var] == 0][outcome_var].mean()
        return treated - control
    
    def _morans_i(self, values, weights):
        """Calculate Moran's I statistic"""
        n = len(values)
        x_bar = values.mean()
        
        # Calculate numerator and denominator
        num = 0
        denom = 0
        W_sum = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    num += weights[i, j] * (values[i] - x_bar) * (values[j] - x_bar)
                    W_sum += weights[i, j]
            denom += (values[i] - x_bar) ** 2
        
        I = (n / W_sum) * (num / denom) if denom != 0 else 0
        
        # Expected value and variance under null
        E_I = -1 / (n - 1)
        
        # Simplified variance (under normality)
        b2 = np.sum((values - x_bar) ** 4) / n / (np.sum((values - x_bar) ** 2) / n) ** 2
        
        var_I = ((n * ((n**2 - 3*n + 3) - (n - 1) * b2)) / 
                ((n - 1) * (n - 2) * (n - 3) * W_sum ** 2))
        
        # Z-score and p-value
        z = (I - E_I) / np.sqrt(var_I) if var_I > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
        
        return {'I': I, 'E_I': E_I, 'var': var_I, 'z': z, 'p_value': p_value}
    
    def _estimate_spatial_autocorrelation(self, y):
        """Estimate spatial autocorrelation coefficient"""
        n = len(y)
        y_lag = self.spatial_weights @ y
        
        # OLS estimate of rho
        rho = np.dot(y_lag, y) / np.dot(y_lag, y_lag) if np.dot(y_lag, y_lag) > 0 else 0
        
        # Bound rho to ensure stability
        max_eigenvalue = np.max(np.abs(np.linalg.eigvals(self.spatial_weights)))
        if max_eigenvalue > 0:
            rho = np.clip(rho, -0.99/max_eigenvalue, 0.99/max_eigenvalue)
        
        return rho


def run_comprehensive_identification(data_path, geometry_path, 
                                    treatment_col='ai_adoption',
                                    outcome_col='productivity'):
    """
    Run comprehensive identification analysis addressing all editor concerns
    """
    print("="*80)
    print("COMPREHENSIVE IDENTIFICATION ANALYSIS")
    print("Addressing JUE Editor Critique #2")
    print("="*80)
    
    # Load data
    data = pd.read_csv(data_path)
    geometry = gpd.read_file(geometry_path)
    
    # Initialize identification framework
    time_periods = sorted(data['time'].unique())
    identifier = SpatialIdentification(data, geometry, time_periods)
    
    results = {}
    
    # 1. Pre-treatment trends analysis
    print("\n1. Testing Pre-Treatment Trends...")
    treatment_time = time_periods[len(time_periods)//2]  # Assume treatment at midpoint
    
    pre_trends = identifier.test_pre_trends(
        treatment_time, outcome_col, treatment_col
    )
    
    print(f"   Parallel trends p-value: {pre_trends['parallel_trends_test']['p_value']:.4f}")
    if pre_trends['parallel_trends_test']['reject_parallel']:
        print("   ⚠️  WARNING: Evidence of differential pre-trends")
        print("   → AI adoption may respond to local trends")
    else:
        print("   ✓ No evidence of differential pre-trends")
    
    results['pre_trends'] = pre_trends
    
    # 2. Spatial placebo test
    print("\n2. Running Spatial Placebo Tests...")
    placebo = identifier.spatial_placebo_test(treatment_col, outcome_col)
    
    print(f"   Actual effect: {placebo['actual_effect']:.4f}")
    print(f"   Placebo p-value: {placebo['p_value']:.4f}")
    print(f"   Effect percentile: {placebo['percentile']:.1f}")
    
    if placebo['p_value'] < 0.05:
        print("   ✓ Effect unlikely due to random spatial pattern")
    else:
        print("   ⚠️  Effect could arise from random spatial assignment")
    
    results['placebo_test'] = placebo
    
    # 3. Decompose local shocks vs spillovers
    print("\n3. Decomposing Local Shocks vs. Spillovers...")
    decomposition = identifier.decompose_local_shocks_vs_spillovers(
        outcome_col, treatment_col
    )
    
    print(f"   Total effect: {decomposition['total_effect']:.4f}")
    if decomposition['direct_local'] is not None:
        print(f"   Direct local effect: {decomposition['direct_local']:.4f}")
        print(f"   Correlated shocks: {decomposition['correlated_shocks']:.4f}")
    
    if 'own_effect' in decomposition['decomposition']:
        print(f"   Own location effect: {decomposition['decomposition']['own_effect']:.4f}")
        print(f"   Spillover effect: {decomposition['decomposition']['spillover_effect']:.4f}")
        print(f"   Spillover share: {decomposition['decomposition']['ratio_spillover']:.1%}")
    
    results['decomposition'] = decomposition
    
    # 4. General equilibrium adjustment
    print("\n4. General Equilibrium Adjustment...")
    market_vars = {
        'price': 'land_price' if 'land_price' in data.columns else 'price',
        'quantity': 'employment' if 'employment' in data.columns else 'output',
        'outcome': outcome_col
    }
    
    ge_adjustment = identifier.general_equilibrium_adjustment(
        decomposition['total_effect'], market_vars
    )
    
    print(f"   Partial equilibrium effect: {ge_adjustment['partial_equilibrium']:.4f}")
    if ge_adjustment['general_equilibrium'] is not None:
        print(f"   General equilibrium effect: {ge_adjustment['general_equilibrium']:.4f}")
        print(f"   GE adjustment factor: {ge_adjustment['adjustment_factor']:.2f}x")
    
    results['ge_adjustment'] = ge_adjustment
    
    # 5. IV estimation if instrument available
    if 'broadband_speed' in data.columns or 'tech_infrastructure' in data.columns:
        print("\n5. Instrumental Variable Estimation...")
        instrument = 'broadband_speed' if 'broadband_speed' in data.columns else 'tech_infrastructure'
        
        iv_results = identifier.heterogeneous_exposure_iv(
            treatment_col, outcome_col, instrument
        )
        
        print(f"   IV estimate: {iv_results['iv_estimate']:.4f}")
        print(f"   Standard error: {iv_results['standard_error']:.4f}")
        print(f"   95% CI: [{iv_results['confidence_interval'][0]:.4f}, "
              f"{iv_results['confidence_interval'][1]:.4f}]")
        print(f"   First-stage F-stat: {iv_results['first_stage_f']:.2f}")
        
        if iv_results['weak_instrument']:
            print("   ⚠️  Warning: Weak instrument (F < 10)")
        
        results['iv_estimation'] = iv_results
    
    # 6. Synthetic control for selected locations
    print("\n6. Spatial Synthetic Control...")
    treated_locations = data[data[treatment_col] == 1]['location_id'].unique()[:3]
    
    synthetic_results = []
    for loc in treated_locations:
        pre_periods = time_periods[:len(time_periods)//2]
        post_periods = time_periods[len(time_periods)//2:]
        
        sc_result = identifier.synthetic_control_spatial(
            loc, outcome_col, pre_periods, post_periods
        )
        
        if sc_result['average_effect'] is not None:
            synthetic_results.append(sc_result['average_effect'])
            print(f"   Location {loc}: Effect = {sc_result['average_effect']:.4f}")
    
    if synthetic_results:
        print(f"   Average synthetic control effect: {np.mean(synthetic_results):.4f}")
        results['synthetic_control'] = {
            'location_effects': synthetic_results,
            'average': np.mean(synthetic_results)
        }
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("IDENTIFICATION SUMMARY")
    print("="*80)
    
    concerns = []
    strengths = []
    
    # Check each test
    if pre_trends['parallel_trends_test']['reject_parallel']:
        concerns.append("Differential pre-trends detected")
    else:
        strengths.append("Parallel pre-trends supported")
    
    if placebo['p_value'] >= 0.05:
        concerns.append("Effect not distinguishable from random spatial pattern")
    else:
        strengths.append("Effect robust to spatial placebo tests")
    
    if decomposition['decomposition'].get('ratio_spillover', 0) > 0.5:
        concerns.append("Majority of effect from spillovers (GE concerns)")
    else:
        strengths.append("Direct effects dominate spillovers")
    
    print("\n✓ Strengths:")
    for s in strengths:
        print(f"  - {s}")
    
    if concerns:
        print("\n⚠️ Concerns to address:")
        for c in concerns:
            print(f"  - {c}")
    
    print("\nRECOMMENDATIONS:")
    print("1. Report all identification tests transparently")
    print("2. Use IV estimates as primary specification if instrument strong")
    print("3. Show robustness across multiple identification strategies")
    print("4. Acknowledge GE effects and report adjusted estimates")
    print("5. Consider spatial synthetic control for key locations")
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 2:
        data_path = sys.argv[1]
        geometry_path = sys.argv[2]
    else:
        # Generate example data
        print("Generating example data for demonstration...")
        
        # Create synthetic panel data
        n_locations = 50
        n_periods = 10
        
        np.random.seed(42)
        
        data = []
        for loc in range(n_locations):
            for t in range(n_periods):
                # Treatment starts at period 5
                treatment = 1 if (t >= 5 and loc < 20) else 0
                
                # Outcome with treatment effect and spatial correlation
                base = np.random.normal(10, 2)
                trend = 0.5 * t
                treatment_effect = 3 * treatment
                noise = np.random.normal(0, 1)
                
                outcome = base + trend + treatment_effect + noise
                
                data.append({
                    'location_id': loc,
                    'time': t,
                    'ai_adoption': treatment,
                    'productivity': outcome,
                    'demand_shock': np.random.normal(0, 1),
                    'cost_shock': np.random.normal(0, 1),
                    'broadband_speed': np.random.uniform(10, 100)
                })
        
        df = pd.DataFrame(data)
        df.to_csv('temp_data.csv', index=False)
        
        # Create dummy geometry
        from shapely.geometry import Point
        points = [Point(np.random.uniform(-10, 10), np.random.uniform(-10, 10)) 
                 for _ in range(n_locations)]
        gdf = gpd.GeoDataFrame({'location_id': range(n_locations)}, geometry=points)
        gdf.to_file('temp_geometry.geojson', driver='GeoJSON')
        
        # Run analysis
        results = run_comprehensive_identification('temp_data.csv', 'temp_geometry.geojson')
        
        # Clean up
        import os
        os.remove('temp_data.csv')
        os.remove('temp_geometry.geojson')
