#!/usr/bin/env python3
"""
Temporal Agglomeration Analyzer
Analyzes how agglomeration effects change over time with demographic transitions

Key Features:
- Time-varying agglomeration coefficients
- Demographic transition impacts
- Economic shock effects
- Spatial reorganization patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class TemporalAgglomerationAnalyzer:
    """
    Analyzes temporal changes in agglomeration effects
    """
    
    def __init__(self, data_dir: str = "data", demographic_dir: str = "data/demographic", 
                 output_dir: str = "results/temporal"):
        self.data_dir = Path(data_dir)
        self.demographic_dir = Path(demographic_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.time_windows = {
            'pre_crisis': (2000, 2007),
            'financial_crisis': (2008, 2010),
            'recovery': (2011, 2019),
            'covid_era': (2020, 2023)
        }
        
        # Demographic transition phases
        self.demographic_phases = {
            'early_aging': (2000, 2010),
            'accelerated_aging': (2011, 2020),
            'super_aging': (2021, 2030)
        }
        
        # Load data containers
        self.population_data = None
        self.labor_data = None
        self.migration_data = None
        self.productivity_data = None
        self.shock_data = None
        
    def load_temporal_data(self):
        """
        Load all temporal datasets
        """
        try:
            # Demographic data
            self.population_data = pd.read_csv(self.demographic_dir / "historical_population_by_age.csv")
            self.labor_data = pd.read_csv(self.demographic_dir / "labor_force_participation.csv")
            self.migration_data = pd.read_csv(self.demographic_dir / "migration_patterns.csv")
            self.productivity_data = pd.read_csv(self.demographic_dir / "productivity_aging_effects.csv")
            self.shock_data = pd.read_csv(self.demographic_dir / "economic_shock_events.csv")
            
            # Original static data (if available)
            try:
                self.establishments_data = pd.read_csv(self.data_dir / "tokyo_establishments.csv")
                self.spatial_data = pd.read_csv(self.data_dir / "tokyo_spatial_distribution.csv")
            except FileNotFoundError:
                self.logger.warning("Static data not found, generating sample data")
                self._generate_sample_static_data()
            
            self.logger.info("All temporal datasets loaded successfully")
        except FileNotFoundError as e:
            self.logger.error(f"Required data file not found: {e}")
            raise
    
    def _generate_sample_static_data(self):
        """
        Generate sample static data if not available
        """
        # Simple sample data for demonstration
        tokyo_wards = [
            'Chiyoda', 'Chuo', 'Minato', 'Shinjuku', 'Bunkyo', 'Taito',
            'Sumida', 'Koto', 'Shinagawa', 'Meguro', 'Ota', 'Setagaya',
            'Shibuya', 'Nakano', 'Suginami', 'Toshima', 'Kita', 'Arakawa',
            'Itabashi', 'Nerima', 'Adachi', 'Katsushika', 'Edogawa'
        ]
        
        # Create spatial data
        spatial_data = []
        for i, ward in enumerate(tokyo_wards):
            spatial_data.append({
                'ward': ward,
                'latitude': 35.6762 + np.random.uniform(-0.1, 0.1),
                'longitude': 139.6503 + np.random.uniform(-0.15, 0.15),
                'area_km2': np.random.uniform(10, 60),
                'ward_type': 'central' if ward in ['Chiyoda', 'Chuo', 'Minato'] else 'outer'
            })
        
        self.spatial_data = pd.DataFrame(spatial_data)
    
    def calculate_temporal_concentration_indices(self) -> pd.DataFrame:
        """
        Calculate how concentration indices change over time
        """
        self.logger.info("Calculating temporal concentration indices...")
        
        results = []
        
        # Aggregate population by industry and ward over time
        for year in range(2000, 2024):
            year_pop = self.population_data[self.population_data['year'] == year]
            year_labor = self.labor_data[self.labor_data['year'] == year]
            
            # Calculate workforce by industry and ward
            for industry_code in year_labor['industry_code'].unique():
                industry_labor = year_labor[year_labor['industry_code'] == industry_code]
                
                ward_employment = []
                for ward in self.spatial_data['ward'].unique():
                    # Calculate employment for this ward-industry-year
                    ward_pop = year_pop[year_pop['ward'] == ward]
                    
                    total_workforce = 0
                    for _, labor_row in industry_labor.iterrows():
                        age_group = labor_row['age_group']
                        participation_rate = labor_row['participation_rate']
                        
                        age_pop = ward_pop[ward_pop['age_group'] == age_group]
                        if not age_pop.empty:
                            workforce = age_pop['population'].iloc[0] * participation_rate
                            total_workforce += workforce
                    
                    ward_employment.append(total_workforce)
                
                # Calculate concentration indices
                employment_array = np.array(ward_employment)
                
                if employment_array.sum() > 0:
                    # Gini coefficient
                    gini = self._calculate_gini_coefficient(employment_array)
                    
                    # Herfindahl index
                    shares = employment_array / employment_array.sum()
                    hhi = np.sum(shares ** 2)
                    
                    # Coefficient of variation
                    cv = np.std(employment_array) / np.mean(employment_array) if np.mean(employment_array) > 0 else 0
                    
                    results.append({
                        'year': year,
                        'industry_code': industry_code,
                        'industry_name': industry_labor['industry_name'].iloc[0],
                        'gini_coefficient': gini,
                        'herfindahl_index': hhi,
                        'coefficient_variation': cv,
                        'total_employment': employment_array.sum(),
                        'n_wards_with_employment': np.sum(employment_array > 0)
                    })
        
        concentration_df = pd.DataFrame(results)
        
        # Save results
        output_path = self.output_dir / "temporal_concentration_indices.csv"
        concentration_df.to_csv(output_path, index=False)
        self.logger.info(f"Temporal concentration indices saved to {output_path}")
        
        return concentration_df
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient
        """
        if len(values) == 0 or np.sum(values) == 0:
            return 0
        
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        if cumsum[-1] == 0:
            return 0
        
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
        return max(0, gini)
    
    def analyze_demographic_transition_effects(self) -> pd.DataFrame:
        """
        Analyze how demographic transitions affect agglomeration patterns
        """
        self.logger.info("Analyzing demographic transition effects...")
        
        results = []
        
        for phase_name, (start_year, end_year) in self.demographic_phases.items():
            phase_years = [y for y in range(start_year, min(end_year + 1, 2024))]
            
            for industry_code in self.labor_data['industry_code'].unique():
                # Calculate demographic composition for this phase
                phase_labor = self.labor_data[
                    (self.labor_data['year'].isin(phase_years)) & 
                    (self.labor_data['industry_code'] == industry_code)
                ]
                
                # Age composition metrics
                age_composition = phase_labor.groupby('age_group')['participation_rate'].mean()
                
                # Dependency ratio (elderly / working age)
                elderly_participation = age_composition.get('elderly', 0)
                working_age_participation = age_composition.get('prime', 0) + age_composition.get('mature', 0)
                dependency_ratio = elderly_participation / max(working_age_participation, 0.001)
                
                # Youth ratio
                youth_ratio = age_composition.get('young', 0) / age_composition.sum()
                
                # Migration attractiveness for young workers
                phase_migration = self.migration_data[
                    (self.migration_data['year'].isin(phase_years)) & 
                    (self.migration_data['age_group'] == 'young')
                ]
                
                avg_youth_migration = phase_migration.groupby('ward_type')['net_migration_rate'].mean()
                central_youth_attraction = avg_youth_migration.get('central', 0)
                
                # Productivity effects
                phase_productivity = self.productivity_data[
                    (self.productivity_data['year'].isin(phase_years)) & 
                    (self.productivity_data['industry_code'] == industry_code)
                ]
                
                avg_tech_boost = phase_productivity['tech_adoption_boost'].mean()
                productivity_variance = phase_productivity.groupby('age_group')['adjusted_productivity'].std().mean()
                
                results.append({
                    'demographic_phase': phase_name,
                    'start_year': start_year,
                    'end_year': min(end_year, 2023),
                    'industry_code': industry_code,
                    'industry_name': phase_labor['industry_name'].iloc[0] if not phase_labor.empty else '',
                    'dependency_ratio': dependency_ratio,
                    'youth_ratio': youth_ratio,
                    'central_youth_attraction': central_youth_attraction,
                    'avg_tech_boost': avg_tech_boost,
                    'productivity_variance': productivity_variance,
                    'phase_duration': len(phase_years)
                })
        
        demographic_effects_df = pd.DataFrame(results)
        
        # Save results
        output_path = self.output_dir / "demographic_transition_effects.csv"
        demographic_effects_df.to_csv(output_path, index=False)
        self.logger.info(f"Demographic transition effects saved to {output_path}")
        
        return demographic_effects_df
    
    def analyze_economic_shock_responses(self) -> pd.DataFrame:
        """
        Analyze how agglomeration patterns respond to economic shocks
        """
        self.logger.info("Analyzing economic shock responses...")
        
        results = []
        
        # Get baseline (pre-shock) metrics
        baseline_years = list(range(2005, 2008))  # Pre-financial crisis
        
        for _, shock_row in self.shock_data.iterrows():
            year = shock_row['year']
            event = shock_row['event']
            industry_code = shock_row['industry_code']
            impact_multiplier = shock_row['impact_multiplier']
            
            # Compare with baseline
            baseline_labor = self.labor_data[
                (self.labor_data['year'].isin(baseline_years)) & 
                (self.labor_data['industry_code'] == industry_code)
            ]
            
            shock_labor = self.labor_data[
                (self.labor_data['year'] == year) & 
                (self.labor_data['industry_code'] == industry_code)
            ]
            
            if not baseline_labor.empty and not shock_labor.empty:
                # Calculate change in labor participation patterns
                baseline_participation = baseline_labor.groupby('age_group')['participation_rate'].mean()
                shock_participation = shock_labor.groupby('age_group')['participation_rate'].mean()
                
                participation_changes = {}
                for age_group in baseline_participation.index:
                    baseline_rate = baseline_participation[age_group]
                    shock_rate = shock_participation.get(age_group, baseline_rate)
                    participation_changes[f'{age_group}_change'] = (shock_rate - baseline_rate) / baseline_rate
                
                # Migration response
                shock_migration = self.migration_data[
                    (self.migration_data['year'] == year)
                ]
                
                migration_volatility = shock_migration['net_migration_rate'].std()
                
                # Productivity impact
                shock_productivity = self.productivity_data[
                    (self.productivity_data['year'] == year) & 
                    (self.productivity_data['industry_code'] == industry_code)
                ]
                
                avg_productivity_impact = shock_productivity['adjusted_productivity'].mean() if not shock_productivity.empty else 1.0
                
                result_row = {
                    'year': year,
                    'event': event,
                    'industry_code': industry_code,
                    'industry_name': shock_row['industry_name'],
                    'impact_multiplier': impact_multiplier,
                    'migration_volatility': migration_volatility,
                    'avg_productivity_impact': avg_productivity_impact
                }
                
                # Add participation changes
                result_row.update(participation_changes)
                
                results.append(result_row)
        
        shock_response_df = pd.DataFrame(results)
        
        # Save results
        output_path = self.output_dir / "economic_shock_responses.csv"
        shock_response_df.to_csv(output_path, index=False)
        self.logger.info(f"Economic shock responses saved to {output_path}")
        
        return shock_response_df
    
    def calculate_time_varying_agglomeration_coefficients(self) -> pd.DataFrame:
        """
        Calculate how agglomeration coefficients change over time
        """
        self.logger.info("Calculating time-varying agglomeration coefficients...")
        
        results = []
        
        # Rolling window analysis (5-year windows)
        window_size = 5
        
        for center_year in range(2002, 2022):  # 2000-2004 to 2018-2022
            window_years = list(range(center_year - window_size//2, center_year + window_size//2 + 1))
            window_years = [y for y in window_years if 2000 <= y <= 2023]
            
            for industry_code in self.labor_data['industry_code'].unique():
                # Aggregate data for this window
                window_data = self._aggregate_window_data(window_years, industry_code)
                
                if len(window_data) < 5:  # Need minimum data points
                    continue
                
                # Calculate agglomeration variables
                X = window_data[[
                    'employment_density', 'market_potential', 'diversity_index', 
                    'distance_to_center', 'avg_age', 'dependency_ratio'
                ]].copy()
                
                y = window_data['productivity_index']
                
                # Handle missing values
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())
                
                if X.isna().any().any() or y.isna().any():
                    continue
                
                # Standardize variables
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Fit model
                model = Ridge(alpha=0.1)  # Ridge regression for stability
                model.fit(X_scaled, y)
                
                r2 = model.score(X_scaled, y)
                
                # Store coefficients
                coeff_dict = {
                    'window_center_year': center_year,
                    'window_start': min(window_years),
                    'window_end': max(window_years),
                    'industry_code': industry_code,
                    'industry_name': self.labor_data[self.labor_data['industry_code'] == industry_code]['industry_name'].iloc[0],
                    'n_observations': len(window_data),
                    'r_squared': r2
                }
                
                # Add coefficients
                for i, col in enumerate(X.columns):
                    coeff_dict[f'{col}_coeff'] = model.coef_[i]
                
                results.append(coeff_dict)
        
        coefficients_df = pd.DataFrame(results)
        
        # Save results
        output_path = self.output_dir / "time_varying_agglomeration_coefficients.csv"
        coefficients_df.to_csv(output_path, index=False)
        self.logger.info(f"Time-varying agglomeration coefficients saved to {output_path}")
        
        return coefficients_df
    
    def _aggregate_window_data(self, years: List[int], industry_code: str) -> pd.DataFrame:
        """
        Aggregate data for a specific time window and industry
        """
        window_data = []
        
        for year in years:
            for ward in self.spatial_data['ward'].unique():
                # Employment data
                year_pop = self.population_data[self.population_data['year'] == year]
                year_labor = self.labor_data[
                    (self.labor_data['year'] == year) & 
                    (self.labor_data['industry_code'] == industry_code)
                ]
                
                ward_pop = year_pop[year_pop['ward'] == ward]
                
                if ward_pop.empty or year_labor.empty:
                    continue
                
                # Calculate employment and demographics
                total_employment = 0
                total_population = 0
                age_weighted_sum = 0
                elderly_pop = 0
                
                age_weights = {'young': 25, 'prime': 45, 'mature': 60, 'elderly': 75}
                
                for _, labor_row in year_labor.iterrows():
                    age_group = labor_row['age_group']
                    participation_rate = labor_row['participation_rate']
                    
                    age_pop_row = ward_pop[ward_pop['age_group'] == age_group]
                    if not age_pop_row.empty:
                        population = age_pop_row['population'].iloc[0]
                        employment = population * participation_rate
                        
                        total_employment += employment
                        total_population += population
                        age_weighted_sum += population * age_weights[age_group]
                        
                        if age_group == 'elderly':
                            elderly_pop += population
                
                if total_employment == 0 or total_population == 0:
                    continue
                
                # Ward characteristics
                ward_spatial = self.spatial_data[self.spatial_data['ward'] == ward].iloc[0]
                
                # Employment density
                employment_density = total_employment / ward_spatial['area_km2']
                
                # Market potential (simplified)
                distance_to_center = np.sqrt(
                    (ward_spatial['latitude'] - 35.6762)**2 + 
                    (ward_spatial['longitude'] - 139.6503)**2
                ) * 111
                
                market_potential = total_employment / max(distance_to_center, 0.1)
                
                # Diversity index (simplified - using employment variation)
                diversity_index = np.random.uniform(0.5, 2.0)  # Placeholder
                
                # Demographics
                avg_age = age_weighted_sum / total_population
                dependency_ratio = elderly_pop / max(total_population - elderly_pop, 1)
                
                # Productivity (from productivity data)
                productivity_rows = self.productivity_data[
                    (self.productivity_data['year'] == year) & 
                    (self.productivity_data['industry_code'] == industry_code)
                ]
                
                if not productivity_rows.empty:
                    productivity_index = productivity_rows['adjusted_productivity'].mean()
                else:
                    productivity_index = 1.0
                
                window_data.append({
                    'year': year,
                    'ward': ward,
                    'employment_density': employment_density,
                    'market_potential': market_potential,
                    'diversity_index': diversity_index,
                    'distance_to_center': distance_to_center,
                    'avg_age': avg_age,
                    'dependency_ratio': dependency_ratio,
                    'productivity_index': productivity_index,
                    'total_employment': total_employment
                })
        
        return pd.DataFrame(window_data)
    
    def run_full_temporal_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Run complete temporal agglomeration analysis
        """
        self.logger.info("Starting full temporal agglomeration analysis...")
        
        # Load data
        self.load_temporal_data()
        
        # Run all analyses
        results = {
            'temporal_concentration': self.calculate_temporal_concentration_indices(),
            'demographic_effects': self.analyze_demographic_transition_effects(),
            'shock_responses': self.analyze_economic_shock_responses(),
            'time_varying_coefficients': self.calculate_time_varying_agglomeration_coefficients()
        }
        
        # Create comprehensive summary
        self._create_temporal_summary(results)
        
        self.logger.info("Temporal agglomeration analysis completed successfully!")
        return results
    
    def _create_temporal_summary(self, results: Dict[str, pd.DataFrame]):
        """
        Create summary of temporal analysis
        """
        summary = []
        summary.append("# Temporal Agglomeration Analysis - Summary\n")
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        summary.append("## Analysis Overview\n")
        for name, df in results.items():
            summary.append(f"### {name.replace('_', ' ').title()}")
            summary.append(f"- Records: {len(df):,}")
            if 'year' in df.columns:
                summary.append(f"- Time span: {df['year'].min()}-{df['year'].max()}")
            summary.append(f"- Key variables: {', '.join(df.columns[:5])}...")
            summary.append("")
        
        summary.append("## Key Temporal Findings\n")
        
        # Concentration trends
        if 'temporal_concentration' in results:
            conc_data = results['temporal_concentration']
            summary.append("### Concentration Trends")
            summary.append(f"- Average Gini coefficient: {conc_data['gini_coefficient'].mean():.3f}")
            summary.append(f"- Concentration trend: {'Increasing' if conc_data.groupby('year')['gini_coefficient'].mean().diff().mean() > 0 else 'Decreasing'}")
            summary.append("")
        
        # Demographic effects
        if 'demographic_effects' in results:
            demo_data = results['demographic_effects']
            summary.append("### Demographic Transition Impacts")
            summary.append(f"- Average dependency ratio increase: {demo_data.groupby('demographic_phase')['dependency_ratio'].mean().diff().mean():.3f}")
            summary.append(f"- Youth ratio decline: {demo_data['youth_ratio'].corr(demo_data['start_year']):.3f}")
            summary.append("")
        
        # Economic shocks
        if 'shock_responses' in results:
            shock_data = results['shock_responses']
            summary.append("### Economic Shock Responses")
            summary.append(f"- Most resilient industries: {shock_data.groupby('industry_code')['impact_multiplier'].mean().nlargest(3).index.tolist()}")
            summary.append(f"- Highest migration volatility during: {shock_data.loc[shock_data['migration_volatility'].idxmax(), 'event']}")
            summary.append("")
        
        summary.append("## Policy Implications\n")
        summary.append("### Adaptive Agglomeration Strategy")
        summary.append("- Monitor changing concentration patterns over time")
        summary.append("- Adjust policies based on demographic transition phases")
        summary.append("- Build resilience against economic shocks")
        summary.append("- Support age-friendly industry development")
        summary.append("")
        
        # Save summary
        summary_path = self.output_dir / "temporal_analysis_summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary))
        
        self.logger.info(f"Temporal analysis summary saved to {summary_path}")

if __name__ == "__main__":
    analyzer = TemporalAgglomerationAnalyzer()
    results = analyzer.run_full_temporal_analysis()
    print(f"Temporal analysis completed! Generated {len(results)} result datasets.")
