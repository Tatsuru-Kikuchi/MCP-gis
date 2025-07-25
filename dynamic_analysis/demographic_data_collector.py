#!/usr/bin/env python3
"""
Demographic Data Collector for Dynamic Agglomeration Analysis
Collects demographic and temporal data for Japan's aging society analysis

Focuses on:
- Population aging trends
- Labor force participation changes
- Migration patterns over time
- Industry workforce demographics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
import json

class DemographicDataCollector:
    """
    Collects demographic and temporal data for dynamic agglomeration analysis
    """
    
    def __init__(self, data_dir: str = "data/demographic"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Time periods for analysis
        self.historical_years = list(range(2000, 2024))
        self.projection_years = list(range(2024, 2051))  # 25-year projection
        
        # Age groups for analysis
        self.age_groups = {
            'young': (15, 34),      # Young workforce
            'prime': (35, 54),      # Prime working age
            'mature': (55, 64),     # Pre-retirement
            'elderly': (65, 100)    # Post-retirement
        }
        
        # Tokyo wards
        self.tokyo_wards = [
            'Chiyoda', 'Chuo', 'Minato', 'Shinjuku', 'Bunkyo', 'Taito',
            'Sumida', 'Koto', 'Shinagawa', 'Meguro', 'Ota', 'Setagaya',
            'Shibuya', 'Nakano', 'Suginami', 'Toshima', 'Kita', 'Arakawa',
            'Itabashi', 'Nerima', 'Adachi', 'Katsushika', 'Edogawa'
        ]
        
        # Industry categories (JSIC)
        self.industries = {
            'D': 'Manufacturing',
            'F': 'Information and communications',
            'G': 'Transport and postal activities',
            'H': 'Wholesale and retail trade',
            'I': 'Finance and insurance',
            'J': 'Real estate and goods rental',
            'K': 'Scientific research and professional services',
            'L': 'Accommodations and eating services',
            'M': 'Living-related and personal services',
            'N': 'Education and learning support',
            'O': 'Medical, health care and welfare'
        }
    
    def generate_historical_population_data(self) -> pd.DataFrame:
        """
        Generate historical population data by age group and ward
        """
        self.logger.info("Generating historical population data...")
        
        np.random.seed(42)
        data = []
        
        for year in self.historical_years:
            for ward in self.tokyo_wards:
                # Base population varies by ward type
                if ward in ['Chiyoda', 'Chuo', 'Minato']:
                    base_pop = np.random.randint(150000, 250000)
                elif ward in ['Shinjuku', 'Shibuya', 'Shinagawa']:
                    base_pop = np.random.randint(300000, 450000)
                else:
                    base_pop = np.random.randint(400000, 700000)
                
                # Age distribution changes over time (aging trend)
                aging_factor = (year - 2000) * 0.02  # 2% per year aging
                
                for age_group, (min_age, max_age) in self.age_groups.items():
                    if age_group == 'young':
                        # Declining young population
                        base_share = 0.25 - aging_factor * 0.5
                        variation = np.random.uniform(0.8, 1.2)
                    elif age_group == 'prime':
                        # Stable prime age
                        base_share = 0.35 - aging_factor * 0.2
                        variation = np.random.uniform(0.9, 1.1)
                    elif age_group == 'mature':
                        # Slightly increasing
                        base_share = 0.20 + aging_factor * 0.1
                        variation = np.random.uniform(0.95, 1.05)
                    else:  # elderly
                        # Rapidly increasing elderly
                        base_share = 0.20 + aging_factor * 0.6
                        variation = np.random.uniform(1.0, 1.3)
                    
                    population = int(base_pop * max(base_share, 0.05) * variation)
                    
                    data.append({
                        'year': year,
                        'ward': ward,
                        'age_group': age_group,
                        'population': population,
                        'min_age': min_age,
                        'max_age': max_age
                    })
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_path = self.data_dir / "historical_population_by_age.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Historical population data saved to {output_path}")
        
        return df
    
    def generate_labor_force_participation_data(self) -> pd.DataFrame:
        """
        Generate labor force participation rates by age group and industry over time
        """
        self.logger.info("Generating labor force participation data...")
        
        np.random.seed(43)
        data = []
        
        for year in self.historical_years:
            for age_group in self.age_groups.keys():
                for industry_code, industry_name in self.industries.items():
                    # Base participation rates vary by age and industry
                    if age_group == 'young':
                        if industry_code in ['F', 'K']:  # Tech, Professional
                            base_rate = 0.45
                        elif industry_code in ['L', 'M']:  # Service
                            base_rate = 0.35
                        else:
                            base_rate = 0.25
                    elif age_group == 'prime':
                        if industry_code in ['I', 'J']:  # Finance, Real Estate
                            base_rate = 0.65
                        elif industry_code == 'D':  # Manufacturing
                            base_rate = 0.55
                        else:
                            base_rate = 0.45
                    elif age_group == 'mature':
                        if industry_code in ['N', 'O']:  # Education, Healthcare
                            base_rate = 0.40
                        else:
                            base_rate = 0.30
                    else:  # elderly
                        if industry_code in ['L', 'M']:  # Service sectors
                            base_rate = 0.15
                        else:
                            base_rate = 0.08
                    
                    # Trends over time
                    time_factor = (year - 2000) / 24  # 24-year span
                    
                    # Technology sectors growing for young people
                    if industry_code == 'F' and age_group == 'young':
                        trend = 0.02 * time_factor
                    # Healthcare growing for all ages (aging society)
                    elif industry_code == 'O':
                        trend = 0.015 * time_factor
                    # Manufacturing declining for young
                    elif industry_code == 'D' and age_group == 'young':
                        trend = -0.01 * time_factor
                    # Elderly working longer
                    elif age_group == 'elderly':
                        trend = 0.005 * time_factor
                    else:
                        trend = np.random.uniform(-0.005, 0.005) * time_factor
                    
                    participation_rate = max(0.01, min(0.8, base_rate + trend))
                    participation_rate *= np.random.uniform(0.9, 1.1)
                    
                    data.append({
                        'year': year,
                        'age_group': age_group,
                        'industry_code': industry_code,
                        'industry_name': industry_name,
                        'participation_rate': participation_rate
                    })
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_path = self.data_dir / "labor_force_participation.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Labor force participation data saved to {output_path}")
        
        return df
    
    def generate_migration_patterns_data(self) -> pd.DataFrame:
        """
        Generate internal migration patterns over time
        """
        self.logger.info("Generating migration patterns data...")
        
        np.random.seed(44)
        data = []
        
        for year in self.historical_years:
            for ward in self.tokyo_wards:
                for age_group in self.age_groups.keys():
                    # Migration patterns vary by age and ward type
                    if ward in ['Chiyoda', 'Chuo', 'Minato']:  # Central wards
                        if age_group == 'young':
                            net_migration_rate = 0.08  # Strong inflow of young workers
                        elif age_group == 'prime':
                            net_migration_rate = 0.03
                        elif age_group == 'mature':
                            net_migration_rate = -0.02  # Outflow for family reasons
                        else:
                            net_migration_rate = -0.05  # Elderly move to suburbs
                    elif ward in ['Shinjuku', 'Shibuya', 'Shinagawa']:  # Sub-centers
                        if age_group == 'young':
                            net_migration_rate = 0.05
                        elif age_group == 'prime':
                            net_migration_rate = 0.02
                        else:
                            net_migration_rate = -0.01
                    else:  # Outer wards
                        if age_group == 'young':
                            net_migration_rate = -0.03  # Outflow to central areas
                        elif age_group in ['mature', 'elderly']:
                            net_migration_rate = 0.02  # Inflow for housing
                        else:
                            net_migration_rate = 0.01
                    
                    # COVID-19 impact (2020-2022)
                    if year >= 2020 and year <= 2022:
                        if ward in ['Chiyoda', 'Chuo', 'Minato']:
                            net_migration_rate *= 0.5  # Reduced inflow due to remote work
                        else:
                            net_migration_rate *= 1.2  # Increased suburban preference
                    
                    # Add random variation
                    net_migration_rate *= np.random.uniform(0.8, 1.2)
                    
                    data.append({
                        'year': year,
                        'ward': ward,
                        'age_group': age_group,
                        'net_migration_rate': net_migration_rate,
                        'ward_type': 'central' if ward in ['Chiyoda', 'Chuo', 'Minato'] else 
                                   'sub_center' if ward in ['Shinjuku', 'Shibuya', 'Shinagawa'] else 'outer'
                    })
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_path = self.data_dir / "migration_patterns.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Migration patterns data saved to {output_path}")
        
        return df
    
    def generate_productivity_aging_effects(self) -> pd.DataFrame:
        """
        Generate data on how aging affects productivity by industry
        """
        self.logger.info("Generating productivity-aging effects data...")
        
        np.random.seed(45)
        data = []
        
        for year in self.historical_years:
            for industry_code, industry_name in self.industries.items():
                for age_group in self.age_groups.keys():
                    # Base productivity by age group (normalized to prime = 1.0)
                    if age_group == 'young':
                        if industry_code == 'F':  # Tech: young advantage
                            base_productivity = 0.95
                        else:
                            base_productivity = 0.85  # Learning curve
                    elif age_group == 'prime':
                        base_productivity = 1.0  # Peak productivity
                    elif age_group == 'mature':
                        if industry_code in ['N', 'O', 'K']:  # Knowledge work
                            base_productivity = 1.05  # Experience premium
                        else:
                            base_productivity = 0.95
                    else:  # elderly
                        if industry_code in ['N', 'L']:  # Education, Service
                            base_productivity = 0.85  # Can contribute
                        else:
                            base_productivity = 0.70
                    
                    # Technology adoption effects over time
                    tech_adoption_rate = min(1.0, (year - 2000) * 0.03)
                    
                    if age_group == 'young':
                        tech_boost = tech_adoption_rate * 0.15  # 15% boost from tech
                    elif age_group == 'prime':
                        tech_boost = tech_adoption_rate * 0.10
                    elif age_group == 'mature':
                        tech_boost = tech_adoption_rate * 0.05
                    else:
                        tech_boost = tech_adoption_rate * 0.02
                    
                    # Industry-specific technology effects
                    if industry_code == 'F':  # IT sector
                        tech_boost *= 2.0
                    elif industry_code in ['I', 'K']:  # Finance, Professional
                        tech_boost *= 1.5
                    elif industry_code in ['L', 'M']:  # Traditional services
                        tech_boost *= 0.5
                    
                    adjusted_productivity = base_productivity + tech_boost
                    adjusted_productivity *= np.random.uniform(0.95, 1.05)
                    
                    data.append({
                        'year': year,
                        'industry_code': industry_code,
                        'industry_name': industry_name,
                        'age_group': age_group,
                        'base_productivity': base_productivity,
                        'tech_adoption_boost': tech_boost,
                        'adjusted_productivity': adjusted_productivity
                    })
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_path = self.data_dir / "productivity_aging_effects.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Productivity-aging effects data saved to {output_path}")
        
        return df
    
    def generate_economic_shock_events(self) -> pd.DataFrame:
        """
        Generate data on economic shocks and their impacts
        """
        self.logger.info("Generating economic shock events data...")
        
        # Define major economic events
        events = [
            {
                'event': 'Global Financial Crisis',
                'start_year': 2008,
                'end_year': 2010,
                'impact_type': 'recession',
                'severity': 0.8
            },
            {
                'event': 'Great East Japan Earthquake',
                'start_year': 2011,
                'end_year': 2012,
                'impact_type': 'supply_shock',
                'severity': 0.9
            },
            {
                'event': 'COVID-19 Pandemic',
                'start_year': 2020,
                'end_year': 2022,
                'impact_type': 'mixed',
                'severity': 1.0
            }
        ]
        
        data = []
        
        for event in events:
            for year in range(event['start_year'], event['end_year'] + 1):
                for industry_code, industry_name in self.industries.items():
                    # Different industries affected differently
                    if event['event'] == 'COVID-19 Pandemic':
                        if industry_code == 'F':  # IT benefited
                            impact_multiplier = 1.2
                        elif industry_code in ['L', 'G']:  # Hospitality, Transport hurt
                            impact_multiplier = 0.6
                        elif industry_code == 'O':  # Healthcare essential
                            impact_multiplier = 1.1
                        else:
                            impact_multiplier = 0.85
                    elif event['event'] == 'Global Financial Crisis':
                        if industry_code == 'I':  # Finance heavily hit
                            impact_multiplier = 0.7
                        elif industry_code == 'J':  # Real estate
                            impact_multiplier = 0.75
                        else:
                            impact_multiplier = 0.9
                    else:  # Earthquake
                        if industry_code == 'D':  # Manufacturing supply chains
                            impact_multiplier = 0.8
                        elif industry_code == 'G':  # Transport
                            impact_multiplier = 0.85
                        else:
                            impact_multiplier = 0.95
                    
                    data.append({
                        'year': year,
                        'event': event['event'],
                        'industry_code': industry_code,
                        'industry_name': industry_name,
                        'impact_multiplier': impact_multiplier,
                        'severity': event['severity'],
                        'impact_type': event['impact_type']
                    })
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_path = self.data_dir / "economic_shock_events.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Economic shock events data saved to {output_path}")
        
        return df
    
    def run_full_demographic_collection(self) -> Dict[str, pd.DataFrame]:
        """
        Run complete demographic data collection
        """
        self.logger.info("Starting full demographic data collection...")
        
        datasets = {
            'population': self.generate_historical_population_data(),
            'labor_participation': self.generate_labor_force_participation_data(),
            'migration_patterns': self.generate_migration_patterns_data(),
            'productivity_aging': self.generate_productivity_aging_effects(),
            'economic_shocks': self.generate_economic_shock_events()
        }
        
        # Create summary
        self._create_demographic_summary(datasets)
        
        self.logger.info("Demographic data collection completed successfully!")
        return datasets
    
    def _create_demographic_summary(self, datasets: Dict[str, pd.DataFrame]):
        """
        Create summary of demographic data collection
        """
        summary = []
        summary.append("# Dynamic Agglomeration Analysis - Demographic Data Summary\n")
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        summary.append("## Dataset Overview\n")
        for name, df in datasets.items():
            summary.append(f"### {name.replace('_', ' ').title()}")
            summary.append(f"- Records: {len(df):,}")
            summary.append(f"- Time span: {df['year'].min()}-{df['year'].max()}")
            summary.append(f"- Columns: {', '.join(df.columns)}")
            summary.append("")
        
        summary.append("## Key Trends Captured\n")
        summary.append("### Demographic Transition")
        summary.append("- Aging population with declining young workforce")
        summary.append("- Changing labor force participation by age group")
        summary.append("- Migration patterns reflecting economic opportunities")
        summary.append("")
        
        summary.append("### Economic Dynamics")
        summary.append("- Technology adoption effects on productivity")
        summary.append("- Industry-specific aging impacts")
        summary.append("- Economic shock responses and resilience")
        summary.append("")
        
        summary.append("### Spatial Patterns")
        summary.append("- Central ward advantages for young workers")
        summary.append("- Suburban preferences for older demographics")
        summary.append("- COVID-19 induced spatial redistribution")
        summary.append("")
        
        # Save summary
        summary_path = self.data_dir / "demographic_data_summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary))
        
        self.logger.info(f"Demographic data summary saved to {summary_path}")

if __name__ == "__main__":
    collector = DemographicDataCollector()
    data = collector.run_full_demographic_collection()
    print(f"Collected {len(data)} demographic datasets successfully!")
