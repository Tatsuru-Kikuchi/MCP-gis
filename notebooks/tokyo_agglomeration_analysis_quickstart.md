# Tokyo Productivity Agglomeration Analysis - Quick Start Guide

This notebook demonstrates how to use the Tokyo productivity agglomeration analysis framework.

## Setup

```python
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path.cwd()))

# Import our modules
from data_collection.tokyo_economic_data_collector import TokyoEconomicDataCollector
from analysis.agglomeration_calculator import AgglomerationCalculator
from visualization.agglomeration_visualizer import AgglomerationVisualizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Step 1: Data Collection

```python
# Initialize data collector
collector = TokyoEconomicDataCollector(data_dir="sample_data")

# Collect all data (this creates sample data for demonstration)
print("Collecting data...")
data = collector.run_full_collection()

print(f"✅ Collected {len(data)} datasets:")
for name, df in data.items():
    print(f"  - {name}: {len(df)} rows, {len(df.columns)} columns")
```

## Step 2: Basic Data Exploration

```python
# Load the generated data
establishments = pd.read_csv("sample_data/tokyo_establishments.csv")
productivity = pd.read_csv("sample_data/tokyo_labor_productivity.csv")
spatial = pd.read_csv("sample_data/tokyo_spatial_distribution.csv")
ai_adoption = pd.read_csv("sample_data/ai_adoption_by_industry.csv")

print("📊 Data Overview:")
print(f"Establishments data: {len(establishments)} records")
print(f"Productivity data: {len(productivity)} records")
print(f"Spatial data: {len(spatial)} wards")
print(f"AI adoption data: {len(ai_adoption)} industries")
```

### Quick Data Visualization

```python
# Top industries by employment
top_employers = establishments.groupby('industry_name')['employees'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
top_employers.plot(kind='bar')
plt.title('Top 10 Industries by Employment in Tokyo')
plt.xlabel('Industry')
plt.ylabel('Total Employees')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

```python
# Productivity by ward type
latest_prod = productivity[productivity['year'] == 2023]
ward_prod = latest_prod.merge(spatial[['ward', 'ward_type']], on='ward')

ward_type_productivity = ward_prod.groupby('ward_type')['productivity_thousand_yen_per_employee'].mean()

plt.figure(figsize=(8, 6))
ward_type_productivity.plot(kind='bar', color=['lightcoral', 'lightblue'])
plt.title('Average Productivity by Ward Type')
plt.xlabel('Ward Type')
plt.ylabel('Productivity (Thousand Yen per Employee)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

## Step 3: Agglomeration Analysis

```python
# Initialize calculator
calculator = AgglomerationCalculator(data_dir="sample_data", output_dir="sample_results")

# Run analysis
print("Running agglomeration analysis...")
results = calculator.run_full_analysis()

print(f"✅ Analysis completed. Generated {len(results)} result files:")
for name in results.keys():
    print(f"  - {name}")
```

### View Key Results

```python
# Load concentration indices
concentration = pd.read_csv("sample_results/concentration_indices.csv")
print("🎯 Industry Concentration Indices:")
print(concentration[['industry_name', 'gini_coefficient', 'herfindahl_index', 'avg_location_quotient']].head(10))
```

```python
# Load agglomeration effects
agglomeration = pd.read_csv("sample_results/agglomeration_effects.csv")
print("📈 Agglomeration Effects on Productivity:")
print(agglomeration[['industry_name', 'employment_density_coeff', 'market_potential_coeff', 'r_squared']].head(10))
```

```python
# Load AI impact analysis
ai_impact = pd.read_csv("sample_results/ai_productivity_impact.csv")
print("🤖 AI Impact on Productivity:")
print(ai_impact[['industry_name', 'ai_adoption_rate', 'productivity_gain_percent']].head(10))
```

## Step 4: Visualization

```python
# Initialize visualizer
visualizer = AgglomerationVisualizer(
    data_dir="sample_data", 
    results_dir="sample_results", 
    viz_dir="sample_visualizations"
)

# Create all visualizations
print("Creating visualizations...")
visualizer.run_all_visualizations()
print("✅ Visualizations completed!")
```

## Step 5: Key Findings Summary

```python
# Load comprehensive summary
summary = pd.read_csv("sample_results/agglomeration_comprehensive_summary.csv")

# Top industries by agglomeration score
top_agglomeration = summary.nlargest(10, 'agglomeration_score')[
    ['industry_name', 'agglomeration_score', 'ai_adoption_rate', 'mean_productivity']
]

print("🏆 Top 10 Industries by Agglomeration Score:")
print(top_agglomeration)
```

```python
# AI vs Agglomeration correlation
import scipy.stats as stats

correlation, p_value = stats.pearsonr(
    summary['ai_adoption_rate'].fillna(0), 
    summary['agglomeration_score'].fillna(0)
)

print(f"\n🔗 AI Adoption vs Agglomeration Score:")
print(f"Correlation: {correlation:.3f}")
print(f"P-value: {p_value:.3f}")
print(f"Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} correlation")
```

## Step 6: Industry-Specific Analysis

```python
def analyze_industry(industry_code, industry_name):
    """Analyze a specific industry in detail"""
    
    print(f"🔍 Detailed Analysis: {industry_name}")
    print("="*50)
    
    # Concentration metrics
    conc_data = concentration[concentration['industry_code'] == industry_code].iloc[0]
    print(f"📊 Concentration Metrics:")
    print(f"  Gini Coefficient: {conc_data['gini_coefficient']:.3f}")
    print(f"  Herfindahl Index: {conc_data['herfindahl_index']:.3f}")
    print(f"  Location Quotient: {conc_data['avg_location_quotient']:.3f}")
    
    # Agglomeration effects
    agg_data = agglomeration[agglomeration['industry_code'] == industry_code]
    if not agg_data.empty:
        agg_data = agg_data.iloc[0]
        print(f"\n📈 Agglomeration Effects:")
        print(f"  Employment Density Coeff: {agg_data['employment_density_coeff']:.3f}")
        print(f"  Market Potential Coeff: {agg_data['market_potential_coeff']:.3f}")
        print(f"  Diversity Coeff: {agg_data['diversity_coeff']:.3f}")
        print(f"  Model R²: {agg_data['r_squared']:.3f}")
    
    # AI impact
    ai_data = ai_impact[ai_impact['industry_code'] == industry_code].iloc[0]
    print(f"\n🤖 AI Impact:")
    print(f"  Adoption Rate: {ai_data['ai_adoption_rate']:.1%}")
    print(f"  Productivity Gain: {ai_data['productivity_gain_percent']:.1f}%")
    print(f"  Investment per Company: ¥{ai_data['ai_investment_million_yen_per_company']:.1f}M")
    
    print("\n")

# Analyze key industries
analyze_industry('F', 'Information and Communications')
analyze_industry('I', 'Finance and Insurance')
analyze_industry('D', 'Manufacturing')
```

## Step 7: Policy Recommendations

```python
# Generate policy insights based on analysis
def generate_policy_recommendations():
    """Generate policy recommendations based on analysis results"""
    
    print("🏛️ POLICY RECOMMENDATIONS")
    print("="*60)
    
    # High agglomeration industries
    high_agg = summary.nlargest(5, 'agglomeration_score')
    print("1. PRIORITY INDUSTRIES FOR AGGLOMERATION SUPPORT:")
    for _, row in high_agg.iterrows():
        print(f"   • {row['industry_name']}")
    
    # High AI potential industries  
    high_ai = summary.nlargest(5, 'productivity_gain_percent')
    print("\n2. PRIORITY INDUSTRIES FOR AI ADOPTION SUPPORT:")
    for _, row in high_ai.iterrows():
        if pd.notna(row['productivity_gain_percent']):
            print(f"   • {row['industry_name']} ({row['productivity_gain_percent']:.1f}% gain potential)")
    
    # Central vs outer ward strategies
    central_industries = summary[summary['employment_density_coeff'] > 0].nlargest(3, 'employment_density_coeff')
    print("\n3. CENTRAL WARD DEVELOPMENT FOCUS:")
    for _, row in central_industries.iterrows():
        if pd.notna(row['employment_density_coeff']):
            print(f"   • {row['industry_name']} (density coeff: {row['employment_density_coeff']:.3f})")
    
    print("\n4. RECOMMENDED ACTIONS:")
    print("   📍 Spatial Strategy:")
    print("     - Enhance transport links to improve market access")
    print("     - Create mixed-use developments in central areas")
    print("     - Support satellite business districts")
    
    print("   🤖 AI Strategy:")
    print("     - Sector-specific AI training programs")
    print("     - SME technology adoption subsidies") 
    print("     - Digital infrastructure investment")
    
    print("   🏢 Industry Strategy:")
    print("     - Support clustering of complementary industries")
    print("     - Create innovation hubs for knowledge spillovers")
    print("     - Develop talent pipelines for high-productivity sectors")

generate_policy_recommendations()
```

## Step 8: Export Results

```python
# Create summary report
def create_summary_report():
    """Create a concise summary report"""
    
    report = []
    report.append("# Tokyo Productivity Agglomeration Analysis - Summary Report")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Key statistics
    total_establishments = establishments['establishments'].sum()
    total_employees = establishments['employees'].sum()
    avg_productivity = productivity[productivity['year'] == 2023]['productivity_thousand_yen_per_employee'].mean()
    
    report.append("## Key Statistics")
    report.append(f"- Total Establishments: {total_establishments:,}")
    report.append(f"- Total Employees: {total_employees:,}")
    report.append(f"- Average Productivity (2023): ¥{avg_productivity:.0f} thousand per employee")
    report.append("")
    
    # Top performers
    report.append("## Top Performing Industries")
    top_5 = summary.nlargest(5, 'agglomeration_score')
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        report.append(f"{i}. {row['industry_name']} (Score: {row['agglomeration_score']:.3f})")
    report.append("")
    
    # AI leaders
    report.append("## AI Adoption Leaders")
    ai_leaders = summary.nlargest(5, 'ai_adoption_rate')
    for i, (_, row) in enumerate(ai_leaders.iterrows(), 1):
        if pd.notna(row['ai_adoption_rate']):
            report.append(f"{i}. {row['industry_name']} ({row['ai_adoption_rate']:.1%} adoption)")
    
    # Save report
    with open("QUICK_ANALYSIS_SUMMARY.md", "w") as f:
        f.write("\n".join(report))
    
    print("📄 Summary report saved to QUICK_ANALYSIS_SUMMARY.md")

create_summary_report()
```

## Conclusion

This notebook demonstrates the key capabilities of the Tokyo productivity agglomeration analysis framework:

1. **Data Collection**: Automated gathering and processing of economic data
2. **Agglomeration Analysis**: Comprehensive measurement of concentration effects
3. **AI Impact Assessment**: Evaluation of technology adoption on productivity  
4. **Spatial Analysis**: Geographic patterns and spillover effects
5. **Visualization**: Multiple chart types and interactive maps
6. **Policy Insights**: Evidence-based recommendations

The modular design allows you to:
- Run individual components independently
- Customize analysis parameters
- Extend with additional data sources
- Adapt for other metropolitan areas

For more detailed analysis, refer to the comprehensive output files and visualizations generated by the framework.

## Next Steps

- Replace sample data with real government data sources
- Extend analysis to firm-level microdata
- Add temporal analysis for trend identification
- Compare with other major metropolitan areas
- Develop policy simulation capabilities
