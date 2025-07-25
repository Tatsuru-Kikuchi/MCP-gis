# Tokyo Productivity Agglomeration Analysis

A comprehensive analysis framework for studying productivity agglomeration effects across industries in Tokyo, with special focus on AI adoption impacts.

## Overview

This project analyzes how geographic concentration of economic activities affects productivity in Tokyo's various industries, and examines the role of artificial intelligence adoption in enhancing these agglomeration benefits.

## Key Research Questions

1. **Agglomeration Effects**: Which industries in Tokyo benefit most from geographic concentration?
2. **Spatial Patterns**: How does proximity to central business districts affect productivity?
3. **AI Impact**: How does AI adoption vary across industries and affect productivity?
4. **Policy Implications**: What are the optimal strategies for regional development and AI promotion?

## Project Structure

```
MCP-gis/
├── data_collection/
│   └── tokyo_economic_data_collector.py    # Data gathering and processing
├── analysis/
│   └── agglomeration_calculator.py         # Core analysis algorithms
├── visualization/
│   └── agglomeration_visualizer.py         # Chart and map generation
├── main_analysis.py                        # Main execution script
├── requirements.txt                        # Python dependencies
└── README_ANALYSIS.md                      # This file
```

## Methodology

### 1. Data Collection
- **Tokyo Statistical Yearbook**: Employment, establishment, and productivity data
- **METI Statistics**: Industry-specific productivity measures
- **Economic Census**: Spatial distribution of economic activities
- **AI Adoption Surveys**: Technology implementation rates by industry

### 2. Agglomeration Metrics
- **Concentration Indices**: Gini coefficient, Herfindahl-Hirschman Index
- **Location Quotients**: Relative industry concentration by ward
- **Spatial Autocorrelation**: Moran's I statistic for productivity clustering
- **Market Potential**: Accessibility-weighted economic mass

### 3. AI Impact Analysis
- **Adoption Rates**: Industry-specific AI implementation levels
- **Productivity Effects**: Measured gains from AI adoption
- **Investment Patterns**: AI spending and returns by sector

### 4. Statistical Models
- **Regression Analysis**: Productivity determinants including agglomeration variables
- **Spatial Econometrics**: Account for geographic spillover effects
- **Panel Data Models**: Time-series analysis of productivity changes

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Tatsuru-Kikuchi/MCP-gis.git
cd MCP-gis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Analysis

**Complete Analysis (Recommended):**
```bash
python main_analysis.py
```

**Step-by-step Execution:**
```bash
# Data collection only
python main_analysis.py --skip-analysis --skip-visualizations

# Analysis only (requires existing data)
python main_analysis.py --skip-data-collection --skip-visualizations

# Visualizations only (requires existing data and results)
python main_analysis.py --skip-data-collection --skip-analysis
```

**Custom Directories:**
```bash
python main_analysis.py --data-dir custom_data --results-dir custom_results
```

### Individual Components

**Data Collection:**
```python
from data_collection.tokyo_economic_data_collector import TokyoEconomicDataCollector

collector = TokyoEconomicDataCollector()
data = collector.run_full_collection()
```

**Analysis:**
```python
from analysis.agglomeration_calculator import AgglomerationCalculator

calculator = AgglomerationCalculator()
results = calculator.run_full_analysis()
```

**Visualization:**
```python
from visualization.agglomeration_visualizer import AgglomerationVisualizer

visualizer = AgglomerationVisualizer()
visualizer.run_all_visualizations()
```

## Output Files

### Data Files (`data/`)
- `tokyo_establishments.csv`: Establishment counts and employment by ward/industry
- `tokyo_labor_productivity.csv`: Productivity time series by industry
- `tokyo_spatial_distribution.csv`: Geographic coordinates and characteristics
- `ai_adoption_by_industry.csv`: AI adoption rates and impact estimates

### Analysis Results (`results/`)
- `concentration_indices.csv`: Industry concentration measures
- `agglomeration_effects.csv`: Regression coefficients for agglomeration variables
- `ai_productivity_impact.csv`: AI adoption effects on productivity
- `agglomeration_comprehensive_summary.csv`: Combined analysis results

### Visualizations (`visualizations/`)
- `concentration_heatmap.png`: Industry concentration patterns
- `agglomeration_effects.png`: Coefficient plots for agglomeration effects
- `ai_impact_analysis.png`: AI adoption and productivity relationships
- `tokyo_productivity_map.html`: Interactive map of Tokyo productivity
- `comprehensive_dashboard.html`: Combined analytical dashboard

## Key Findings (Expected)

### Agglomeration Effects
- **Central Ward Advantage**: Industries in Chiyoda, Chuo, and Minato show highest productivity
- **Employment Density**: Positive correlation between worker concentration and productivity
- **Knowledge Spillovers**: Evidence of positive spatial autocorrelation in innovation-intensive sectors

### AI Adoption Patterns
- **Technology Leaders**: Information & Communications (35% adoption), Finance (28%), Professional Services (25%)
- **Productivity Gains**: 5-25% improvement from AI implementation
- **Investment Returns**: Positive correlation between AI spending and productivity growth

### Industry Variations
- **High Agglomeration Benefit**: Finance, IT, Professional Services
- **Moderate Benefit**: Manufacturing, Wholesale Trade
- **Low Benefit**: Construction, Retail, Personal Services

## Policy Implications

### Urban Planning
- Support mixed-use development in central areas
- Improve transportation links to enhance market access
- Create innovation districts for knowledge-intensive industries

### AI Strategy
- Sector-specific AI adoption support programs
- SME technology upgrade incentives
- Digital infrastructure investment

### Regional Development
- Balance central concentration with satellite development
- Support outer ward specialization in appropriate industries
- Remote work infrastructure for distributed productivity

## Data Sources and Limitations

### Primary Sources
- [Tokyo Statistical Yearbook](https://www.toukei.metro.tokyo.lg.jp/tnenkan/tn-eindex.htm)
- [METI Industry Statistics](https://www.meti.go.jp/english/statistics/)
- [e-Stat Government Statistics](https://www.e-stat.go.jp/en)

### Limitations
- Sample data used for demonstration (replace with actual data collection)
- Limited to Tokyo 23 special wards (could expand to Greater Tokyo Area)
- AI adoption data based on surveys (may not capture all implementations)
- Cross-sectional analysis (longitudinal data would enhance insights)

## Future Enhancements

1. **Real Data Integration**: Connect to actual government data APIs
2. **Firm-level Analysis**: Micro-data for more precise estimates
3. **Dynamic Modeling**: Panel data analysis over time
4. **International Comparison**: Benchmark against other global cities
5. **Policy Simulation**: Model impacts of different interventions

## Contributing

Contributions are welcome! Please focus on:
- Data source connections and API integration
- Additional analytical methods
- Enhanced visualization techniques
- Documentation and code quality improvements

## License

This project is for research and educational purposes. Please cite appropriately if used in academic work.

## Contact

For questions about methodology, data sources, or collaboration opportunities, please open an issue in the repository.
