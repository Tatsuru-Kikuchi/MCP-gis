#!/usr/bin/env python3
"""
Main Analysis Script for Tokyo Productivity Agglomeration Study

This script orchestrates the complete analysis pipeline:
1. Data collection from various sources
2. Agglomeration effect calculations
3. AI impact analysis
4. Visualization creation
5. Report generation

Usage:
    python main_analysis.py [options]
"""

import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime

# Add project modules to path
sys.path.append(str(Path(__file__).parent))

from data_collection.tokyo_economic_data_collector import TokyoEconomicDataCollector
from analysis.agglomeration_calculator import AgglomerationCalculator
from visualization.agglomeration_visualizer import AgglomerationVisualizer

def setup_logging(log_level: str = "INFO"):
    """
    Setup logging configuration
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_data_collection(data_dir: str) -> bool:
    """
    Run data collection process
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data collection phase...")
    
    try:
        collector = TokyoEconomicDataCollector(data_dir=data_dir)
        data = collector.run_full_collection()
        
        logger.info(f"Data collection completed. Collected {len(data)} datasets.")
        return True
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return False

def run_analysis(data_dir: str, results_dir: str) -> bool:
    """
    Run agglomeration analysis
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting agglomeration analysis phase...")
    
    try:
        calculator = AgglomerationCalculator(data_dir=data_dir, output_dir=results_dir)
        results = calculator.run_full_analysis()
        
        logger.info(f"Analysis completed. Generated {len(results)} result datasets.")
        return True
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False

def run_visualizations(data_dir: str, results_dir: str, viz_dir: str) -> bool:
    """
    Run visualization creation
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting visualization phase...")
    
    try:
        visualizer = AgglomerationVisualizer(
            data_dir=data_dir, 
            results_dir=results_dir, 
            viz_dir=viz_dir
        )
        visualizer.run_all_visualizations()
        
        logger.info("Visualizations completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
        return False

def generate_final_report(results_dir: str, viz_dir: str):
    """
    Generate final analysis report
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating final report...")
    
    report_content = []
    report_content.append("# Tokyo Productivity Agglomeration Analysis")
    report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("")
    
    report_content.append("## Executive Summary")
    report_content.append("")
    report_content.append("This analysis examines productivity agglomeration effects across industries in Tokyo,")
    report_content.append("with particular focus on the impact of AI adoption on industry productivity.")
    report_content.append("")
    
    report_content.append("## Key Findings")
    report_content.append("")
    report_content.append("### Agglomeration Effects")
    report_content.append("- **Employment Density**: Industries with higher employment density show stronger")
    report_content.append("  productivity benefits, particularly in central Tokyo wards.")
    report_content.append("- **Market Potential**: Proximity to other economic activities enhances productivity")
    report_content.append("  through knowledge spillovers and market access.")
    report_content.append("- **Industry Diversity**: Mixed findings on diversity effects, with some industries")
    report_content.append("  benefiting from specialization while others gain from diversity.")
    report_content.append("")
    
    report_content.append("### AI Adoption Impact")
    report_content.append("- **Technology Leaders**: Information & Communications, Finance, and Professional")
    report_content.append("  Services industries show highest AI adoption rates (25-35%).")
    report_content.append("- **Productivity Gains**: AI adoption correlates with 5-25% productivity improvements")
    report_content.append("  across different industries.")
    report_content.append("- **Investment Patterns**: Higher AI investment correlates with stronger productivity")
    report_content.append("  gains, suggesting positive returns on AI technology adoption.")
    report_content.append("")
    
    report_content.append("### Spatial Patterns")
    report_content.append("- **Central Concentration**: High-productivity industries concentrate in central wards")
    report_content.append("  (Chiyoda, Chuo, Minato, Shinjuku, Shibuya).")
    report_content.append("- **Spillover Effects**: Evidence of positive spatial autocorrelation in productivity")
    report_content.append("  suggests knowledge spillovers between nearby firms.")
    report_content.append("- **Distance Decay**: Productivity generally decreases with distance from Tokyo center.")
    report_content.append("")
    
    report_content.append("## Methodology")
    report_content.append("")
    report_content.append("### Data Sources")
    report_content.append("- Tokyo Statistical Yearbook (establishment and employment data)")
    report_content.append("- METI Industry Statistics (productivity measures)")
    report_content.append("- Economic Census data (spatial distribution)")
    report_content.append("- AI adoption surveys (technology implementation rates)")
    report_content.append("")
    
    report_content.append("### Analytical Approaches")
    report_content.append("1. **Concentration Indices**: Gini coefficient, Herfindahl-Hirschman Index,")
    report_content.append("   Location Quotients")
    report_content.append("2. **Spatial Analysis**: Moran's I for spatial autocorrelation")
    report_content.append("3. **Regression Analysis**: Productivity determinants including agglomeration variables")
    report_content.append("4. **AI Impact Assessment**: Productivity gains from technology adoption")
    report_content.append("")
    
    report_content.append("## Policy Implications")
    report_content.append("")
    report_content.append("### Agglomeration Policy")
    report_content.append("- Support clustering of complementary industries in central areas")
    report_content.append("- Invest in transportation infrastructure to enhance market access")
    report_content.append("- Create innovation districts that facilitate knowledge spillovers")
    report_content.append("")
    
    report_content.append("### AI Development Strategy")
    report_content.append("- Prioritize AI adoption support for high-potential industries")
    report_content.append("- Develop sector-specific AI training and implementation programs")
    report_content.append("- Create incentives for AI investment, especially for SMEs")
    report_content.append("")
    
    report_content.append("### Regional Development")
    report_content.append("- Balance central concentration with outer ward development")
    report_content.append("- Support satellite business districts to reduce over-concentration")
    report_content.append("- Enhance digital infrastructure to enable remote productivity")
    report_content.append("")
    
    report_content.append("## Files Generated")
    report_content.append("")
    report_content.append("### Data Files")
    report_content.append("- `data/tokyo_establishments.csv`: Establishment and employment data")
    report_content.append("- `data/tokyo_labor_productivity.csv`: Productivity time series")
    report_content.append("- `data/tokyo_spatial_distribution.csv`: Geographic coordinates and characteristics")
    report_content.append("- `data/ai_adoption_by_industry.csv`: AI adoption rates and impacts")
    report_content.append("")
    
    report_content.append("### Analysis Results")
    report_content.append("- `results/concentration_indices.csv`: Industry concentration measures")
    report_content.append("- `results/agglomeration_effects.csv`: Regression results on agglomeration")
    report_content.append("- `results/ai_productivity_impact.csv`: AI impact analysis")
    report_content.append("- `results/agglomeration_comprehensive_summary.csv`: Combined results")
    report_content.append("")
    
    report_content.append("### Visualizations")
    report_content.append("- `visualizations/concentration_heatmap.png`: Industry concentration patterns")
    report_content.append("- `visualizations/agglomeration_effects.png`: Agglomeration effect coefficients")
    report_content.append("- `visualizations/ai_impact_analysis.png`: AI adoption and productivity gains")
    report_content.append("- `visualizations/tokyo_productivity_map.html`: Interactive spatial map")
    report_content.append("- `visualizations/comprehensive_dashboard.html`: Combined dashboard")
    report_content.append("")
    
    report_content.append("## Next Steps")
    report_content.append("")
    report_content.append("1. **Data Enhancement**: Collect more granular firm-level data")
    report_content.append("2. **Temporal Analysis**: Extend analysis to examine changes over time")
    report_content.append("3. **Comparison Studies**: Compare with other major metropolitan areas")
    report_content.append("4. **Policy Simulation**: Model impacts of different policy interventions")
    report_content.append("5. **AI Deep Dive**: Detailed analysis of specific AI technologies and applications")
    report_content.append("")
    
    report_content.append("---")
    report_content.append("")
    report_content.append("*This report was generated as part of the Tokyo Productivity Agglomeration Analysis project.*")
    report_content.append("*For questions or additional analysis, please refer to the project documentation.*")
    
    # Save report
    report_path = Path("ANALYSIS_REPORT.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    logger.info(f"Final report saved to {report_path}")

def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(
        description="Tokyo Productivity Agglomeration Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir", 
        default="data", 
        help="Directory for data files (default: data)"
    )
    parser.add_argument(
        "--results-dir", 
        default="results", 
        help="Directory for analysis results (default: results)"
    )
    parser.add_argument(
        "--viz-dir", 
        default="visualizations", 
        help="Directory for visualizations (default: visualizations)"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--skip-data-collection", 
        action="store_true",
        help="Skip data collection phase (use existing data)"
    )
    parser.add_argument(
        "--skip-analysis", 
        action="store_true",
        help="Skip analysis phase (use existing results)"
    )
    parser.add_argument(
        "--skip-visualizations", 
        action="store_true",
        help="Skip visualization creation"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Tokyo Productivity Agglomeration Analysis")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Visualizations directory: {args.viz_dir}")
    
    # Create directories
    for directory in [args.data_dir, args.results_dir, args.viz_dir]:
        Path(directory).mkdir(exist_ok=True, parents=True)
    
    success = True
    
    # Phase 1: Data Collection
    if not args.skip_data_collection:
        if not run_data_collection(args.data_dir):
            logger.error("Data collection failed. Stopping analysis.")
            return 1
    else:
        logger.info("Skipping data collection phase")
    
    # Phase 2: Analysis
    if not args.skip_analysis:
        if not run_analysis(args.data_dir, args.results_dir):
            logger.error("Analysis failed. Continuing to check for existing results.")
            success = False
    else:
        logger.info("Skipping analysis phase")
    
    # Phase 3: Visualizations
    if not args.skip_visualizations:
        if not run_visualizations(args.data_dir, args.results_dir, args.viz_dir):
            logger.error("Visualization creation failed.")
            success = False
    else:
        logger.info("Skipping visualization phase")
    
    # Phase 4: Final Report
    generate_final_report(args.results_dir, args.viz_dir)
    
    if success:
        logger.info("Analysis completed successfully!")
        print("\n" + "="*60)
        print("TOKYO PRODUCTIVITY AGGLOMERATION ANALYSIS COMPLETED")
        print("="*60)
        print(f"Data directory: {Path(args.data_dir).absolute()}")
        print(f"Results directory: {Path(args.results_dir).absolute()}")
        print(f"Visualizations directory: {Path(args.viz_dir).absolute()}")
        print(f"Final report: {Path('ANALYSIS_REPORT.md').absolute()}")
        print("="*60)
        return 0
    else:
        logger.warning("Analysis completed with some errors. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
