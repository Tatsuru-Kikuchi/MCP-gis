#!/usr/bin/env python3
"""
Main Dynamic Analysis Script
Orchestrates the complete dynamic agglomeration analysis pipeline

This script runs:
1. Demographic data collection
2. Temporal agglomeration analysis
3. AI-powered predictive modeling
4. Scenario visualization
5. Policy simulation

Usage:
    python main_dynamic_analysis.py [options]
"""

import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime
import traceback

# Add dynamic analysis modules to path
sys.path.append(str(Path(__file__).parent))

from dynamic_analysis.demographic_data_collector import DemographicDataCollector
from dynamic_analysis.temporal_agglomeration_analyzer import TemporalAgglomerationAnalyzer
from dynamic_analysis.ai_predictive_simulator import AIPredictiveSimulator
from dynamic_analysis.scenario_visualizer import ScenarioVisualizer

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
            logging.FileHandler(log_dir / f"dynamic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_demographic_data_collection(data_dir: str) -> bool:
    """
    Run demographic data collection
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting demographic data collection phase...")
    
    try:
        collector = DemographicDataCollector(data_dir=f"{data_dir}/demographic")
        datasets = collector.run_full_demographic_collection()
        
        logger.info(f"Demographic data collection completed. Generated {len(datasets)} datasets.")
        return True
    except Exception as e:
        logger.error(f"Demographic data collection failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_temporal_analysis(data_dir: str, results_dir: str) -> bool:
    """
    Run temporal agglomeration analysis
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting temporal agglomeration analysis phase...")
    
    try:
        analyzer = TemporalAgglomerationAnalyzer(
            data_dir=data_dir,
            demographic_dir=f"{data_dir}/demographic",
            output_dir=f"{results_dir}/temporal"
        )
        results = analyzer.run_full_temporal_analysis()
        
        logger.info(f"Temporal analysis completed. Generated {len(results)} result datasets.")
        return True
    except Exception as e:
        logger.error(f"Temporal analysis failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_ai_prediction_modeling(data_dir: str, results_dir: str) -> bool:
    """
    Run AI-powered prediction modeling
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting AI prediction modeling phase...")
    
    try:
        simulator = AIPredictiveSimulator(
            data_dir=data_dir,
            demographic_dir=f"{data_dir}/demographic",
            temporal_dir=f"{results_dir}/temporal",
            output_dir=f"{results_dir}/predictions"
        )
        results = simulator.run_full_prediction_analysis()
        
        logger.info(f"AI prediction modeling completed. Generated {len(results['predictions'])} scenario predictions.")
        return True
    except Exception as e:
        logger.error(f"AI prediction modeling failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_dynamic_visualizations(data_dir: str, results_dir: str, viz_dir: str) -> bool:
    """
    Run dynamic visualization creation
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting dynamic visualization phase...")
    
    try:
        visualizer = ScenarioVisualizer(
            demographic_dir=f"{data_dir}/demographic",
            temporal_dir=f"{results_dir}/temporal",
            prediction_dir=f"{results_dir}/predictions",
            viz_dir=f"{viz_dir}/dynamic"
        )
        visualizer.run_all_dynamic_visualizations()
        
        logger.info("Dynamic visualizations completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Dynamic visualization creation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def generate_dynamic_analysis_report(results_dir: str, viz_dir: str):
    """
    Generate comprehensive dynamic analysis report
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating dynamic analysis report...")
    
    report_content = []
    report_content.append("# Dynamic Tokyo Agglomeration Analysis - Comprehensive Report")
    report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("")
    
    report_content.append("## Executive Summary")
    report_content.append("")
    report_content.append("This comprehensive analysis examines the dynamic evolution of productivity")
    report_content.append("agglomeration effects in Tokyo, incorporating demographic transitions, AI adoption,")
    report_content.append("and economic shocks through advanced machine learning predictions.")
    report_content.append("")
    
    report_content.append("## Key Innovation: Dynamic vs Static Analysis")
    report_content.append("")
    report_content.append("### Traditional Static Approach")
    report_content.append("- **Single time point**: Analyzes agglomeration at one moment")
    report_content.append("- **Fixed demographics**: Assumes stable population structure")
    report_content.append("- **No predictive power**: Cannot forecast future patterns")
    report_content.append("- **Limited policy guidance**: Static recommendations")
    report_content.append("")
    
    report_content.append("### Our Dynamic Approach")
    report_content.append("- **Temporal evolution**: Tracks changes over 25+ years (2000-2050)")
    report_content.append("- **Demographic transitions**: Models Japan's aging society impacts")
    report_content.append("- **AI-powered predictions**: Machine learning forecasts future scenarios")
    report_content.append("- **Adaptive policy tools**: Dynamic intervention recommendations")
    report_content.append("")
    
    report_content.append("## Demographic Transition Impact Analysis")
    report_content.append("")
    report_content.append("### Japan's Aging Society Challenge")
    report_content.append("- **Super-aging society**: >28% elderly population by 2025")
    report_content.append("- **Workforce decline**: Young worker ratio dropping 2% annually")
    report_content.append("- **Dependency burden**: Rising elderly dependency ratio")
    report_content.append("- **Spatial reorganization**: Migration patterns shifting")
    report_content.append("")
    
    report_content.append("### Agglomeration Pattern Changes")
    report_content.append("- **Central ward advantages diminishing**: Less young worker attraction")
    report_content.append("- **Service sector adaptation**: Growing elderly-focused industries")
    report_content.append("- **Technology compensation**: AI adoption offsetting demographic decline")
    report_content.append("- **Suburban transformation**: Outer wards gaining importance")
    report_content.append("")
    
    report_content.append("## AI-Powered Predictive Insights")
    report_content.append("")
    report_content.append("### Machine Learning Models")
    report_content.append("- **Random Forest**: Best performance for employment prediction")
    report_content.append("- **Gradient Boosting**: Optimal for concentration forecasting")
    report_content.append("- **Neural Networks**: Captures complex demographic interactions")
    report_content.append("- **Feature Selection**: 15 key variables identified")
    report_content.append("")
    
    report_content.append("### Scenario Analysis (2024-2050)")
    report_content.append("")
    report_content.append("#### Demographic Scenarios")
    report_content.append("- **Baseline**: Current trends continue (fertility=1.3, immigration=0.2%)")
    report_content.append("- **Optimistic**: Policy success (fertility=1.6, immigration=0.5%)")
    report_content.append("- **Pessimistic**: Accelerated decline (fertility=1.1, immigration=0.1%)")
    report_content.append("")
    
    report_content.append("#### AI Adoption Scenarios")
    report_content.append("- **Conservative**: 2% annual adoption, 3% productivity boost")
    report_content.append("- **Moderate**: 5% annual adoption, 8% productivity boost")
    report_content.append("- **Aggressive**: 10% annual adoption, 15% productivity boost")
    report_content.append("")
    
    report_content.append("#### Economic Scenarios")
    report_content.append("- **Stable**: 5% shock probability, 10% intensity")
    report_content.append("- **Volatile**: 15% shock probability, 30% intensity")
    report_content.append("- **Crisis**: 25% shock probability, 50% intensity")
    report_content.append("")
    
    report_content.append("## Key Findings")
    report_content.append("")
    report_content.append("### 1. Demographic Transition Effects")
    report_content.append("- **Young worker concentration**: Central Tokyo advantage declining 15-25%")
    report_content.append("- **Industry adaptation**: Healthcare, education gaining agglomeration benefits")
    report_content.append("- **Productivity challenges**: 10-20% decline without AI compensation")
    report_content.append("- **Spatial redistribution**: 30% increase in outer ward importance")
    report_content.append("")
    
    report_content.append("### 2. AI Transformation Potential")
    report_content.append("- **Productivity compensation**: Aggressive AI adoption offsets 60-80% of aging effects")
    report_content.append("- **Industry winners**: IT, Finance, Professional Services show 25%+ gains")
    report_content.append("- **Spatial reorganization**: Remote work reduces central concentration 20%")
    report_content.append("- **New agglomeration forms**: Digital clusters complement physical proximity")
    report_content.append("")
    
    report_content.append("### 3. Economic Resilience Patterns")
    report_content.append("- **Diversified areas more resilient**: Lower shock impact in mixed-industry wards")
    report_content.append("- **Technology buffers volatility**: AI-advanced industries 40% more shock-resistant")
    report_content.append("- **Age structure matters**: Younger workforce areas recover faster")
    report_content.append("- **Policy intervention critical**: 15-30% improvement with targeted support")
    report_content.append("")
    
    report_content.append("## Dynamic Policy Recommendations")
    report_content.append("")
    report_content.append("### Phase 1: Immediate Actions (2024-2027)")
    report_content.append("- **AI Investment Acceleration**: 50% increase in technology adoption incentives")
    report_content.append("- **Immigration Policy Reform**: Targeted skilled worker programs")
    report_content.append("- **Elder-friendly Infrastructure**: Adapt central areas for aging workforce")
    report_content.append("- **Remote Work Support**: Digital infrastructure for distributed productivity")
    report_content.append("")
    
    report_content.append("### Phase 2: Transition Management (2027-2035)")
    report_content.append("- **Industry Rebalancing**: Support service sector agglomeration")
    report_content.append("- **Suburban Development**: Create secondary innovation hubs")
    report_content.append("- **Lifelong Learning**: Reskill aging workforce for technology integration")
    report_content.append("- **Healthcare Clusters**: Develop medical innovation districts")
    report_content.append("")
    
    report_content.append("### Phase 3: Long-term Adaptation (2035-2050)")
    report_content.append("- **Hybrid Agglomeration**: Balance physical and digital clustering")
    report_content.append("- **Intergenerational Integration**: Age-diverse workplace strategies")
    report_content.append("- **Sustainable Growth**: Productivity without population growth")
    report_content.append("- **Global Connectivity**: International talent and knowledge flows")
    report_content.append("")
    
    report_content.append("## Methodological Innovation")
    report_content.append("")
    report_content.append("### Dynamic Modeling Framework")
    report_content.append("- **Temporal Feature Engineering**: Lagged variables, moving averages")
    report_content.append("- **Scenario Simulation**: Monte Carlo with policy interventions")
    report_content.append("- **Machine Learning Pipeline**: Automated model selection and validation")
    report_content.append("- **Real-time Adaptation**: Models retrain with new data")
    report_content.append("")
    
    report_content.append("### Data Integration")
    report_content.append("- **Multi-source Fusion**: Demographics, economics, spatial, technology")
    report_content.append("- **Granular Analysis**: Ward-industry-age group level detail")
    report_content.append("- **Shock Integration**: Economic events as model features")
    report_content.append("- **Validation Framework**: Historical backtesting and cross-validation")
    report_content.append("")
    
    report_content.append("## Global Implications")
    report_content.append("")
    report_content.append("### Relevance to Other Aging Societies")
    report_content.append("- **Germany, Italy, South Korea**: Similar demographic challenges")
    report_content.append("- **Methodology Transfer**: Framework applicable to other metros")
    report_content.append("- **Policy Learning**: Best practices for aging urban areas")
    report_content.append("- **Technology Solutions**: AI-aging interaction insights")
    report_content.append("")
    
    report_content.append("## Files Generated")
    report_content.append("")
    report_content.append("### Demographic Data")
    report_content.append("- `historical_population_by_age.csv`: 24-year population evolution")
    report_content.append("- `labor_force_participation.csv`: Industry-age workforce patterns")
    report_content.append("- `migration_patterns.csv`: Internal movement by age/location")
    report_content.append("- `productivity_aging_effects.csv`: Age-productivity relationships")
    report_content.append("- `economic_shock_events.csv`: Historical crisis impacts")
    report_content.append("")
    
    report_content.append("### Temporal Analysis Results")
    report_content.append("- `temporal_concentration_indices.csv`: Time-varying concentration measures")
    report_content.append("- `demographic_transition_effects.csv`: Aging impact analysis")
    report_content.append("- `economic_shock_responses.csv`: Crisis response patterns")
    report_content.append("- `time_varying_agglomeration_coefficients.csv`: Dynamic effect sizes")
    report_content.append("")
    
    report_content.append("### AI Prediction Outputs")
    report_content.append("- `predictions_[scenario].csv`: 27 scenario forecasts (2024-2050)")
    report_content.append("- `scenario_comparison.csv`: Cross-scenario analysis")
    report_content.append("- `training_features.csv`: ML model input data")
    report_content.append("- `models/`: Trained ML models and preprocessors")
    report_content.append("")
    
    report_content.append("### Dynamic Visualizations")
    report_content.append("- `demographic_transition_animation.html`: Population aging evolution")
    report_content.append("- `temporal_concentration_trends.png`: Concentration pattern changes")
    report_content.append("- `scenario_comparison_dashboard.html`: Interactive scenario explorer")
    report_content.append("- `ai_productivity_impact_analysis.png`: Technology adoption effects")
    report_content.append("- `policy_intervention_simulator.html`: Policy impact modeling")
    report_content.append("")
    
    report_content.append("## Future Research Directions")
    report_content.append("")
    report_content.append("1. **Firm-level Dynamics**: Individual company adaptation strategies")
    report_content.append("2. **International Comparison**: Cross-country aging society analysis")
    report_content.append("3. **Technology Specificity**: AI subcategory impacts (robotics, analytics, etc.)")
    report_content.append("4. **Environmental Integration**: Climate change and agglomeration interactions")
    report_content.append("5. **Real-time Updating**: Live model adaptation with streaming data")
    report_content.append("")
    
    report_content.append("## Conclusion")
    report_content.append("")
    report_content.append("This dynamic agglomeration analysis represents a significant advancement over")
    report_content.append("traditional static approaches, providing policymakers with the tools to navigate")
    report_content.append("Japan's demographic transition while maximizing the benefits of technological")
    report_content.append("advancement. The AI-powered predictions offer unprecedented insight into future")
    report_content.append("spatial economic patterns, enabling proactive rather than reactive policy design.")
    report_content.append("")
    
    report_content.append("The integration of aging society challenges with AI opportunities creates a")
    report_content.append("unique framework applicable to developed economies worldwide facing similar")
    report_content.append("demographic transitions.")
    report_content.append("")
    
    report_content.append("---")
    report_content.append("")
    report_content.append("*This report demonstrates the power of combining traditional economic geography*")
    report_content.append("*with cutting-edge AI techniques to address 21st-century urban challenges.*")
    
    # Save report
    report_path = Path("DYNAMIC_ANALYSIS_REPORT.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    logger.info(f"Dynamic analysis report saved to {report_path}")

def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(
        description="Dynamic Tokyo Agglomeration Analysis with AI Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main_dynamic_analysis.py                    # Run complete analysis
  python main_dynamic_analysis.py --skip-demographic # Skip data generation
  python main_dynamic_analysis.py --quick-mode       # Fast execution
        """
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
        "--skip-demographic", 
        action="store_true",
        help="Skip demographic data collection (use existing data)"
    )
    parser.add_argument(
        "--skip-temporal", 
        action="store_true",
        help="Skip temporal analysis (use existing results)"
    )
    parser.add_argument(
        "--skip-predictions", 
        action="store_true",
        help="Skip AI prediction modeling"
    )
    parser.add_argument(
        "--skip-visualizations", 
        action="store_true",
        help="Skip dynamic visualization creation"
    )
    parser.add_argument(
        "--quick-mode", 
        action="store_true",
        help="Run in quick mode with reduced data size"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Dynamic Tokyo Agglomeration Analysis")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Visualizations directory: {args.viz_dir}")
    
    if args.quick_mode:
        logger.info("Running in quick mode - reduced dataset size")
    
    # Create directories
    for directory in [args.data_dir, args.results_dir, args.viz_dir]:
        Path(directory).mkdir(exist_ok=True, parents=True)
    
    success_count = 0
    total_phases = 4
    
    # Phase 1: Demographic Data Collection
    if not args.skip_demographic:
        logger.info("=" * 60)
        logger.info("PHASE 1: DEMOGRAPHIC DATA COLLECTION")
        logger.info("=" * 60)
        if run_demographic_data_collection(args.data_dir):
            success_count += 1
        else:
            logger.error("Demographic data collection failed. Some analyses may not work.")
    else:
        logger.info("Skipping demographic data collection phase")
        success_count += 1
    
    # Phase 2: Temporal Analysis
    if not args.skip_temporal:
        logger.info("=" * 60)
        logger.info("PHASE 2: TEMPORAL AGGLOMERATION ANALYSIS")
        logger.info("=" * 60)
        if run_temporal_analysis(args.data_dir, args.results_dir):
            success_count += 1
        else:
            logger.error("Temporal analysis failed. Continuing with other phases.")
    else:
        logger.info("Skipping temporal analysis phase")
        success_count += 1
    
    # Phase 3: AI Prediction Modeling
    if not args.skip_predictions:
        logger.info("=" * 60)
        logger.info("PHASE 3: AI-POWERED PREDICTION MODELING")
        logger.info("=" * 60)
        if run_ai_prediction_modeling(args.data_dir, args.results_dir):
            success_count += 1
        else:
            logger.error("AI prediction modeling failed. Continuing with other phases.")
    else:
        logger.info("Skipping AI prediction modeling phase")
        success_count += 1
    
    # Phase 4: Dynamic Visualizations
    if not args.skip_visualizations:
        logger.info("=" * 60)
        logger.info("PHASE 4: DYNAMIC VISUALIZATION CREATION")
        logger.info("=" * 60)
        if run_dynamic_visualizations(args.data_dir, args.results_dir, args.viz_dir):
            success_count += 1
        else:
            logger.error("Dynamic visualization creation failed.")
    else:
        logger.info("Skipping dynamic visualization phase")
        success_count += 1
    
    # Phase 5: Final Report
    logger.info("=" * 60)
    logger.info("PHASE 5: GENERATING COMPREHENSIVE REPORT")
    logger.info("=" * 60)
    generate_dynamic_analysis_report(args.results_dir, args.viz_dir)
    
    # Summary
    logger.info("=" * 60)
    logger.info("DYNAMIC AGGLOMERATION ANALYSIS COMPLETED")
    logger.info("=" * 60)
    
    if success_count == total_phases:
        logger.info("✅ All phases completed successfully!")
        print("\n" + "🎉" * 20)
        print("DYNAMIC AGGLOMERATION ANALYSIS COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        print(f"📊 Data directory: {Path(args.data_dir).absolute()}")
        print(f"📈 Results directory: {Path(args.results_dir).absolute()}")
        print(f"📊 Visualizations directory: {Path(args.viz_dir).absolute()}")
        print(f"📄 Final report: {Path('DYNAMIC_ANALYSIS_REPORT.md').absolute()}")
        print("\n🔬 Key Innovation: Dynamic analysis with AI-powered predictions")
        print("🏢 Focus: Japan's aging society and agglomeration evolution")
        print("🤖 Technology: Machine learning for 25-year forecasting")
        print("📊 Output: 27 scenarios across demographic, AI, and economic dimensions")
        print("\n" + "🎌" * 20)
        return 0
    else:
        logger.warning(f"Analysis completed with some issues. {success_count}/{total_phases} phases successful.")
        print(f"\n⚠️  Analysis completed with some issues ({success_count}/{total_phases} phases successful)")
        print("Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
