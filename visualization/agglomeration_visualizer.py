#!/usr/bin/env python3
"""
Agglomeration Analysis Visualizer
Creates comprehensive visualizations for Tokyo productivity agglomeration analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
from pathlib import Path
import logging
from typing import Dict, List, Optional

class AgglomerationVisualizer:
    """
    Creates visualizations for agglomeration analysis results
    """
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results", 
                 viz_dir: str = "visualizations"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load analysis results
        """
        results = {}
        
        try:
            results['concentration'] = pd.read_csv(self.results_dir / "concentration_indices.csv")
            results['agglomeration'] = pd.read_csv(self.results_dir / "agglomeration_effects.csv")
            results['ai_impact'] = pd.read_csv(self.results_dir / "ai_productivity_impact.csv")
            results['summary'] = pd.read_csv(self.results_dir / "agglomeration_comprehensive_summary.csv")
            
            # Load spatial data
            results['spatial'] = pd.read_csv(self.data_dir / "tokyo_spatial_distribution.csv")
            results['productivity'] = pd.read_csv(self.data_dir / "tokyo_labor_productivity.csv")
            
            self.logger.info("All result datasets loaded successfully")
        except FileNotFoundError as e:
            self.logger.error(f"Result file not found: {e}")
            raise
        
        return results
    
    def create_concentration_heatmap(self, results: Dict[str, pd.DataFrame]):
        """
        Create heatmap of industry concentration indices
        """
        self.logger.info("Creating concentration indices heatmap...")
        
        concentration_data = results['concentration'].copy()
        
        # Prepare data for heatmap
        metrics = ['gini_coefficient', 'herfindahl_index', 'avg_location_quotient']
        heatmap_data = concentration_data[['industry_name'] + metrics].set_index('industry_name')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Normalize data for better visualization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        heatmap_data_scaled = pd.DataFrame(
            scaler.fit_transform(heatmap_data),
            index=heatmap_data.index,
            columns=heatmap_data.columns
        )
        
        # Create heatmap
        sns.heatmap(heatmap_data_scaled, 
                   annot=True, 
                   cmap='RdYlBu_r',
                   center=0,
                   fmt='.2f',
                   cbar_kws={'label': 'Standardized Score'})
        
        plt.title('Industry Concentration Indices in Tokyo\n(Standardized Scores)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Concentration Measures', fontsize=12)
        plt.ylabel('Industry', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / "concentration_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Concentration heatmap saved to {output_path}")
    
    def create_agglomeration_effects_chart(self, results: Dict[str, pd.DataFrame]):
        """
        Create chart showing agglomeration effects by industry
        """
        self.logger.info("Creating agglomeration effects chart...")
        
        agglomeration_data = results['agglomeration'].copy()
        
        # Sort by mean productivity
        agglomeration_data = agglomeration_data.sort_values('mean_productivity', ascending=True)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Employment density coefficient
        bars1 = ax1.barh(agglomeration_data['industry_name'], 
                         agglomeration_data['employment_density_coeff'])
        ax1.set_title('Employment Density Effect on Productivity', fontweight='bold')
        ax1.set_xlabel('Coefficient')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Color bars based on positive/negative
        for bar, coeff in zip(bars1, agglomeration_data['employment_density_coeff']):
            bar.set_color('green' if coeff > 0 else 'red')
        
        # 2. Market potential coefficient
        bars2 = ax2.barh(agglomeration_data['industry_name'], 
                         agglomeration_data['market_potential_coeff'])
        ax2.set_title('Market Potential Effect on Productivity', fontweight='bold')
        ax2.set_xlabel('Coefficient')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        for bar, coeff in zip(bars2, agglomeration_data['market_potential_coeff']):
            bar.set_color('green' if coeff > 0 else 'red')
        
        # 3. Diversity coefficient
        bars3 = ax3.barh(agglomeration_data['industry_name'], 
                         agglomeration_data['diversity_coeff'])
        ax3.set_title('Industry Diversity Effect on Productivity', fontweight='bold')
        ax3.set_xlabel('Coefficient')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        for bar, coeff in zip(bars3, agglomeration_data['diversity_coeff']):
            bar.set_color('green' if coeff > 0 else 'red')
        
        # 4. R-squared values
        bars4 = ax4.barh(agglomeration_data['industry_name'], 
                         agglomeration_data['r_squared'])
        ax4.set_title('Model Fit (R²)', fontweight='bold')
        ax4.set_xlabel('R-squared')
        ax4.set_xlim(0, 1)
        
        # Color by R-squared value
        for bar, r2 in zip(bars4, agglomeration_data['r_squared']):
            if r2 > 0.7:
                bar.set_color('darkgreen')
            elif r2 > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Adjust layout
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Agglomeration Effects on Industry Productivity in Tokyo', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / "agglomeration_effects.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Agglomeration effects chart saved to {output_path}")
    
    def create_ai_impact_visualization(self, results: Dict[str, pd.DataFrame]):
        """
        Create visualization of AI impact on productivity
        """
        self.logger.info("Creating AI impact visualization...")
        
        ai_data = results['ai_impact'].copy()
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. AI Adoption Rate by Industry
        ai_sorted = ai_data.sort_values('ai_adoption_rate', ascending=True)
        bars1 = ax1.barh(ai_sorted['industry_name'], ai_sorted['ai_adoption_rate'] * 100)
        ax1.set_title('AI Adoption Rate by Industry', fontweight='bold')
        ax1.set_xlabel('Adoption Rate (%)')
        
        # Color by adoption level
        for bar, rate in zip(bars1, ai_sorted['ai_adoption_rate']):
            if rate > 0.25:
                bar.set_color('darkgreen')
            elif rate > 0.15:
                bar.set_color('orange')
            else:
                bar.set_color('lightcoral')
        
        # 2. Productivity Gain from AI
        ai_gain_sorted = ai_data.sort_values('productivity_gain_percent', ascending=True)
        bars2 = ax2.barh(ai_gain_sorted['industry_name'], ai_gain_sorted['productivity_gain_percent'])
        ax2.set_title('Productivity Gain from AI Adoption', fontweight='bold')
        ax2.set_xlabel('Productivity Gain (%)')
        
        for bar, gain in zip(bars2, ai_gain_sorted['productivity_gain_percent']):
            if gain > 5:
                bar.set_color('darkgreen')
            elif gain > 2:
                bar.set_color('orange')
            else:
                bar.set_color('lightcoral')
        
        # 3. Scatter: AI Adoption vs Productivity Gain
        scatter = ax3.scatter(ai_data['ai_adoption_rate'] * 100, 
                            ai_data['productivity_gain_percent'],
                            s=ai_data['ai_investment_million_yen_per_company'] * 5,
                            alpha=0.7, c=range(len(ai_data)), cmap='viridis')
        
        ax3.set_title('AI Adoption vs Productivity Gain\n(Size = Investment)', fontweight='bold')
        ax3.set_xlabel('AI Adoption Rate (%)')
        ax3.set_ylabel('Productivity Gain (%)')
        
        # Add trend line
        z = np.polyfit(ai_data['ai_adoption_rate'] * 100, ai_data['productivity_gain_percent'], 1)
        p = np.poly1d(z)
        ax3.plot(ai_data['ai_adoption_rate'] * 100, p(ai_data['ai_adoption_rate'] * 100), 
                "r--", alpha=0.8)
        
        # 4. Investment vs Productivity Gain
        bars4 = ax4.scatter(ai_data['ai_investment_million_yen_per_company'],
                           ai_data['productivity_gain_percent'],
                           s=100, alpha=0.7)
        ax4.set_title('AI Investment vs Productivity Gain', fontweight='bold')
        ax4.set_xlabel('AI Investment (Million Yen per Company)')
        ax4.set_ylabel('Productivity Gain (%)')
        
        # Add labels for top performers
        for idx, row in ai_data.iterrows():
            if row['productivity_gain_percent'] > ai_data['productivity_gain_percent'].quantile(0.8):
                ax4.annotate(row['industry_name'][:10], 
                           (row['ai_investment_million_yen_per_company'], 
                            row['productivity_gain_percent']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Adjust layout
        for ax in [ax1, ax2, ax3, ax4]:
            if ax in [ax1, ax2]:
                ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('AI Impact on Industry Productivity in Tokyo', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / "ai_impact_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"AI impact visualization saved to {output_path}")
    
    def create_interactive_tokyo_map(self, results: Dict[str, pd.DataFrame]):
        """
        Create interactive map of Tokyo with productivity data
        """
        self.logger.info("Creating interactive Tokyo map...")
        
        spatial_data = results['spatial']
        productivity_data = results['productivity']
        
        # Get latest productivity data
        latest_prod = productivity_data[productivity_data['year'] == 2023]
        
        # Aggregate by ward
        ward_productivity = latest_prod.groupby('ward').agg({
            'productivity_thousand_yen_per_employee': 'mean'
        }).reset_index()
        
        # Merge with spatial data
        map_data = spatial_data.merge(ward_productivity, on='ward', how='left')
        
        # Create base map centered on Tokyo
        tokyo_center = [35.6762, 139.6503]
        m = folium.Map(location=tokyo_center, zoom_start=11, 
                      tiles='CartoDB positron')
        
        # Add ward productivity as circles
        for idx, row in map_data.iterrows():
            if pd.notna(row['productivity_thousand_yen_per_employee']):
                # Circle size based on productivity
                radius = (row['productivity_thousand_yen_per_employee'] / 1000) * 2
                
                # Color based on ward type
                color = 'red' if row['ward_type'] == 'central' else 'blue'
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=radius,
                    popup=f"{row['ward']}<br>Productivity: {row['productivity_thousand_yen_per_employee']:.1f} thousand yen",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Ward Productivity</h4>
        <i class="fa fa-circle" style="color:red"></i> Central Wards<br>
        <i class="fa fa-circle" style="color:blue"></i> Outer Wards<br>
        Circle size ∝ Productivity
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        output_path = self.viz_dir / "tokyo_productivity_map.html"
        m.save(str(output_path))
        
        self.logger.info(f"Interactive Tokyo map saved to {output_path}")
    
    def create_comprehensive_dashboard(self, results: Dict[str, pd.DataFrame]):
        """
        Create comprehensive dashboard with Plotly
        """
        self.logger.info("Creating comprehensive dashboard...")
        
        summary_data = results['summary']
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Agglomeration Score by Industry',
                'AI Adoption vs Agglomeration Score',
                'Productivity vs Employment Concentration',
                'Industry Performance Matrix'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Agglomeration Score ranking
        top_industries = summary_data.nlargest(15, 'agglomeration_score')
        fig.add_trace(
            go.Bar(
                x=top_industries['agglomeration_score'],
                y=top_industries['industry_name'],
                orientation='h',
                name='Agglomeration Score',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. AI Adoption vs Agglomeration Score
        fig.add_trace(
            go.Scatter(
                x=summary_data['ai_adoption_rate'],
                y=summary_data['agglomeration_score'],
                mode='markers+text',
                text=summary_data['industry_code'],
                textposition='top center',
                name='Industries',
                marker=dict(size=10, opacity=0.7)
            ),
            row=1, col=2
        )
        
        # 3. Productivity vs Concentration
        fig.add_trace(
            go.Scatter(
                x=summary_data['gini_coefficient'],
                y=summary_data['mean_productivity'],
                mode='markers+text',
                text=summary_data['industry_code'],
                textposition='top center',
                name='Concentration vs Productivity',
                marker=dict(size=10, opacity=0.7, color='orange')
            ),
            row=2, col=1
        )
        
        # 4. Performance Matrix (Productivity vs AI Gain)
        fig.add_trace(
            go.Scatter(
                x=summary_data['productivity_gain_percent'],
                y=summary_data['mean_productivity'],
                mode='markers+text',
                text=summary_data['industry_code'],
                textposition='top center',
                name='Performance Matrix',
                marker=dict(
                    size=summary_data['ai_adoption_rate'] * 50,
                    opacity=0.7,
                    color='green'
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Tokyo Industry Agglomeration & AI Impact Dashboard",
            title_x=0.5,
            showlegend=False,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Agglomeration Score", row=1, col=1)
        fig.update_xaxes(title_text="AI Adoption Rate", row=1, col=2)
        fig.update_xaxes(title_text="Gini Coefficient", row=2, col=1)
        fig.update_xaxes(title_text="Productivity Gain from AI (%)", row=2, col=2)
        
        fig.update_yaxes(title_text="Industry", row=1, col=1)
        fig.update_yaxes(title_text="Agglomeration Score", row=1, col=2)
        fig.update_yaxes(title_text="Mean Productivity", row=2, col=1)
        fig.update_yaxes(title_text="Mean Productivity", row=2, col=2)
        
        # Save dashboard
        output_path = self.viz_dir / "comprehensive_dashboard.html"
        fig.write_html(str(output_path))
        
        self.logger.info(f"Comprehensive dashboard saved to {output_path}")
    
    def run_all_visualizations(self):
        """
        Run all visualization functions
        """
        self.logger.info("Starting comprehensive visualization process...")
        
        # Load results
        results = self.load_results()
        
        # Create all visualizations
        self.create_concentration_heatmap(results)
        self.create_agglomeration_effects_chart(results)
        self.create_ai_impact_visualization(results)
        self.create_interactive_tokyo_map(results)
        self.create_comprehensive_dashboard(results)
        
        self.logger.info("All visualizations completed successfully!")

if __name__ == "__main__":
    visualizer = AgglomerationVisualizer()
    visualizer.run_all_visualizations()
    print("All visualizations created successfully!")
