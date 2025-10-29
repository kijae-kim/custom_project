"""
Multi-Year Data Integration and Correlation Analysis
Merges preprocessed data from 2020-2024 and performs comprehensive analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')


def setup_plot_style():
    """Setup matplotlib style for English labels"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10


def load_preprocessed_data(data_dir, years=None):
    """Load and merge preprocessed data from multiple years"""
    data_dir = Path(data_dir)

    # Find all preprocessed pickle files
    pkl_files = sorted(data_dir.glob('seoul_*_preprocessed.pkl'))

    if years:
        pkl_files = [f for f in pkl_files if any(str(year) in f.stem for year in years)]

    if not pkl_files:
        raise ValueError(f"No preprocessed files found in {data_dir}")

    print(f"Loading {len(pkl_files)} file(s)...")

    dfs = []
    for pkl_file in pkl_files:
        print(f"  Loading: {pkl_file.name}")
        df = pd.read_pickle(pkl_file)
        dfs.append(df)

    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    print(f"\n=== Merged Data ===")
    print(f"Total rows: {len(merged_df):,}")
    print(f"Columns: {merged_df.shape[1]}")
    print(f"Memory: {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Year-Quarter range: {merged_df['STDR_YYQU_CD'].min()} - {merged_df['STDR_YYQU_CD'].max()}")

    return merged_df


def analyze_temporal_correlation(df, save_dir):
    """Analyze correlation patterns across time zones"""
    print("\n=== Time Zone Correlation Analysis ===")

    time_cols = ['TMZON_00_06_SELNG_AMT', 'TMZON_06_11_SELNG_AMT', 'TMZON_11_14_SELNG_AMT',
                 'TMZON_14_17_SELNG_AMT', 'TMZON_17_21_SELNG_AMT', 'TMZON_21_24_SELNG_AMT']

    corr = df[time_cols].corr()

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                xticklabels=['00-06', '06-11', '11-14', '14-17', '17-21', '21-24'],
                yticklabels=['00-06', '06-11', '11-14', '14-17', '17-21', '21-24'])
    plt.title('Sales Correlation by Time Zone (Multi-Year)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    output_file = save_dir / 'correlation_timezone.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # Print insights
    print("\nKey findings:")
    night_corr = corr.loc['TMZON_17_21_SELNG_AMT', 'TMZON_21_24_SELNG_AMT']
    print(f"  - Night hours (17-21 & 21-24) correlation: {night_corr:.3f}")

    return corr


def analyze_weekly_correlation(df, save_dir):
    """Analyze correlation patterns across days of week"""
    print("\n=== Day of Week Correlation Analysis ===")

    day_cols = ['MON_SELNG_AMT', 'TUES_SELNG_AMT', 'WED_SELNG_AMT',
                'THUR_SELNG_AMT', 'FRI_SELNG_AMT', 'SAT_SELNG_AMT', 'SUN_SELNG_AMT']

    corr = df[day_cols].corr()

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                xticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.title('Sales Correlation by Day of Week (Multi-Year)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    output_file = save_dir / 'correlation_weekday.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # Print insights
    print("\nKey findings:")
    weekday_avg = corr.loc['MON_SELNG_AMT':'FRI_SELNG_AMT', 'MON_SELNG_AMT':'FRI_SELNG_AMT'].values
    weekday_avg = weekday_avg[np.triu_indices_from(weekday_avg, k=1)].mean()
    print(f"  - Average weekday correlation: {weekday_avg:.3f}")

    weekend_corr = corr.loc['SAT_SELNG_AMT', 'SUN_SELNG_AMT']
    print(f"  - Weekend correlation (Sat-Sun): {weekend_corr:.3f}")

    return corr


def analyze_age_correlation(df, save_dir):
    """Analyze correlation patterns across age groups"""
    print("\n=== Age Group Correlation Analysis ===")

    age_cols = ['AGRDE_10_SELNG_AMT', 'AGRDE_20_SELNG_AMT', 'AGRDE_30_SELNG_AMT',
                'AGRDE_40_SELNG_AMT', 'AGRDE_50_SELNG_AMT', 'AGRDE_60_ABOVE_SELNG_AMT']

    corr = df[age_cols].corr()

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                xticklabels=['10s', '20s', '30s', '40s', '50s', '60+'],
                yticklabels=['10s', '20s', '30s', '40s', '50s', '60+'])
    plt.title('Sales Correlation by Age Group (Multi-Year)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    output_file = save_dir / 'correlation_age.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # Print insights
    print("\nKey findings:")
    young_corr = corr.loc['AGRDE_20_SELNG_AMT', 'AGRDE_30_SELNG_AMT']
    print(f"  - Young adult correlation (20s-30s): {young_corr:.3f}")

    core_corr = corr.loc['AGRDE_20_SELNG_AMT':'AGRDE_40_SELNG_AMT',
                         'AGRDE_20_SELNG_AMT':'AGRDE_40_SELNG_AMT'].values
    core_corr = core_corr[np.triu_indices_from(core_corr, k=1)].mean()
    print(f"  - Core consumer group (20s-40s) avg correlation: {core_corr:.3f}")

    senior_corr = corr.loc['AGRDE_50_SELNG_AMT', 'AGRDE_60_ABOVE_SELNG_AMT']
    print(f"  - Senior correlation (50s-60+): {senior_corr:.3f}")

    return corr


def analyze_time_trends(df, save_dir):
    """Analyze sales trends over time"""
    print("\n=== Time Series Trend Analysis ===")

    # Group by year-quarter
    time_series = df.groupby('STDR_YYQU_CD').agg({
        'THSMON_SELNG_AMT': 'sum',
        'THSMON_SELNG_CO': 'sum'
    }).reset_index()

    time_series['avg_transaction'] = time_series['THSMON_SELNG_AMT'] / time_series['THSMON_SELNG_CO']

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Total sales trend
    axes[0].plot(time_series['STDR_YYQU_CD'], time_series['THSMON_SELNG_AMT'] / 1e9,
                 marker='o', linewidth=2, markersize=8)
    axes[0].set_title('Total Sales Trend by Quarter', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Year-Quarter')
    axes[0].set_ylabel('Total Sales (Billion KRW)')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    # Average transaction amount trend
    axes[1].plot(time_series['STDR_YYQU_CD'], time_series['avg_transaction'] / 1000,
                 marker='s', linewidth=2, markersize=8, color='coral')
    axes[1].set_title('Average Transaction Amount Trend', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Year-Quarter')
    axes[1].set_ylabel('Average Amount (Thousand KRW)')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    output_file = save_dir / 'trend_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # Print statistics
    print("\nTrend statistics:")
    total_growth = (time_series['THSMON_SELNG_AMT'].iloc[-1] / time_series['THSMON_SELNG_AMT'].iloc[0] - 1) * 100
    print(f"  - Total sales growth: {total_growth:+.2f}%")

    avg_growth = (time_series['avg_transaction'].iloc[-1] / time_series['avg_transaction'].iloc[0] - 1) * 100
    print(f"  - Average transaction growth: {avg_growth:+.2f}%")

    return time_series


def analyze_by_district_type(df, save_dir):
    """Analyze correlation patterns by district type"""
    print("\n=== District Type Analysis ===")

    district_stats = df.groupby('TRDAR_SE_CD_NM').agg({
        'THSMON_SELNG_AMT': ['sum', 'mean', 'count'],
        'WKEND_SELNG_AMT': 'sum',
        'MDWK_SELNG_AMT': 'sum'
    }).round(2)

    # Calculate weekend ratio
    district_stats['weekend_ratio'] = (district_stats[('WKEND_SELNG_AMT', 'sum')] /
                                       (district_stats[('WKEND_SELNG_AMT', 'sum')] +
                                        district_stats[('MDWK_SELNG_AMT', 'sum')]) * 100)

    district_stats.columns = ['total_sales', 'avg_sales', 'count', 'weekend_sales', 'weekday_sales', 'weekend_ratio']
    district_stats = district_stats.sort_values('total_sales', ascending=False)

    print("\nDistrict type statistics:")
    print(district_stats)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Total sales by district type
    axes[0].bar(district_stats.index, district_stats['total_sales'] / 1e12)
    axes[0].set_title('Total Sales by District Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Total Sales (Trillion KRW)')
    axes[0].tick_params(axis='x', rotation=45)

    # Weekend ratio by district type
    axes[1].bar(district_stats.index, district_stats['weekend_ratio'], color='coral')
    axes[1].set_title('Weekend Sales Ratio by District Type', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Weekend Sales Ratio (%)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    output_file = save_dir / 'district_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    return district_stats


def create_correlation_summary(df, save_dir):
    """Create comprehensive correlation summary"""
    print("\n=== Comprehensive Correlation Summary ===")

    # Select key variables for overall correlation
    key_vars = [
        'THSMON_SELNG_AMT',
        'MDWK_SELNG_AMT',
        'WKEND_SELNG_AMT',
        'TMZON_17_21_SELNG_AMT',
        'TMZON_21_24_SELNG_AMT',
        'AGRDE_20_SELNG_AMT',
        'AGRDE_30_SELNG_AMT',
        'AGRDE_40_SELNG_AMT',
        'ML_SELNG_AMT',
        'FML_SELNG_AMT'
    ]

    corr = df[key_vars].corr()

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Key Variables Correlation Matrix (Multi-Year)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    output_file = save_dir / 'correlation_comprehensive.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    return corr


def main():
    parser = argparse.ArgumentParser(description='Merge and analyze multi-year data')
    parser.add_argument('--input', '-i', default='data/processed', help='Directory with preprocessed data')
    parser.add_argument('--output', '-o', default='analysis_results', help='Output directory for analysis results')
    parser.add_argument('--years', nargs='+', type=int, help='Specific years to include')

    args = parser.parse_args()

    # Setup
    setup_plot_style()
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Multi-Year Data Integration and Analysis")
    print("="*60)

    # Load and merge data
    df = load_preprocessed_data(args.input, args.years)

    # Save merged data
    merged_output = save_dir / 'merged_data_all_years.pkl'
    df.to_pickle(merged_output)
    print(f"\nMerged data saved: {merged_output}")

    # Perform analyses
    analyze_temporal_correlation(df, save_dir)
    analyze_weekly_correlation(df, save_dir)
    analyze_age_correlation(df, save_dir)
    analyze_time_trends(df, save_dir)
    analyze_by_district_type(df, save_dir)
    create_correlation_summary(df, save_dir)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Results saved in: {save_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
