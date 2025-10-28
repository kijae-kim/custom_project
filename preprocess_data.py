"""
Seoul Commercial District Sales Data Preprocessing Script
Processes sales data from 2020 to 2024 and enriches it with administrative district information.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import argparse

def get_location_mapping_table():
    """
    Loads the commercial district shapefile and returns a clean mapping table
    from commercial district code to administrative district info.
    """
    shapefile_path = 'data/geo_code/서울시 상권분석서비스(영역-상권).shp'
    print(f"--- Loading location mapping data from: {shapefile_path} ---")
    try:
        gdf = gpd.read_file(shapefile_path, encoding='utf-8')
    except Exception as e:
        print(f"!!! Critical Error: Could not read the shapefile. {e} !!!")
        print("Please ensure the file exists and is not corrupted.")
        return None

    # Create a clean mapping DataFrame
    mapping_df = gdf[['TRDAR_CD', 'SIGNGU_CD', 'SIGNGU_CD_', 'ADSTRD_CD', 'ADSTRD_CD_']].copy()
    mapping_df.rename(columns={'SIGNGU_CD_': 'SIGNGU_NM', 'ADSTRD_CD_': 'ADSTRD_NM'}, inplace=True)
    mapping_df.drop_duplicates(subset=['TRDAR_CD'], inplace=True)
    
    print("--- Location mapping table created successfully. ---\n")
    return mapping_df

def load_csv_data(file_path, encoding='euc-kr'):
    """Load CSV data with specified encoding"""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path, encoding=encoding)
    print(f"  Loaded: {len(df):,} rows x {df.shape[1]} columns")
    return df

def remove_unnecessary_columns(df):
    """Remove redundant and unnecessary columns"""
    columns_to_remove = [
        '상권_구분_코드', '서비스_업종_코드', '당월_매출_건수', '주중_매출_건수',
        '주말_매출_건수', '월요일_매출_건수', '화요일_매출_건수', '수요일_매출_건수',
        '목요일_매출_건수', '금요일_매출_건수', '토요일_매출_건수', '일요일_매출_건수',
        '시간대_건수~06_매출_건수', '시간대_건수~11_매출_건수', '시간대_건수~14_매출_건수',
        '시간대_건수~17_매출_건수', '시간대_건수~21_매출_건수', '시간대_건수~24_매출_건수',
        '남성_매출_건수', '여성_매출_건수', '연령대_10_매출_건수', '연령대_20_매출_건수',
        '연령대_30_매출_건수', '연령대_40_매출_건수', '연령대_50_매출_건수',
        '연령대_60_이상_매출_건수'
    ]
    existing_cols = [col for col in columns_to_remove if col in df.columns]
    df = df.drop(columns=existing_cols)
    print(f"  Removed {len(existing_cols)} unnecessary columns")
    return df

def rename_columns(df):
    """Convert Korean column names to English"""
    column_mapping = {
        '기준_년분기_코드': 'STDR_YYQU_CD', '상권_구분_코드_명': 'TRDAR_SE_CD_NM',
        '상권_코드': 'TRDAR_CD', '상권_코드_명': 'TRDAR_CD_NM', '서비스_업종_코드_명': 'SVC_INDUTY_CD_NM',
        '당월_매출_금액': 'THSMON_SELNG_AMT', '주중_매출_금액': 'MDWK_SELNG_AMT',
        '주말_매출_금액': 'WKEND_SELNG_AMT', '월요일_매출_금액': 'MON_SELNG_AMT',
        '화요일_매출_금액': 'TUES_SELNG_AMT', '수요일_매출_금액': 'WED_SELNG_AMT',
        '목요일_매출_금액': 'THUR_SELNG_AMT', '금요일_매출_금액': 'FRI_SELNG_AMT',
        '토요일_매출_금액': 'SAT_SELNG_AMT', '일요일_매출_금액': 'SUN_SELNG_AMT',
        '시간대_00~06_매출_금액': 'TMZON_00_06_SELNG_AMT', '시간대_06~11_매출_금액': 'TMZON_06_11_SELNG_AMT',
        '시간대_11~14_매출_금액': 'TMZON_11_14_SELNG_AMT', '시간대_14~17_매출_금액': 'TMZON_14_17_SELNG_AMT',
        '시간대_17~21_매출_금액': 'TMZON_17_21_SELNG_AMT', '시간대_21~24_매출_금액': 'TMZON_21_24_SELNG_AMT',
        '남성_매출_금액': 'ML_SELNG_AMT', '여성_매출_금액': 'FML_SELNG_AMT',
        '연령대_10_매출_금액': 'AGRDE_10_SELNG_AMT', '연령대_20_매출_금액': 'AGRDE_20_SELNG_AMT',
        '연령대_30_매출_금액': 'AGRDE_30_SELNG_AMT', '연령대_40_매출_금액': 'AGRDE_40_SELNG_AMT',
        '연령대_50_매출_금액': 'AGRDE_50_SELNG_AMT', '연령대_60_이상_매출_금액': 'AGRDE_60_ABOVE_SELNG_AMT'
    }
    df = df.rename(columns=column_mapping)
    print(f"  Renamed {len(column_mapping)} columns to English")
    return df

def check_data_quality(df):
    """Check for missing values and duplicates"""
    print("\n=== Data Quality Check ===")
    missing = df.isnull().sum()
    missing_total = missing.sum()
    if missing_total > 0:
        print(f"  Total missing values: {missing_total:,}")
        # Show columns with missing values
        print(missing[missing > 0])
    else:
        print("  Missing values: None")
    duplicates = df.duplicated().sum()
    print(f"  Duplicate rows: {duplicates:,}")
    return missing_total, duplicates

def clean_data(df):
    """Clean data by removing duplicates and handling missing values"""
    print("\n=== Data Cleaning ===")
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"  Duplicates removed: {removed:,}")

    # Handle missing values - specifically for location data if merge fails
    if 'ADSTRD_NM' in df.columns and df['ADSTRD_NM'].isnull().any():
        print(f"  Missing location data for {df['ADSTRD_NM'].isnull().sum():,} rows. Filling with 'Unknown'.")
        df[['SIGNGU_NM', 'ADSTRD_NM']] = df[['SIGNGU_NM', 'ADSTRD_NM']].fillna('Unknown')

    missing_counts = df.isnull().sum().sum()
    if missing_counts > 0:
        df = df.fillna(0)
        print(f"  Other missing values filled with 0: {missing_counts:,}")

    df['STDR_YYQU_CD'] = df['STDR_YYQU_CD'].astype(str)
    print(f"  Final data: {len(df):,} rows x {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df

def process_file(input_file, output_dir, mapping_df):
    """Process a single data file and merge it with location data"""
    df = load_csv_data(input_file)
    df = remove_unnecessary_columns(df)
    df = rename_columns(df)

    # Merge with location data
    print("  Merging with location data...")
    df['TRDAR_CD'] = df['TRDAR_CD'].astype(str)
    mapping_df['TRDAR_CD'] = mapping_df['TRDAR_CD'].astype(str)
    df = pd.merge(df, mapping_df, on='TRDAR_CD', how='left')
    print(f"  Location data merged. New columns added: {list(mapping_df.columns.drop('TRDAR_CD'))}")

    check_data_quality(df)
    df = clean_data(df)

    # Save processed data
    file_name = Path(input_file).stem
    output_csv = output_dir / f"{file_name}_preprocessed.csv"
    output_pkl = output_dir / f"{file_name}_preprocessed.pkl"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig') # Use utf-8-sig for better CSV compatibility
    df.to_pickle(output_pkl)
    print(f"\n=== Saved ===\n  CSV: {output_csv}\n  Pickle: {output_pkl}")
    return df

def main():
    parser = argparse.ArgumentParser(description='Preprocess Seoul commercial district sales data')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file or directory')
    parser.add_argument('--output', '-o', default='data/processed', help='Output directory')
    parser.add_argument('--years', nargs='+', type=int, help='Specific years to process (e.g., 2020 2021)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the location mapping table once at the beginning
    mapping_df = get_location_mapping_table()
    if mapping_df is None:
        return # Stop execution if mapping table fails to load

    input_path = Path(args.input)
    if input_path.is_file():
        print(f"\n{'='*60}\nProcessing: {input_path.name}\n{'='*60}")
        process_file(input_path, output_dir, mapping_df)
    elif input_path.is_dir():
        csv_files = list(input_path.glob('seoul_*.csv'))
        if args.years:
            csv_files = [f for f in csv_files if any(str(y) in f.stem for y in args.years)]
        
        if not csv_files:
            print(f"No CSV files found in {input_path} for the specified years.")
            return

        print(f"\nFound {len(csv_files)} file(s) to process.")
        for csv_file in sorted(csv_files):
            print(f"\n{'='*60}\nProcessing: {csv_file.name}\n{'='*60}")
            process_file(csv_file, output_dir, mapping_df)
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == '__main__':
    main()