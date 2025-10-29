"""
전처리된 데이터를 통합하여 분석용 CSV 파일 생성

- full_data.csv: 전체 원본 데이터 (모든 컬럼 포함)
- processed_data.csv: 핵심 분석 컬럼만 포함 (경제활동가능인구 매출 포함)
"""

import pandas as pd
from pathlib import Path

def load_all_preprocessed_data(data_dir='data/processed'):
    """전처리된 모든 pickle 파일 로드 및 통합"""
    print(f"Loading preprocessed data from {data_dir}...")
    pickle_files = sorted(Path(data_dir).glob('seoul_*_preprocessed.pkl'))

    if not pickle_files:
        raise FileNotFoundError(f"No preprocessed pickle files found in {data_dir}")

    print(f"Found {len(pickle_files)} files:")
    for pf in pickle_files:
        print(f"  - {pf.name}")

    df_list = [pd.read_pickle(file) for file in pickle_files]
    df = pd.concat(df_list, ignore_index=True)

    print(f"\n✅ Loaded and concatenated {len(pickle_files)} files")
    print(f"   Total records: {len(df):,}")
    print(f"   Total columns: {df.shape[1]}")

    return df

def create_full_data(df, output_dir='data'):
    """전체 데이터를 CSV로 저장"""
    print("\n" + "="*60)
    print("Creating full_data.csv...")
    print("="*60)

    # 연도와 분기 컬럼 추가
    df['YEAR'] = df['STDR_YYQU_CD'].str[:4]
    df['QUARTER'] = df['STDR_YYQU_CD'].str[4:]

    # 경제활동가능인구 매출 컬럼 추가
    df['ECNMY_ACTIVE_POP_SELNG_AMT'] = (
        df['AGRDE_20_SELNG_AMT'] +
        df['AGRDE_30_SELNG_AMT'] +
        df['AGRDE_40_SELNG_AMT'] +
        df['AGRDE_50_SELNG_AMT'] +
        df['AGRDE_60_ABOVE_SELNG_AMT']
    )

    output_path = Path(output_dir) / 'full_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Saved: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {df.shape[1]}")
    print(f"   Size: {file_size_mb:.2f} MB")

    return df

def create_processed_data(df, output_dir='data'):
    """핵심 분석 컬럼만 포함한 데이터 생성"""
    print("\n" + "="*60)
    print("Creating processed_data.csv...")
    print("="*60)

    # 연도와 분기가 이미 추가되어 있음 (create_full_data에서)
    # 경제활동가능인구 매출도 이미 추가되어 있음

    # 필요한 컬럼만 선택
    essential_columns = [
        # 기본 정보
        'STDR_YYQU_CD',           # 기준년분기코드
        'YEAR',                   # 연도
        'QUARTER',                # 분기

        # 지역 정보
        'SIGNGU_CD',              # 자치구 코드
        'SIGNGU_NM',              # 자치구명
        'ADSTRD_CD',              # 행정동 코드
        'ADSTRD_NM',              # 행정동명

        # 상권 정보
        'TRDAR_SE_CD_NM',         # 상권 유형명 (골목상권, 발달상권 등)
        'TRDAR_CD',               # 상권 코드
        'TRDAR_CD_NM',            # 상권명

        # 업종 정보
        'SVC_INDUTY_CD_NM',       # 서비스업종명

        # 총 매출
        'THSMON_SELNG_AMT',       # 당월 총 매출액

        # 주중/주말 매출
        'MDWK_SELNG_AMT',         # 주중 매출액
        'WKEND_SELNG_AMT',        # 주말 매출액

        # 요일별 매출
        'MON_SELNG_AMT',          # 월요일 매출액
        'TUES_SELNG_AMT',         # 화요일 매출액
        'WED_SELNG_AMT',          # 수요일 매출액
        'THUR_SELNG_AMT',         # 목요일 매출액
        'FRI_SELNG_AMT',          # 금요일 매출액
        'SAT_SELNG_AMT',          # 토요일 매출액
        'SUN_SELNG_AMT',          # 일요일 매출액

        # 시간대별 매출
        'TMZON_00_06_SELNG_AMT',  # 00-06시 매출액
        'TMZON_06_11_SELNG_AMT',  # 06-11시 매출액
        'TMZON_11_14_SELNG_AMT',  # 11-14시 매출액
        'TMZON_14_17_SELNG_AMT',  # 14-17시 매출액
        'TMZON_17_21_SELNG_AMT',  # 17-21시 매출액
        'TMZON_21_24_SELNG_AMT',  # 21-24시 매출액

        # 성별 매출
        'ML_SELNG_AMT',           # 남성 매출액
        'FML_SELNG_AMT',          # 여성 매출액

        # 경제활동가능인구 매출 (20대~60대)
        'ECNMY_ACTIVE_POP_SELNG_AMT',  # 경제활동가능인구 매출 (20대~60대 합계)

        # 참고용 개별 연령대 (10대, 60대 이상은 제외하고 경제활동인구만)
        'AGRDE_20_SELNG_AMT',     # 20대 매출액
        'AGRDE_30_SELNG_AMT',     # 30대 매출액
        'AGRDE_40_SELNG_AMT',     # 40대 매출액
        'AGRDE_50_SELNG_AMT',     # 50대 매출액
        'AGRDE_60_ABOVE_SELNG_AMT',  # 60대 이상 매출액
    ]

    # 컬럼 존재 여부 확인
    missing_cols = [col for col in essential_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing columns: {missing_cols}")
        essential_columns = [col for col in essential_columns if col in df.columns]

    # 핵심 컬럼만 선택
    df_processed = df[essential_columns].copy()

    # 데이터 타입 최적화 (메모리 절약)
    print("\n데이터 타입 최적화 중...")

    # 숫자형 컬럼들을 int64로 변환 (매출액)
    amount_cols = [col for col in df_processed.columns if 'SELNG_AMT' in col]
    for col in amount_cols:
        df_processed[col] = df_processed[col].fillna(0).astype('int64')

    # 코드 컬럼들을 문자열로
    code_cols = ['STDR_YYQU_CD', 'SIGNGU_CD', 'ADSTRD_CD', 'TRDAR_CD']
    for col in code_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)

    # 저장
    output_path = Path(output_dir) / 'processed_data.csv'
    df_processed.to_csv(output_path, index=False, encoding='utf-8-sig')

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Saved: {output_path}")
    print(f"   Rows: {len(df_processed):,}")
    print(f"   Columns: {df_processed.shape[1]}")
    print(f"   Size: {file_size_mb:.2f} MB")

    # 컬럼 정보 출력
    print("\n📋 포함된 컬럼 정보:")
    print("\n1. 기본 정보 (3개):")
    print("   - STDR_YYQU_CD, YEAR, QUARTER")

    print("\n2. 지역 정보 (4개):")
    print("   - SIGNGU_CD, SIGNGU_NM (자치구)")
    print("   - ADSTRD_CD, ADSTRD_NM (행정동)")

    print("\n3. 상권 정보 (3개):")
    print("   - TRDAR_SE_CD_NM (상권 유형)")
    print("   - TRDAR_CD, TRDAR_CD_NM (상권)")

    print("\n4. 업종 정보 (1개):")
    print("   - SVC_INDUTY_CD_NM")

    print("\n5. 총 매출 (1개):")
    print("   - THSMON_SELNG_AMT")

    print("\n6. 주중/주말 매출 (2개):")
    print("   - MDWK_SELNG_AMT, WKEND_SELNG_AMT")

    print("\n7. 요일별 매출 (7개):")
    print("   - MON~SUN_SELNG_AMT")

    print("\n8. 시간대별 매출 (6개):")
    print("   - TMZON_00_06 ~ TMZON_21_24_SELNG_AMT")

    print("\n9. 성별 매출 (2개):")
    print("   - ML_SELNG_AMT, FML_SELNG_AMT")

    print("\n10. 경제활동가능인구 매출 (6개):")
    print("   - ECNMY_ACTIVE_POP_SELNG_AMT (20~60대 합계) ⭐")
    print("   - AGRDE_20~60_ABOVE_SELNG_AMT (개별 연령대)")

    return df_processed

def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("전처리 데이터 통합 작업 시작")
    print("="*60)

    # 1. 전처리된 데이터 로드
    df = load_all_preprocessed_data()

    # 2. full_data.csv 생성 (모든 컬럼)
    df_full = create_full_data(df)

    # 3. processed_data.csv 생성 (핵심 컬럼만)
    df_processed = create_processed_data(df_full)

    print("\n" + "="*60)
    print("✅ 모든 작업 완료!")
    print("="*60)

    print("\n생성된 파일:")
    print("  1. data/full_data.csv")
    print("     - 전체 원본 데이터 (모든 컬럼 포함)")
    print(f"     - {len(df_full):,} rows × {df_full.shape[1]} columns")

    print("\n  2. data/processed_data.csv")
    print("     - 핵심 분석 컬럼만 포함")
    print(f"     - {len(df_processed):,} rows × {df_processed.shape[1]} columns")
    print("     - 경제활동가능인구 매출 컬럼 포함 ⭐")

    print("\n💡 사용 방법:")
    print("   - 상세 분석: full_data.csv")
    print("   - 일반 분석: processed_data.csv (추천)")

if __name__ == '__main__':
    main()
