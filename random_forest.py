import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main function to run the RandomForest feature importance analysis."""
    # 1. 데이터 로드 및 전처리 (이전 스크립트와 동일)
    print('📂 데이터 로딩 및 전처리 중...')
    sales_df = pd.read_csv('data/processed_data.csv', encoding='utf-8-sig')
    people_df = pd.read_csv('data/population/생활인구_전처리완료.csv', encoding='utf-8-sig')

    sales_df['ADSTRD_CD'] = sales_df['ADSTRD_CD'].astype(str)
    people_df['행정동코드'] = people_df['행정동코드'].astype(str)
    people_df_renamed = people_df.rename(columns={
        '행정동코드': 'ADSTRD_CD', '연도': 'YEAR', '분기': 'QUARTER'
    })

    sales_agg = sales_df.groupby(['ADSTRD_CD', 'ADSTRD_NM', 'YEAR', 'QUARTER']).agg({
        'THSMON_SELNG_AMT': 'sum', 'MDWK_SELNG_AMT': 'sum', 'WKEND_SELNG_AMT': 'sum',
        'TMZON_00_06_SELNG_AMT': 'sum', 'TMZON_06_11_SELNG_AMT': 'sum', 'TMZON_11_14_SELNG_AMT': 'sum',
        'TMZON_14_17_SELNG_AMT': 'sum', 'TMZON_17_21_SELNG_AMT': 'sum', 'TMZON_21_24_SELNG_AMT': 'sum',
        'ML_SELNG_AMT': 'sum', 'FML_SELNG_AMT': 'sum', 'AGRDE_20_SELNG_AMT': 'sum', 'AGRDE_30_SELNG_AMT': 'sum',
        'AGRDE_40_SELNG_AMT': 'sum', 'AGRDE_50_SELNG_AMT': 'sum', 'AGRDE_60_ABOVE_SELNG_AMT': 'sum'
    }).reset_index()

    merged_df = sales_agg.merge(people_df_renamed, on=['ADSTRD_CD', 'YEAR', 'QUARTER'], how='inner')
    print(f'✅ 데이터 준비 완료: {len(merged_df):,}개 행')

    # 2. 활력 지수 구성요소 계산 (이전 스크립트와 동일)
    print('\n🎯 4대 활력 지표 점수 계산 중...')
    vitality_df = merged_df.copy()
    epsilon = 1e-6
    
    q_sales_std = vitality_df.groupby('ADSTRD_CD')['THSMON_SELNG_AMT'].transform('std')
    q_sales_mean = vitality_df.groupby('ADSTRD_CD')['THSMON_SELNG_AMT'].transform('mean')
    vitality_df['매출_안정성'] = 1 - (q_sales_std / (q_sales_mean + epsilon)).clip(0, 1)
    vitality_df['주말_비율'] = vitality_df['WKEND_SELNG_AMT'] / (vitality_df['THSMON_SELNG_AMT'] + epsilon)
    vitality_df['주말_균형도'] = 1 - abs(vitality_df['주말_비율'] - 0.3)
    vitality_df['매출역동성_점수'] = (vitality_df['매출_안정성'] * 20 + vitality_df['주말_균형도'] * 15).clip(0, 35)

    vitality_df['1인당_총매출'] = vitality_df['THSMON_SELNG_AMT'] / (vitality_df['총생활인구_평균'] + epsilon)
    per_capita_90th = vitality_df['1인당_총매출'].quantile(0.90)
    vitality_df['소비효율성_점수'] = (vitality_df['1인당_총매출'] / (per_capita_90th + epsilon) * 25).clip(0, 25)

    vitality_df['야간매출'] = vitality_df['TMZON_21_24_SELNG_AMT'] + vitality_df['TMZON_00_06_SELNG_AMT']
    vitality_df['야간매출_비율'] = vitality_df['야간매출'] / (vitality_df['THSMON_SELNG_AMT'] + epsilon)
    time_cols = ['TMZON_00_06_SELNG_AMT', 'TMZON_06_11_SELNG_AMT', 'TMZON_11_14_SELNG_AMT', 'TMZON_14_17_SELNG_AMT', 'TMZON_17_21_SELNG_AMT', 'TMZON_21_24_SELNG_AMT']
    time_mean = vitality_df[time_cols].mean(axis=1)
    vitality_df['시간대_균등도'] = 1 - (vitality_df[time_cols].std(axis=1) / (time_mean + epsilon)).clip(0, 1)
    vitality_df['시간회복력_점수'] = ((vitality_df['야간매출_비율'] * 100) * 0.15 + vitality_df['시간대_균등도'] * 10).clip(0, 25)

    age_cols = ['AGRDE_20_SELNG_AMT', 'AGRDE_30_SELNG_AMT', 'AGRDE_40_SELNG_AMT', 'AGRDE_50_SELNG_AMT', 'AGRDE_60_ABOVE_SELNG_AMT']
    age_mean = vitality_df[age_cols].mean(axis=1)
    vitality_df['연령_균등도'] = 1 - (vitality_df[age_cols].std(axis=1) / (age_mean + epsilon)).clip(0, 1)
    total_gender_sales = vitality_df['ML_SELNG_AMT'] + vitality_df['FML_SELNG_AMT']
    vitality_df['여성매출_비율'] = vitality_df['FML_SELNG_AMT'] / (total_gender_sales + epsilon)
    vitality_df['성별_균형도'] = 1 - abs(vitality_df['여성매출_비율'] - 0.5) * 2
    vitality_df['경제다양성_점수'] = (vitality_df['연령_균등도'] * 8 + vitality_df['성별_균형도'] * 7).clip(0, 15)
    print('✅ 4대 지표 점수 계산 완료')
    score_cols = ['매출역동성_점수', '소비효율성_점수', '시간회복력_점수', '경제다양성_점수']

    # 3. 목표 변수(Y) 생성: 다음 분기 매출액
    print('\n🎯 목표 변수(다음 분기 매출액) 생성 중...')
    # 행정동코드로 정렬해야 shift 연산이 올바르게 적용됨
    vitality_df.sort_values(['ADSTRD_CD', 'YEAR', 'QUARTER'], inplace=True)
    vitality_df['target_sales'] = vitality_df.groupby('ADSTRD_CD')['THSMON_SELNG_AMT'].shift(-1)
    
    # 최종 데이터셋 (NaN 값 제거)
    final_df = vitality_df.dropna(subset=['target_sales'] + score_cols)
    print('✅ 목표 변수 생성 완료')

    # 4. 모델 학습을 위한 데이터 준비
    features = ['매출역동성_점수', '소비효율성_점수', '시간회복력_점수', '경제다양성_점수']
    X = final_df[features]
    y = final_df['target_sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'\n🤖 모델 학습 데이터 준비 완료: {len(X_train):,}개 행으로 학습 시작')

    # 5. RandomForest 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print('✅ RandomForest 모델 학습 완료')
    
    # 모델 성능 평가 (R-squared)
    r2_score = model.score(X_test, y_test)
    print(f'- 모델 예측 정확도 (R-squared): {r2_score:.4f}')

    # 6. 특성 중요도 추출 및 가중치 계산
    rf_importances = model.feature_importances_
    rf_weights = (rf_importances / np.sum(rf_importances)) * 100 # 100점 만점으로 변환

    print('\n\n📊 가중치 비교 (RandomForest 기반)')
    print('='*60)
    original_weights = {'매출역동성': 35, '소비효율성': 25, '시간회복력': 25, '경제다양성': 15}
    print(f'{'지표':<15} | {'기존 가중치':^20} | {'RandomForest 가중치':^20}')
    print('-'*60)
    for i, feature in enumerate(features):
        feature_name = feature.replace('_점수', '')
        print(f'{feature_name:<15} | {original_weights[feature_name]:^20.1f} | {rf_weights[i]:^20.1f}')
    print('='*60)

if __name__ == '__main__':
    main()

