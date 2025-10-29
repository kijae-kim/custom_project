
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main function to run the PCA analysis for city vitality index."""
    # 1. 데이터 로드 및 전처리
    print('📂 데이터 로딩 및 전처리 중...')
    sales_df = pd.read_csv('data/processed_data.csv', encoding='utf-8-sig')
    people_df = pd.read_csv('data/population/생활인구_전처리완료.csv', encoding='utf-8-sig')

    sales_df['ADSTRD_CD'] = sales_df['ADSTRD_CD'].astype(str)
    people_df['행정동코드'] = people_df['행정동코드'].astype(str)
    people_df_renamed = people_df.rename(columns={
        '행정동코드': 'ADSTRD_CD',
        '연도': 'YEAR',
        '분기': 'QUARTER'
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

    # 2. 활력 지수 구성요소 계산
    print('\n🎯 4대 활력 지표 점수 계산 중...')
    vitality_df = merged_df.copy()
    epsilon = 1e-6

    # 매출 역동성
    q_sales_std = vitality_df.groupby('ADSTRD_CD')['THSMON_SELNG_AMT'].transform('std')
    q_sales_mean = vitality_df.groupby('ADSTRD_CD')['THSMON_SELNG_AMT'].transform('mean')
    vitality_df['매출_안정성'] = 1 - (q_sales_std / (q_sales_mean + epsilon)).clip(0, 1)
    vitality_df['주말_비율'] = vitality_df['WKEND_SELNG_AMT'] / (vitality_df['THSMON_SELNG_AMT'] + epsilon)
    vitality_df['주말_균형도'] = 1 - abs(vitality_df['주말_비율'] - 0.3)
    vitality_df['매출역동성_점수'] = (vitality_df['매출_안정성'] * 20 + vitality_df['주말_균형도'] * 15).clip(0, 35)

    # 소비 효율성
    vitality_df['1인당_총매출'] = vitality_df['THSMON_SELNG_AMT'] / (vitality_df['총생활인구_평균'] + epsilon)
    per_capita_90th = vitality_df['1인당_총매출'].quantile(0.90)
    vitality_df['소비효율성_점수'] = (vitality_df['1인당_총매출'] / (per_capita_90th + epsilon) * 25).clip(0, 25)

    # 시간 회복력
    vitality_df['야간매출'] = vitality_df['TMZON_21_24_SELNG_AMT'] + vitality_df['TMZON_00_06_SELNG_AMT']
    vitality_df['야간매출_비율'] = vitality_df['야간매출'] / (vitality_df['THSMON_SELNG_AMT'] + epsilon)
    time_cols = ['TMZON_00_06_SELNG_AMT', 'TMZON_06_11_SELNG_AMT', 'TMZON_11_14_SELNG_AMT', 'TMZON_14_17_SELNG_AMT', 'TMZON_17_21_SELNG_AMT', 'TMZON_21_24_SELNG_AMT']
    time_mean = vitality_df[time_cols].mean(axis=1)
    vitality_df['시간대_균등도'] = 1 - (vitality_df[time_cols].std(axis=1) / (time_mean + epsilon)).clip(0, 1)
    vitality_df['시간회복력_점수'] = ((vitality_df['야간매출_비율'] * 100) * 0.15 + vitality_df['시간대_균등도'] * 10).clip(0, 25)

    # 경제 다양성
    age_cols = ['AGRDE_20_SELNG_AMT', 'AGRDE_30_SELNG_AMT', 'AGRDE_40_SELNG_AMT', 'AGRDE_50_SELNG_AMT', 'AGRDE_60_ABOVE_SELNG_AMT']
    age_mean = vitality_df[age_cols].mean(axis=1)
    vitality_df['연령_균등도'] = 1 - (vitality_df[age_cols].std(axis=1) / (age_mean + epsilon)).clip(0, 1)
    total_gender_sales = vitality_df['ML_SELNG_AMT'] + vitality_df['FML_SELNG_AMT']
    vitality_df['여성매출_비율'] = vitality_df['FML_SELNG_AMT'] / (total_gender_sales + epsilon)
    vitality_df['성별_균형도'] = 1 - abs(vitality_df['여성매출_비율'] - 0.5) * 2
    vitality_df['경제다양성_점수'] = (vitality_df['연령_균등도'] * 8 + vitality_df['성별_균형도'] * 7).clip(0, 15)

    score_cols = ['매출역동성_점수', '소비효율성_점수', '시간회복력_점수', '경제다양성_점수']
    vitality_df.dropna(subset=score_cols, inplace=True)
    print('✅ 4대 지표 점수 계산 완료')

    # 3. PCA를 이용한 가중치 산출
    print('\n🔬 PCA 분석 시작...')
    features = ['매출역동성_점수', '소비효율성_점수', '시간회복력_점수', '경제다양성_점수']
    X = vitality_df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print('✅ 데이터 정규화 완료')

    pca = PCA(n_components=4)
    pca.fit(X_scaled)
    print('✅ PCA 모델 학습 완료')

    explained_variance_ratio = pca.explained_variance_ratio_
    print(f'- 첫 번째 주성분(PC1)의 설명력: {explained_variance_ratio[0]*100:.2f}%')

    pc1_loadings = pca.components_[0]
    abs_loadings = np.abs(pc1_loadings)
    pca_weights = (abs_loadings / np.sum(abs_loadings)) * 100

    print('\n📊 가중치 비교')
    print('='*50)
    original_weights = {'매출역동성': 35, '소비효율성': 25, '시간회복력': 25, '경제다양성': 15}
    print(f'{'지표':<10} | {'기존 가중치':^15} | {'PCA 가중치':^15}')
    print('-'*50)
    for i, feature in enumerate(features):
        feature_name = feature.replace('_점수', '')
        print(f'{feature_name:<10} | {original_weights[feature_name]:^15.1f} | {pca_weights[i]:^15.1f}')
    print('='*50)

    # 4. PCA 기반 신규 활력 지수 계산 및 비교
    print('\n\n📊 기존 지수 vs PCA 지수 통계 비교')
    vitality_df['경제활력지수_기존'] = (vitality_df['매출역동성_점수'] * 0.35 + 
                                    vitality_df['소비효율성_점수'] * 0.25 + 
                                    vitality_df['시간회복력_점수'] * 0.25 + 
                                    vitality_df['경제다양성_점수'] * 0.15)

    weights_norm = pca_weights / 100
    vitality_df['경제활력지수_PCA'] = (vitality_df['매출역동성_점수'] * weights_norm[0] + 
                                     vitality_df['소비효율성_점수'] * weights_norm[1] + 
                                     vitality_df['시간회복력_점수'] * weights_norm[2] + 
                                     vitality_df['경제다양성_점수'] * weights_norm[3])

    print(vitality_df[['경제활력지수_기존', '경제활력지수_PCA']].describe())

if __name__ == '__main__':
    main()
