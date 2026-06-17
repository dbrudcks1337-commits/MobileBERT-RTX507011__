"""
원본 데이터 EDA 차트 생성
- 수집한 원본 데이터(rtx5070.csv) 기반
- chart0_eda_length.png : 댓글 길이 분포
- chart0_eda_vader.png  : VADER 점수 분포
"""

import pandas as pd
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import os


def eda():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 원본 데이터 로드
    print("📂 원본 데이터 로드 중...")
    df = pd.read_csv(os.path.join(current_dir, "rtx5070.csv"), encoding="utf-8-sig")
    df = df.dropna(subset=['comment'])
    print(f"✅ 총 {len(df)}개 댓글 로드 완료")

    # ── 차트 1: 댓글 길이 분포 ──────────────────────────────
    df['word_count'] = df['comment'].apply(lambda x: len(str(x).split()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Original Data EDA - Comment Length Distribution', fontsize=13, fontweight='bold')

    # 전체 분포 (100단어 이하)
    axes[0].hist(df['word_count'].clip(upper=100), bins=40, color='#2196F3', alpha=0.8, range=(0, 100))
    axes[0].set_title('Word Count Distribution (capped at 100)', fontsize=11)
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(df['word_count'].median(), color='red', linestyle='--', label=f"Median: {df['word_count'].median():.0f}")
    axes[0].axvline(df['word_count'].mean(), color='orange', linestyle='--', label=f"Mean: {df['word_count'].mean():.1f}")
    axes[0].legend()

    # 5단어 미만 제거 기준 표시
    axes[1].hist(df['word_count'].clip(upper=30), bins=30, color='#2196F3', alpha=0.8, range=(0, 30))
    axes[1].axvline(5, color='red', linestyle='--', linewidth=2, label='Removal Threshold (< 5 words)')
    axes[1].set_title('Word Count Distribution (capped at 30)', fontsize=11)
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    p1 = os.path.join(current_dir, "chart0_eda_length.png")
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 저장: chart0_eda_length.png")

    # 기초 통계
    print(f"\n📊 댓글 길이 기초 통계")
    print(f"  평균: {df['word_count'].mean():.1f}단어")
    print(f"  중앙값: {df['word_count'].median():.0f}단어")
    print(f"  최대: {df['word_count'].max()}단어")
    print(f"  5단어 미만: {(df['word_count'] < 5).sum()}개 ({(df['word_count'] < 5).sum()/len(df)*100:.1f}%)")

    # ── 차트 2: VADER 점수 분포 ──────────────────────────────
    print("\n🤖 VADER 점수 계산 중...")
    nltk.download('vader_lexicon', quiet=True)
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(str(t))['compound'] for t in tqdm(df['comment'].tolist())]
    df['vader_score'] = scores

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle('Original Data EDA - VADER Sentiment Score Distribution', fontsize=13, fontweight='bold')

    # 전체 점수 분포
    axes2[0].hist(df['vader_score'], bins=50, color='#9C27B0', alpha=0.8)
    axes2[0].axvline(0.05, color='red', linestyle='--', linewidth=2, label='Labeling Threshold (0.05)')
    axes2[0].set_title('VADER Compound Score Distribution', fontsize=11)
    axes2[0].set_xlabel('Compound Score')
    axes2[0].set_ylabel('Frequency')
    axes2[0].legend()

    # 긍정/부정 비율 파이차트
    pos = (df['vader_score'] >= 0.05).sum()
    neg = (df['vader_score'] < 0.05).sum()
    axes2[1].pie([neg, pos], labels=['Negative', 'Positive'],
                 colors=['#F44336', '#4CAF50'], autopct='%1.1f%%',
                 startangle=90, textprops={'fontsize': 12})
    axes2[1].set_title(f'Sentiment Ratio (Total: {len(df):,})', fontsize=11)

    plt.tight_layout()
    p2 = os.path.join(current_dir, "chart0_eda_vader.png")
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 저장: chart0_eda_vader.png")

    print(f"\n📊 VADER 점수 기초 통계")
    print(f"  긍정(≥0.05): {pos}개 ({pos/len(df)*100:.1f}%)")
    print(f"  부정(<0.05): {neg}개 ({neg/len(df)*100:.1f}%)")
    print(f"\n🎉 EDA 차트 생성 완료!")
    print(f"  chart0_eda_length.png")
    print(f"  chart0_eda_vader.png")


if __name__ == "__main__":
    eda()