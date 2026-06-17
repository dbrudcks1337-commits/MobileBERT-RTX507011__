import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import os


def create_labeled_data():
    print("📥 VADER 감성 분석기 다운로드 중...")
    nltk.download('vader_lexicon', quiet=True)
    analyzer = SentimentIntensityAnalyzer()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "rtx5070.csv")

    print(f"📂 데이터 불러오는 중...")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    df = df.dropna(subset=['comment'])
    comments = df['comment'].tolist()

    print("🤖 긍정/부정 자동 라벨링 시작...")
    sentiments = []

    for text in tqdm(comments):
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            sentiments.append(1)   # 긍정
        else:
            sentiments.append(0)   # 부정

    labeled_df = pd.DataFrame({
        'Text': comments,
        'Sentiment': sentiments
    })

    output_path = os.path.join(current_dir, "rtx5070_labeled.csv")
    labeled_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    pos = sentiments.count(1)
    neg = sentiments.count(0)
    total = len(sentiments)

    print(f"\n📊 --- 라벨링 완료 ---")
    print(f"총 댓글 수: {total}개")
    print(f"👍 긍정(1): {pos}개 ({pos/total*100:.1f}%)")
    print(f"👎 부정(0): {neg}개 ({neg/total*100:.1f}%)")
    print(f"✅ 저장 완료: {output_path}")


if __name__ == "__main__":
    create_labeled_data()
