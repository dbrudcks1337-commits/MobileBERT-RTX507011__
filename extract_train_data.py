import pandas as pd
import os


def extract_train_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 원본 라벨링 데이터 로드
    print("📂 라벨링 데이터 로드 중...")
    input_path = os.path.join(current_dir, "rtx5070_labeled.csv")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = df.dropna(subset=['Text'])
    print(f"✅ 원본 데이터: {len(df)}개")
    print(f"   긍정(1): {(df['Sentiment']==1).sum()}개")
    print(f"   부정(0): {(df['Sentiment']==0).sum()}개")

    # 2. 전처리 1 - 문장 길이 제한 (5단어 미만 제거)
    df['word_count'] = df['Text'].apply(lambda x: len(str(x).split()))
    df = df[df['word_count'] >= 5].reset_index(drop=True)
    df = df.drop(columns=['word_count'])
    print(f"\n문장 길이 제한 후 (5단어 이상): {len(df)}개")
    print(f"→ 이 데이터가 '분석 대상 데이터'입니다.")

    SAMPLE_PER_CLASS = 2000
    pos_sample = df[df['Sentiment'] == 1].sample(n=SAMPLE_PER_CLASS, random_state=2026).copy()
    neg_sample = df[df['Sentiment'] == 0].sample(n=SAMPLE_PER_CLASS, random_state=2026).copy()

    train_df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=2026).reset_index(drop=True)

    # 4. 저장
    output_path = os.path.join(current_dir, "rtx5070_train.csv")
    train_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n📊 --- 학습 데이터 추출 결과 ---")
    print(f"추출 기준: 5단어 미만 제거 후 긍정/부정 각 {SAMPLE_PER_CLASS}개 균등 추출")
    print(f"긍정(1): {SAMPLE_PER_CLASS}개")
    print(f"부정(0): {SAMPLE_PER_CLASS}개")
    print(f"총 학습 데이터: {len(train_df)}개")
    print(f"✅ 저장 완료: {output_path}")


if __name__ == "__main__":
    extract_train_data()
