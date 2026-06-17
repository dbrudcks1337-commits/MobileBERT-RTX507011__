import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
import os


def analyze():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 분석 대상 데이터 로드 (전체)
    print("📂 데이터 로드 중...")
    df = pd.read_csv(os.path.join(current_dir, "rtx5070_labeled.csv"), encoding="utf-8-sig")
    df = df.dropna(subset=['Text'])
    comments = df['Text'].tolist()
    print(f"✅ {len(comments)}개 댓글 로드 완료")

    # 2. 모델 로드
    print("🤖 모델 로드 중...")
    model_path = os.path.join(current_dir, "mobilebert_final.tp")
    tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
    model = MobileBertForSequenceClassification.from_pretrained(model_path, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 사용 장치: {device}")
    model.to(device)
    model.eval()

    # 3. 예측
    print("📝 토큰화 중...")
    encodings = tokenizer(comments, truncation=True, padding=True, max_length=128, return_tensors="pt")
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)

    print("🔥 감성 분석 시작...")
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            predictions.extend(preds)

    # 4. 결과 저장
    df['sentiment'] = predictions
    df['sentiment_label'] = df['sentiment'].map({0: 'Negative(부정)', 1: 'Positive(긍정)'})
    output_path = os.path.join(current_dir, "rtx5070_sentiment_result.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    # 5. 통계
    total = len(df)
    pos = (df['sentiment'] == 1).sum()
    neg = (df['sentiment'] == 0).sum()

    print(f"\n📊 --- 분석 결과 ---")
    print(f"총 댓글 수: {total}개")
    print(f"👍 긍정(1): {pos}개 ({pos/total*100:.1f}%)")
    print(f"👎 부정(0): {neg}개 ({neg/total*100:.1f}%)")

    # 6. 차트 생성
    print("\n📈 차트 생성 중...")
    colors = ['#F44336', '#4CAF50']
    labels = ['Negative', 'Positive']
    counts = [neg, pos]

    # 차트 1: 파이 + 막대
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('RTX 5070 YouTube Comment Sentiment Analysis\n(MobileBERT Fine-tuned)', fontsize=13, fontweight='bold')

    axes[0].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    axes[0].set_title(f'Sentiment Distribution\n(Total: {total:,} comments)', fontsize=11)

    bars = axes[1].bar(labels, counts, color=colors, width=0.45)
    axes[1].set_title('Comment Count by Sentiment', fontsize=11)
    axes[1].set_ylabel('Number of Comments')
    axes[1].set_ylim(0, max(counts) * 1.15)
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f'{count:,}\n({count/total*100:.1f}%)',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "chart1_sentiment.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  차트 1 저장 완료")

    # 차트 2: 학습 데이터 라벨 분포
    train_df = pd.read_csv(os.path.join(current_dir, "rtx5070_train.csv"), encoding="utf-8-sig")
    label_counts = train_df['Sentiment'].value_counts().sort_index()
    fig2, ax = plt.subplots(figsize=(7, 4))
    bars2 = ax.bar(['Negative (0)', 'Positive (1)'],
                   [label_counts.get(0, 0), label_counts.get(1, 0)],
                   color=colors, width=0.4)
    ax.set_title('Label Distribution (Training Data)', fontsize=11)
    ax.set_ylabel('Count')
    ax.set_ylim(0, max(label_counts.values) * 1.15)
    for bar, count in zip(bars2, [label_counts.get(0, 0), label_counts.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "chart2_label.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  차트 2 저장 완료")

    # 차트 3: 댓글 길이 분포
    df['word_count'] = df['Text'].apply(lambda x: len(str(x).split()))
    pos_len = df[df['sentiment'] == 1]['word_count']
    neg_len = df[df['sentiment'] == 0]['word_count']

    fig3, ax = plt.subplots(figsize=(9, 4))
    ax.hist(pos_len.clip(upper=100), bins=40, alpha=0.6, color='#4CAF50', label=f'Positive (n={pos:,})', range=(0, 100))
    ax.hist(neg_len.clip(upper=100), bins=40, alpha=0.6, color='#F44336', label=f'Negative (n={neg:,})', range=(0, 100))
    ax.set_title('Comment Length Distribution by Sentiment (words, capped at 100)', fontsize=11)
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "chart3_length.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  차트 3 저장 완료")

    print(f"\n🎉 완료!")
    print(f"  결과 CSV:  rtx5070_sentiment_result.csv")
    print(f"  차트 1:    chart1_sentiment.png")
    print(f"  차트 2:    chart2_label.png")
    print(f"  차트 3:    chart3_length.png")


if __name__ == "__main__":
    analyze()
