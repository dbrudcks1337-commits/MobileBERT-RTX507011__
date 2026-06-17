import os
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT


def get_all_comments_no_filter(video_data, target_total_comments=30000):
    downloader = YoutubeCommentDownloader()
    comment_list = []

    print(
        f"🚀 [필터 해제 전체 수집] 목표 데이터 {target_total_comments}건 수집을 시작합니다!"
    )

    for idx, video in enumerate(video_data, 1):
        if len(comment_list) >= target_total_comments:
            break

        url = video["url"]
        print(
            f"🔄 [{idx}/{len(video_data)}] 영상 모든 댓글 수집 중... (현재 누적 데이터: {len(comment_list)}개)"
        )
        try:
            comments = downloader.get_comments_from_url(
                url, sort_by=SORT_BY_RECENT
            )
            video_count = 0

            for comment in comments:
                # 💡 국물도 안 버리고 어떤 필터링도 없이 무조건 다 담습니다!
                comment_list.append(
                    {
                        "comment": comment["text"],
                        "likes": comment["votes"],
                        "date": comment["time"],
                    }
                )
                video_count += 1

                if len(comment_list) % 1000 == 0 and len(comment_list) > 0:
                    print(f"   📥 누적 데이터 총 {len(comment_list)}개 확보 완료...")

                if len(comment_list) >= target_total_comments:
                    break

            print(f"   ✅ 이 영상에서 {video_count}개 수집 완료!\n")
        except Exception as e:
            print(f"   ❌ 에러 발생으로 패스: {e}")
            continue

    # 💡 analyze.py 파일과 바로 연동되도록 mobilebert_imdb.tp 폴더 안에 자동 저장되도록 설정
    df = pd.DataFrame(comment_list)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "mobilebert_imdb.tp/mobilebert_imdb.tp")

    # 혹시 폴더가 없으면 만들어주는 안전장치
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    filename = os.path.join(target_dir, "../rtx5070.csv")
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(
        f"\n🎉 [최종 완료] 필터 없이 총 {len(df)}개 댓글 저장 완료! 파일 경로: '{filename}'"
    )


# 🔴 수집 타겟 영상 리스트 (필터를 안 쓰므로 pure 속성은 제거함)
VIDEO_DATA = [
    {"url": "https://www.youtube.com/watch?v=rUaztYdgoj0"},  # Linus Tech Tips
    {"url": "https://www.youtube.com/watch?v=ntSylZ1Bp1Y"},  # Gamers Nexus
    {"url": "https://www.youtube.com/watch?v=qPGDVh_cQb0"},  # Hardware Unboxed
    {"url": "https://www.youtube.com/watch?v=BX4y9t-BjXg"},  # 실사용 종합 후기
    {"url": "https://www.youtube.com/watch?v=PXRu0d8fznI"},  # 30종 게임 종합 리뷰
    {"url": "https://www.youtube.com/watch?v=4PKzsj4OLmc"},  #
    {"url": "https://www.youtube.com/watch?v=g7yNt5PDCLM"},  # vs
    {"url": "https://www.youtube.com/watch?v=ZMC0ySge530"},  # vs 5060Ti (13개 게임 벤치)
    {"url": "https://www.youtube.com/watch?v=YOjvmqENeKk"},  # vs
    {"url": "https://www.youtube.com/watch?v=q3fnU0eK0So"},  # vs
    {"url": "https://www.youtube.com/watch?v=bTA5EWEhjUs"},  # vs
    {"url": "https://www.youtube.com/watch?v=jp-_WP1PFgw"},  # vs
    {"url": "https://www.youtube.com/watch?v=-GvLt7DIhiU"},
    {"url": "https://www.youtube.com/watch?v=D6anpTqpyDM"},
    {"url": "https://www.youtube.com/watch?v=IktcKqvUhKA"},
    {"url": "https://www.youtube.com/watch?v=GWJX4TAdPZA&t=1370s"},
    {"url": "https://www.youtube.com/watch?v=7UYILx7jv9s"},
    {"url": "https://www.youtube.com/watch?v=Z9xGUuXUcQg"},
    {"url": "https://www.youtube.com/watch?v=PvYG3dlwz9w"},
    {"url": "https://www.youtube.com/watch?v=TaCorPpKLcg&t=140s"},
    {"url": "https://www.youtube.com/watch?v=6DcYu3ImTgk"}, # 9070과 비교
    {"url": "https://www.youtube.com/watch?v=YOoydMZoBWs"} # 가격이 비슷한 9070xt와 비교
]

# 실행
get_all_comments_no_filter(VIDEO_DATA, target_total_comments=30000)