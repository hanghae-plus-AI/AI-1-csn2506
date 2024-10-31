import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import chardet

# Load GPT key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("환경 변수 'OPENAI_API_KEY'가 설정되지 않았습니다. .env 파일에 올바른 API 키를 추가하세요.")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

def extract_date_from_query(query):
    """자연어 쿼리에서 날짜 추출"""
    prompt = f"""
다음 쿼리에서 언급된 날짜를 'MM.DD' 형식으로 변환하세요. 오늘은 {datetime.now().strftime('%Y년 %m월 %d일')}입니다.
현재 날짜를 기준으로 상대적인 날짜를 계산하세요.

쿼리: {query}

출력 형식:
MM.DD

예시 입력/출력:
- 내일 일정 알려줘 → {(datetime.now() + timedelta(days=1)).strftime('%m.%d')}
- 다음주 월요일 일정 확인해줘 → 다음 월요일의 날짜를 MM.DD 형식으로
- 오늘 일정 → {datetime.now().strftime('%m.%d')}
- 3일 뒤 일정 → {(datetime.now() + timedelta(days=3)).strftime('%m.%d')}

날짜만 출력하세요.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        target_date = response.choices[0].message.content.strip()
        return target_date
    except Exception as e:
        st.error(f"날짜 추출 중 오류 발생: {str(e)}")
        return datetime.now().strftime("%m.%d")

# Streamlit UI
def main():
    st.title("이메일 일정 분석기")
    st.write("일정을 분석하고 추출하기 위해 JSON 형식의 이메일 데이터를 업로드하세요.")

    uploaded_file = st.file_uploader("JSON 파일을 선택하세요", type=["json"])
    
    # 자연어 쿼리 입력
    query = st.text_input(
        "확인하고 싶은 일정을 자연어로 입력하세요",
        placeholder="예: 내일 일정 확인해줘, 다음주 월요일 일정 알려줘",
        key="query"
    )
    
    if uploaded_file is not None and query:
        if uploaded_file.name.endswith(".json"):
            email_data = parse_json_file(uploaded_file)
        else:
            st.write("지원하지 않는 파일 형식입니다.")
            return

        # 분석 버튼 추가
        if st.button("일정 분석하기"):
            # 쿼리에서 날짜 추출
            target_date = extract_date_from_query(query)
            
            with st.spinner(f"일정을 분석하고 있습니다..."):
                summaries = analyze_emails(email_data, target_date)
                schedule = get_schedule(summaries, target_date)

                if schedule:
                    st.write(f"요청하신 날짜 ({target_date})의 일정은 다음과 같습니다:")
                    for item in schedule:
                        st.write(item)
                else:
                    st.write(f"요청하신 날짜 ({target_date})의 일정이 없습니다.")

# Function to parse JSON file
def parse_json_file(json_file):
    raw_data = json_file.read()
    encoding = chardet.detect(raw_data)['encoding']
    email_data = json.loads(raw_data.decode(encoding, errors='replace'))
    return email_data

# Function to summarize email and extract schedule details
def analyze_emails(email_data, target_date):
    summaries = []
    prompt = (
        f" json형식으로 된 email에서 content와 날짜가 일치 하는 content를 요약하고 {target_date} 날짜에 맞는 일정 세부 정보를 추출하세요."
        f" email: {email_data}\n"
        f"\n예시 출력:\n"
        f"{target_date}의 일정은 다음과 같습니다.\n"
        f"ㅇㅇㅇ 미팅\n장소: 눈꽃 한우\n시간: 오후 1시\n내용: 회의 후 식사"
        f"\n중복되는 말은 제거하고 같은 날짜는 하나로 보여주고 내용은 50자 이내로 요약해주세요.\n"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    summaries.append(response.choices[0].message.content.strip())
    return summaries

# Function to get schedule from analyzed summaries
def get_schedule(summaries, target_date):
    schedule = []
    for summary in summaries:
        if target_date in summary:
            schedule.append(summary)
    return schedule

if __name__ == "__main__":
    main()