import pandas as pd
import re

# CSV 파일 읽어오기
input_csv_path = 'union_data_before.csv'
output_csv_path = 'union_data.csv'

df = pd.read_csv(input_csv_path)

# 전처리 함수 정의
def preprocess_text(text):
    # 정규 표현식 패턴 정의: 한글과 일반적인 특수 문자 이외에 모든 문자 제거
    pattern = r'[^\wㄱ-ㅎㅏ-ㅣ가-힣@#$%^&*()[\]{}\-=_;:,.<>/\"\'~+]+'
    
    # 패턴과 일치하는 문자 제거하고 공백으로 대체
    cleaned_text = re.sub(pattern, ' ', text)
    
    return cleaned_text

# 'union_introduction' 열에 대해 전처리 적용
df['union_introduction'] = df['union_introduction'].apply(preprocess_text)

# 전처리가 완료된 DataFrame을 새로운 CSV 파일로 저장
df.to_csv(output_csv_path, index=False)