from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 ตั้งค่าเชื่อมต่อ Google Sheets
GOOGLE_SHEET_NAME = "Data_project_like_course_branch"  # ✨ เปลี่ยนเป็นชื่อไฟล์ Google Sheet ของคุณ
CREDENTIALS_FILE = "service_account.json"  # ✨ เปลี่ยนเป็นชื่อไฟล์ JSON ของ Service Account

def connect_google_sheets():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        sheet = client.open(GOOGLE_SHEET_NAME).sheet1
        return sheet
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Sheets connection error: {str(e)}")

# โหลดข้อมูล
def load_data():
    df = pd.read_excel('Respon.xlsx')
    Weight = pd.read_excel('Weight.xlsx')
    branch_data = pd.read_excel('BranchID.Name.xlsx')

    df = df.drop(['Timestamp', 'User'], axis=1)

    label_encoder = LabelEncoder()
    df['Course'] = label_encoder.fit_transform(df['Course'])

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = LabelEncoder().fit_transform(df[column])

    score_data = df.drop(['Course', 'Branch'], axis=1)

    return df, Weight, branch_data, score_data, label_encoder

def process_personality_answers(personality_answers: Dict[str, str]) -> list:
    processed = {int(k.replace('answers[', '').replace(']', '')): int(v) for k, v in personality_answers.items()}
    return [processed[k] for k in sorted(processed.keys())]

def process_subject_scores(scores: Dict[str, str]) -> list:
    processed = {int(k.replace('scores[', '').replace(']', '')): float(v) for k, v in scores.items()}
    return [processed[k] for k in sorted(processed.keys())]

def get_recommended_courses(personality_values: list, score_data, label_encoder, df) -> List[str]:
    course_similarity = cosine_similarity([personality_values], score_data)[0]
    top_courses_indices = np.argsort(course_similarity)[-5:][::-1]
    return list(label_encoder.inverse_transform(df.iloc[top_courses_indices]['Course']))

def get_recommended_branches(courses: List[str], subject_scores: list, branch_data, Weight) -> List[str]:
    recommended_branches = []
    all_branch_ids = branch_data[branch_data['Course'].isin(courses)]['BranchID'].values
    relevant_weights = Weight[Weight['BranchID'].isin(all_branch_ids)]

    if relevant_weights.empty:
        return []

    user_scores_array = np.array(subject_scores).reshape(1, -1)
    filtered_weight = relevant_weights.drop(columns=['BranchID']).fillna(0)
    cosine_similarities = cosine_similarity(user_scores_array, filtered_weight)[0]

    results = pd.DataFrame({
        'BranchID': relevant_weights['BranchID'].values,
        'Similarity': cosine_similarities
    })

    top_branches = results[results['Similarity'] > 0].nlargest(10, 'Similarity')

    for _, row in top_branches.iterrows():
        branch_info = branch_data.loc[branch_data['BranchID'] == row['BranchID']]
        if not branch_info.empty:
            branch_name = branch_info['Branch'].values[0]
            similarity = float(row['Similarity'] * 100)
            recommended_branches.append(f"{branch_name} (ค่า Similarity: {similarity:.2f}%)")

    return recommended_branches

@app.post("/api/recommend")
async def recommend(payload: Dict[str, Dict[str, str]]):
    try:
        df, Weight, branch_data, score_data, label_encoder = load_data()

        personality_values = process_personality_answers(payload['personality_answers'])
        subject_values = process_subject_scores(payload['scores'])

        recommended_courses = get_recommended_courses(personality_values, score_data, label_encoder, df)
        recommended_branches = get_recommended_branches(recommended_courses, subject_values, branch_data, Weight)

        return {
            "ผลการแนะนำ": {
                "คณะที่แนะนำตามบุคลิก": [{"name": course} for course in recommended_courses],
                "สาขาที่แนะนำตามน้ำหนักคะแนนและบุคลิก": recommended_branches or ["ไม่มีสาขาที่ตรงกับเกณฑ์ของคุณ"]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-liked-result")
async def save_liked_result(data: Dict):
    try:
        print("🔹 Data received:", data)  # ✅ Debug เช็คข้อมูลที่รับมา
        sheet = connect_google_sheets()

        new_data = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            *[v for v in data['personalityAnswers'].values()],
            *[v for v in data['scores'].values()],
            *[c['name'] for c in data['recommendations']['คณะที่แนะนำตามบุคลิก']],
            *data['recommendations']['สาขาที่แนะนำตามน้ำหนักคะแนนและบุคลิก']
        ]

        print("✅ Data to be saved:", new_data)  # ✅ Debug เช็คข้อมูลก่อนบันทึก
        sheet.append_row(new_data)

        return {"success": True, "message": "Data saved to Google Sheets successfully"}
    except Exception as e:
        print("🔥 ERROR:", str(e))  # ✅ Debug เช็ค Error ที่เกิดขึ้น
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)