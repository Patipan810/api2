import os
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://itbit0267.cpkkuhost.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ไฟล์ข้อมูล
GITHUB_FILES = {
    "Respon.xlsx": "https://raw.githubusercontent.com/USERNAME/REPO/BRANCH/Respon.xlsx",
    "Weight.xlsx": "https://raw.githubusercontent.com/USERNAME/REPO/BRANCH/Weight.xlsx",
    "BranchID.Name.xlsx": "https://raw.githubusercontent.com/USERNAME/REPO/BRANCH/BranchID.Name.xlsx"
}

# ✅ โหลดข้อมูล
def load_data():
    try:
        df = pd.read_excel("Respon.xlsx")
        weight = pd.read_excel("Weight.xlsx")
        branch_data = pd.read_excel("BranchID.Name.xlsx")

        if df.empty or weight.empty or branch_data.empty:
            raise HTTPException(status_code=500, detail="One or more data files are empty!")

        # 🔹 ใช้ LabelEncoder กับทุกคอลัมน์ที่เป็นข้อความ
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col].astype(str))

        df = df.drop(['Timestamp', 'User'], axis=1, errors='ignore')
        score_data = df.drop(['Course', 'Branch'], axis=1, errors='ignore')

        return df, weight, branch_data, score_data, label_encoders

    except Exception as e:
        logging.error(f"Error reading Excel files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading Excel files: {str(e)}")

# ✅ ประมวลผลข้อมูล
def process_personality_answers(personality_answers: dict) -> list:
    cleaned_answers = {k.replace('answers[', '').replace(']', ''): v for k, v in personality_answers.items()}
    return [int(cleaned_answers[key]) for key in sorted(cleaned_answers)]

def process_subject_scores(scores: dict) -> list:
    cleaned_scores = {k.replace('answers[', '').replace(']', ''): v for k, v in scores.items()}
    return [float(cleaned_scores[key]) for key in sorted(cleaned_scores)]

# ✅ แนะนำคณะ
def get_recommended_courses(personality_values, df, score_data, label_encoders):
    similarity_scores = cosine_similarity([personality_values], score_data)[0]
    top_courses_indices = np.argsort(similarity_scores)[-5:][::-1]

    return list(label_encoders['Course'].inverse_transform(df.iloc[top_courses_indices]['Course']))

# ✅ แนะนำสาขา
def get_recommended_branches(courses, subject_scores, branch_data, weight):
    relevant_branches = branch_data[branch_data['Course'].isin(courses)]['BranchID'].values
    if relevant_branches.size == 0:
        return []

    relevant_weights = weight[weight['BranchID'].isin(relevant_branches)]
    user_scores_array = np.array(subject_scores).reshape(1, -1)  # ✅ ไม่ตัดขนาด

    filtered_weight = relevant_weights.drop(columns=['BranchID'])

    cosine_similarities = cosine_similarity(user_scores_array, filtered_weight)[0]
    results = pd.DataFrame({'BranchID': relevant_weights['BranchID'].values, 'Similarity': cosine_similarities})

    top_branches = results.nlargest(10, 'Similarity')
    recommended_branches = []

    for _, row in top_branches.iterrows():
        branch_info = branch_data.loc[branch_data['BranchID'] == row['BranchID']]
        if not branch_info.empty:
            branch_name = branch_info['Branch'].values[0]
            similarity = float(row['Similarity'] * 100)
            recommended_branches.append(f"{branch_name} (ค่า Similarity: {similarity:.2f}%)")

    return recommended_branches

# ✅ API แนะนำ
@app.post("/api/recommend")
async def recommend(payload: dict):
    try:
        df, weight, branch_data, score_data, label_encoders = load_data()
        personality_values = process_personality_answers(payload['personality_answers'])
        subject_values = process_subject_scores(payload['scores'])

        recommended_courses = get_recommended_courses(personality_values, df, score_data, label_encoders)
        recommended_branches = get_recommended_branches(recommended_courses, subject_values, branch_data, weight)

        return {
            "ผลการแนะนำ": {
                "คณะที่แนะนำตามบุคลิก": [{"name": course} for course in recommended_courses],
                "สาขาที่แนะนำตามน้ำหนักคะแนนและบุคลิก": recommended_branches or ["ไม่มีสาขาที่ตรงกับเกณฑ์ของคุณ"]
            }
        }
    except Exception as e:
        logging.error(f"Error in /api/recommend: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
