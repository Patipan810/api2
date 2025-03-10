import os
import base64
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2.service_account import Credentials
import gspread
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# ✅ อนุญาต CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://itbit0267.cpkkuhost.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ตั้งค่า URL ไฟล์จาก GitHub
GITHUB_FILES = {
    "Respon.xlsx": "https://raw.githubusercontent.com/USERNAME/REPO/BRANCH/Respon.xlsx",
    "Weight.xlsx": "https://raw.githubusercontent.com/USERNAME/REPO/BRANCH/Weight.xlsx",
    "BranchID.Name.xlsx": "https://raw.githubusercontent.com/USERNAME/REPO/BRANCH/BranchID.Name.xlsx"
}

# ✅ ฟังก์ชันโหลดไฟล์จาก GitHub
def download_file(url: str, filename: str):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        logging.debug(f"✅ Downloaded {filename} successfully.")
    else:
        logging.error(f"❌ Failed to download {filename} from GitHub")
        raise HTTPException(status_code=500, detail=f"Failed to download {filename} from GitHub")

# ✅ โหลดข้อมูลจากไฟล์ .xlsx
def load_data():
    try:
        for filename, url in GITHUB_FILES.items():
            if not os.path.exists(filename):
                logging.debug(f"📂 Downloading {filename} from {url}")
                download_file(url, filename)
        
        df = pd.read_excel("Respon.xlsx")
        weight = pd.read_excel("Weight.xlsx")
        branch_data = pd.read_excel("BranchID.Name.xlsx")
        
        if df.empty or weight.empty or branch_data.empty:
            logging.error("❌ One or more dataframes are empty!")
            raise HTTPException(status_code=500, detail="One or more data files are empty!")
        
        df = df.drop(['Timestamp', 'User'], axis=1, errors='ignore')
        label_encoder = LabelEncoder()
        df['Course'] = label_encoder.fit_transform(df['Course'].astype(str))
        
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = LabelEncoder().fit_transform(df[column].astype(str))
        
        score_data = df.drop(['Course', 'Branch'], axis=1, errors='ignore')
        return df, weight, branch_data, score_data, label_encoder
    except Exception as e:
        logging.error(f"❌ Error reading Excel files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading Excel files: {str(e)}")

# ✅ เชื่อมต่อ Google Sheets
def connect_google_sheets():
    try:
        logging.debug("🔑 Connecting to Google Sheets...")
        google_credentials_base64 = os.getenv('GOOGLE_CREDENTIALS')

        if not google_credentials_base64:
            logging.error("❌ GOOGLE_CREDENTIALS is missing!")
            raise HTTPException(status_code=500, detail="Google Sheets credentials not found!")

        google_credentials_json = base64.b64decode(google_credentials_base64)
        credentials_dict = json.loads(google_credentials_json)
        creds = Credentials.from_service_account_info(credentials_dict)
        client = gspread.authorize(creds)
        sheet = client.open("Data_project_like_course_branch").sheet1
        logging.debug("✅ Connected to Google Sheets successfully!")
        return sheet
    except Exception as e:
        logging.error(f"❌ Google Sheets connection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Google Sheets connection error: {str(e)}")
        
# ✅ ฟังก์ชันคำนวณการแนะนำสาขา
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

# ✅ API แนะนำ
@app.post("/api/recommend")
async def recommend(payload: Dict[str, Dict[str, str]]):
    try:
        logging.debug(f"📥 Received Payload: {payload}")
        df, weight, branch_data, score_data, label_encoder = load_data()
        personality_values = [int(payload['personality_answers'][key]) for key in sorted(payload['personality_answers'])]
        subject_values = [float(payload['scores'][key]) for key in sorted(payload['scores'])]
        similarity_scores = cosine_similarity([personality_values], score_data)[0]
        recommended_courses = list(label_encoder.inverse_transform(df.iloc[np.argsort(similarity_scores)[-5:][::-1]]['Course']))
        logging.debug(f"🎓 Recommended Courses: {recommended_courses}")
        recommended_branches = get_recommended_branches(recommended_courses, subject_values, branch_data, weight)
        logging.debug(f"🏫 Recommended Branches: {recommended_branches}")
        return {
            "ผลการแนะนำ": {
                "คณะที่แนะนำตามบุคลิก": [{"name": course} for course in recommended_courses],
                "สาขาที่แนะนำตามน้ำหนักคะแนนและบุคลิก": recommended_branches or ["ไม่มีสาขาที่ตรงกับเกณฑ์ของคุณ"]
            }
        }
    except Exception as e:
        logging.error(f"❌ Error in /api/recommend: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ✅ API บันทึกผล
@app.post("/api/save-liked-result")
async def save_liked_result(data: Dict):
    try:
        sheet = connect_google_sheets()
        new_data = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + list(data['personalityAnswers'].values()) + list(data['scores'].values()) + [c['name'] for c in data['recommendations']['คณะที่แนะนำตามบุคลิก']] + data['recommendations']['สาขาที่แนะนำตามน้ำหนักคะแนนและบุคลิก']
        sheet.append_row(new_data)
        return {"success": True, "message": "Data saved successfully"}
    except Exception as e:
        logging.error(f"❌ Error saving to Google Sheets: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
