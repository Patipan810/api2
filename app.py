import os
import base64
import json
import logging
import requests
from datetime import datetime
from pytz import timezone
from typing import Dict, List

import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from fastapi import HTTPException

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()
thai_tz = timezone("Asia/Bangkok")

# อนุญาต CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://itbit0267.cpkkuhost.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ตั้งค่า URL ไฟล์จาก GitHub (ต้องแทนที่ USERNAME, REPO, BRANCH ด้วยค่าที่ถูกต้อง)
GITHUB_FILES = {
    "Respon.xlsx": "https://raw.githubusercontent.com/Patipan810/api2/main/Respon.xlsx",
    "Weight.xlsx": "https://raw.githubusercontent.com/Patipan810/api2/main/Weight.xlsx",
    "BranchID.Name.xlsx": "https://raw.githubusercontent.com/Patipan810/api2/main/BranchID.Name.xlsx"
}

# ตั้งค่าชื่อ Google Sheet ของคุณ
GOOGLE_SHEET_NAME = "Data_project_like_course_branch"

# เชื่อมค่อกับgoogle sheetโดยแปลง เป็น base 64
def connect_google_sheets():
    try:
        # อ่านค่า Base64 จาก Environment Variable
        encoded_credentials = os.getenv("GOOGLE_CREDENTIALS")
        if not encoded_credentials:
            logging.error("🚨 Missing Google Credentials in Environment Variables!")
            raise HTTPException(status_code=500, detail="Missing Google Credentials in Environment Variables")
        logging.info("🚀 GOOGLE_CREDENTIALS: %s", os.getenv("GOOGLE_CREDENTIALS"))
        # ถอดรหัส Base64 เป็น JSON
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        credentials_info = json.loads(decoded_credentials)

        # ใช้ JSON ที่ได้มาเพื่อเชื่อมต่อ Google Sheets
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
        client = gspread.authorize(creds)

        # เปิด Google Sheet
        sheet = client.open(GOOGLE_SHEET_NAME).sheet1
        logging.info("✅ Google Sheets connected successfully!")
        return sheet
    except json.JSONDecodeError as json_error:
        logging.error(f"🚨 JSON Decode Error: {json_error}")
        raise HTTPException(status_code=500, detail=f"Invalid Google Credentials format: {str(json_error)}")
    except Exception as e:
        logging.error(f"🚨 Google Sheets connection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Google Sheets connection error: {str(e)}")

# ฟังก์ชันดาวน์โหลดไฟล์
def download_file(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        with open(filename, 'wb') as f:
            f.write(response.content)
        logging.info(f"Downloaded {filename}")
    except Exception as e:
        logging.error(f"Error downloading {filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error downloading {filename}: {str(e)}")

# ฟังก์ชันโหลดข้อมูล
def load_data():
    try:
        for filename, url in GITHUB_FILES.items():
            if not os.path.exists(filename):
                download_file(url, filename)
        
        df = pd.read_excel("Respon.xlsx")
        weight = pd.read_excel("Weight.xlsx")
        branch_data = pd.read_excel("BranchID.Name.xlsx")
        
        if df.empty or weight.empty or branch_data.empty:
            raise HTTPException(status_code=500, detail="One or more data files are empty!")

        # 2. ลบคอลัมน์ที่ไม่ต้องการออก
        df = df.drop(['Timestamp', 'User'], axis=1, errors='ignore')

        # 3. ตรวจสอบว่ามีคอลัมน์ที่ต้องใช้หรือไม่
        if 'Course' in df.columns:
            label_encoder = LabelEncoder()
            df['Course'] = label_encoder.fit_transform(df['Course'].astype(str))
        else:
            raise HTTPException(status_code=500, detail="Missing 'Course' column in data.")

        # ตรวจสอบว่าคอลัมน์ใดที่ไม่ใช่ตัวเลข และแปลงด้วย LabelEncoder
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = LabelEncoder().fit_transform(df[column])

        # 4. สร้าง DataFrame สำหรับคะแนนทั้งหมด
        score_data = df.drop(['Course', 'Branch'], axis=1, errors='ignore')
        
        return df, weight, branch_data, score_data, label_encoder
    except Exception as e:
        logging.error(f"Error reading Excel files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading Excel files: {str(e)}")

# ✅ ฟังก์ชันประมวลผลข้อมูล
def process_personality_answers(personality_answers: Dict[str, str]) -> List[int]:
    processed = {int(k.replace('answers[', '').replace(']', '')): int(v) for k, v in personality_answers.items()}
    return [processed[k] for k in sorted(processed.keys())]

def process_subject_scores(scores: Dict[str, str]) -> List[float]:
    processed = {int(k.replace('scores[', '').replace(']', '')): float(v) for k, v in scores.items()}
    return [processed[k] for k in sorted(processed.keys())]

# ✅ ฟังก์ชันแนะนำคณะ
def get_recommended_courses(personality_values: List[int], df, score_data, label_encoder) -> List[str]:
    similarity_scores = cosine_similarity([personality_values], score_data)[0]
    top_courses = np.argsort(similarity_scores)[-5:][::-1]
    return list(label_encoder.inverse_transform(df.iloc[top_courses]['Course']))

# ✅ ฟังก์ชันแนะนำสาขา
def get_recommended_branches(courses: List[str], subject_scores: List[float], branch_data, weight) -> List[str]:
    relevant_branches = branch_data[branch_data['Course'].isin(courses)]
    if relevant_branches.empty:
        return []
    
    relevant_weights = weight[weight['BranchID'].isin(relevant_branches['BranchID'].values)]
    user_scores_array = np.array(subject_scores).reshape(1, -1)
    filtered_weight = relevant_weights.drop(columns=['BranchID']).fillna(0)
    
    min_cols = min(user_scores_array.shape[1], filtered_weight.shape[1])
    user_scores_array = user_scores_array[:, :min_cols]
    filtered_weight = filtered_weight.iloc[:, :min_cols]
    
    cosine_similarities = cosine_similarity(user_scores_array, filtered_weight)[0]
    results = pd.DataFrame({
        'BranchID': relevant_weights['BranchID'].values,
        'Similarity': cosine_similarities
    })
    
    top_branches = results[results['Similarity'] > 0].nlargest(10, 'Similarity')
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
async def recommend(payload: Dict[str, Dict[str, str]]):
    try:
        df, weight, branch_data, score_data, label_encoder = load_data()
        personality_values = process_personality_answers(payload['personality_answers'])
        subject_values = process_subject_scores(payload['scores'])
        
        recommended_courses = get_recommended_courses(personality_values, df, score_data, label_encoder)
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

@app.post("/api/save-liked-result")
async def save_liked_result(data: Dict):
    try:
        logging.info("🔹 Data received: %s", json.dumps(data, ensure_ascii=False))

        sheet = connect_google_sheets()
        if not sheet:
            raise HTTPException(status_code=500, detail="Google Sheets connection failed")

        required_keys = ["personalityAnswers", "scores", "recommendations"]
        for key in required_keys:
            if key not in data:
                raise HTTPException(status_code=400, detail=f"Missing key: {key}")

        if "คณะที่แนะนำตามบุคลิก" not in data["recommendations"] or "สาขาที่แนะนำตามน้ำหนักคะแนนและบุคลิก" not in data["recommendations"]:
            raise HTTPException(status_code=400, detail="Missing keys in recommendations")

        new_data = [
            datetime.now(thai_tz).strftime("%Y-%m-%d %H:%M:%S"),  # ✅ เวลาไทย (GMT+7)
            *[str(v) for v in data["personalityAnswers"].values()],
            *[str(v) for v in data["scores"].values()],
            *[c["name"] for c in data["recommendations"]["คณะที่แนะนำตามบุคลิก"]],
            *data["recommendations"]["สาขาที่แนะนำตามน้ำหนักคะแนนและบุคลิก"]
        ]

        logging.info("✅ Data structure: %s", json.dumps(new_data, ensure_ascii=False))

        try:
            sheet.append_row(new_data)
            logging.info("✅ Data appended successfully!")
        except Exception as e:
            logging.error("🔥 Failed to append row: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")

        return {"success": True, "message": "Data saved to Google Sheets successfully"}

    except KeyError as e:
        logging.error("🚨 KeyError: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Missing key in request data: {str(e)}")
    except Exception as e:
        logging.error("🔥 ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
