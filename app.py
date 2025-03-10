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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build  # เพิ่มการ import นี้
from googleapiclient.errors import HttpError

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

def connect_google_sheets():
    try:
        service_account_info = json.loads(base64.b64decode(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']))
        creds = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=creds)
        return service.spreadsheets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Google Sheets: {str(e)}")
# ✅ อนุญาต CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://itbit0267.cpkkuhost.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ ตั้งค่า URL ไฟล์จาก GitHub (ต้องแทนที่ USERNAME, REPO, BRANCH ด้วยค่าที่ถูกต้อง)
GITHUB_FILES = {
    "Respon.xlsx": "https://raw.githubusercontent.com/Patipan810/api2/main/Respon.xlsx",
    "Weight.xlsx": "https://raw.githubusercontent.com/Patipan810/api2/main/Weight.xlsx",
    "BranchID.Name.xlsx": "https://raw.githubusercontent.com/Patipan810/api2/main/BranchID.Name.xlsx"
}

# ✅ ฟังก์ชันดาวน์โหลดไฟล์
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

# ✅ ฟังก์ชันโหลดข้อมูล
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

# ระบบแนะนำ
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
        
# ระบบถูกใจ        
@app.post("/api/save-liked-result")
async def save_liked_result(data: Dict):
    try:
        print("🔹 Data received:", data)  # ✅ Debug เช็คข้อมูลที่รับมา
        sheet_service = connect_google_sheets()

        # ค่า spreadsheetId ของคุณที่ได้จาก URL ของ Google Sheets
        spreadsheet_id = "14_IsqgYU6CLCoyheoc4qW5bNWLZp2u5zhqd4g-rqxX8"  # แทนที่ด้วย spreadsheetId ของคุณ

        # ข้อมูลที่ต้องการบันทึกลงใน Google Sheets
        new_data = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            *[v for v in data['personalityAnswers'].values()],
            *[v for v in data['scores'].values()],
            *[c['name'] for c in data['recommendations']['คณะที่แนะนำตามบุคลิก']],
            *data['recommendations']['สาขาที่แนะนำตามน้ำหนักคะแนนและบุคลิก']
        ]

        print("✅ Data to be saved:", new_data)  # ✅ Debug เช็คข้อมูลก่อนบันทึก

        # ใช้ method append เพื่อนำข้อมูลใหม่ไปแทรกที่แถวสุดท้าย
        sheet_service.values().append(
            spreadsheetId=spreadsheet_id,
            range="Sheet1!A1",  # ระบุ Range ที่คุณต้องการบันทึก เช่น "Sheet1!A1" 
            valueInputOption="RAW",  # เลือกเป็น RAW ถ้าไม่ต้องการการแปลงข้อมูล
            body={"values": [new_data]}  # ข้อมูลที่ต้องการเพิ่ม
        ).execute()

        return {"success": True, "message": "Data saved to Google Sheets successfully"}
    except HttpError as e:
        print("🔥 ERROR:", str(e))  # ✅ Debug เช็ค Error ที่เกิดขึ้น
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print("🔥 ERROR:", str(e))  # ✅ Debug เช็ค Error อื่น ๆ
        raise HTTPException(status_code=500, detail=str(e))

