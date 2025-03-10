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

# ✅ ฟังก์ชันโหลดข้อมูลจากไฟล์
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

        # 3. แปลงคอลัมน์ 'Course' ให้เป็นตัวเลขโดยใช้ LabelEncoder
        label_encoder = LabelEncoder()
        df['Course'] = label_encoder.fit_transform(df['Course'].astype(str))

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
    return [int(personality_answers[key]) for key in sorted(personality_answers)]

def process_subject_scores(scores: Dict[str, str]) -> List[float]:
    return [float(scores[key]) for key in sorted(scores)]

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
        # โหลดข้อมูล
        df, weight, branch_data, score_data, label_encoder = load_data()

        # ประมวลผลคำตอบของผู้ใช้
        personality_values = process_personality_answers(payload['personality_answers'])
        subject_values = process_subject_scores(payload['scores'])
        
        # แนะนำคณะ
        recommended_courses = get_recommended_courses(personality_values, df, score_data, label_encoder)

        # แนะนำสาขา
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
