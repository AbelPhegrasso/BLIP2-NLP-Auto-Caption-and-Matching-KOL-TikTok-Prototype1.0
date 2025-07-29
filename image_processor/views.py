from django.shortcuts import render
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from keybert import KeyBERT # spacy ไม่ได้ใช้ใน KeyBERT โดยตรง, แต่ถ้าใช้ในส่วนอื่นก็โหลดได้
result = {}
# --- โหลดโมเดลและข้อมูลเพียงครั้งเดียวเมื่อแอปพลิเคชันเริ่มต้น ---
# วิธีนี้จะช่วยประหยัดทรัพยากรและเวลาในการโหลดโมเดลซ้ำๆ ทุก request
try:
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip2-opt-2.7b")
    kw_model = KeyBERT() # spacy.load("en_core_web_sm") ไม่ได้ใช้โดยตรงใน KeyBERT
    
    Path ที่ถูกต้องสำหรับไฟล์ CSV
    keywords_csv_path = os.path.join(settings.BASE_DIR, 'image_processor', 'data', 'keywords.csv')
    dataset_csv_path = os.path.join(settings.BASE_DIR, 'image_processor', 'data', 'Dataset.csv')
    # keywords_csv_path = "C:\\Users\\AbelH\\Documents\\AI project\\Ai_tiktok\\image_processor\\data\\keywords.csv"
    # dataset_csv_path = "C:\\Users\\AbelH\\Documents\\AI project\\Ai_tiktok\\image_processor\\data\\Dataset.csv"
    csv_file_keyword = pd.read_csv(keywords_csv_path)
    df_dataset = pd.read_csv(dataset_csv_path) # โหลด dataset ครั้งเดียว
    
    MODELS_LOADED = True
except Exception as e:
    print(f"Error loading models or CSV files: {e}")
    MODELS_LOADED = False
# -----------------------------------------------------------------

def predict_image(image_path):
    if not MODELS_LOADED:
        return "Error: Models not loaded."
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=100)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption
    except Exception as e:
        return f"Error during image prediction: {e}"

def NLP(caption):
    if not MODELS_LOADED:
        return {"Error": "Models not loaded."}
    
    try:
        keywords = kw_model.extract_keywords(caption, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5)
        result["Keywords"] = [kw[0] for kw in keywords]

        # ตรวจสอบว่าคอลัมน์มีอยู่จริงก่อนใช้งาน
        if any(kw in caption.lower() for kw in csv_file_keyword.iloc[:,0].astype(str)):
            result["Category"] = ['เสริมความงาม', 'คลินิกเสริมความงาม', 'clinic', 'ศัลยกรรม', 'ผู้หญิงสวย']
        elif any(kw in caption.lower() for kw in csv_file_keyword.iloc[:,2].astype(str)):
            result["Category"] = ['อาหารเสริม', 'กินดีชีวิตดี', 'คอลลาเจน', 'วิตามิน', 'บำรุง']
        elif any(kw in caption.lower() for kw in csv_file_keyword.iloc[:,1].astype(str)):
            result["Category"] = ['ออกกำลังกาย', 'สุขภาพดี', 'ลดพุง', 'ดูเเลตนเอง', 'ออกกำลังกายที่บ้าน']
        else:
            result["Category"] = ['other']
        return result
    except Exception as e:
        return {"Error": f"Error during NLP processing: {e}"}

def matching(result):
    matched_rows = []
    
    for index, row in df_dataset.iterrows():
        text = str(row[2]).lower()  
        if any(keyword.lower() in text for keyword in result.get("Category", []) + result.get("Keywords", [])):
            matched_rows.append(row)

    matched_df = pd.DataFrame(matched_rows)

    if matched_df.empty:
        print("ไม่พบข้อมูลที่แมตช์")
        return [] # Return empty DataFrame

    # ตรวจสอบว่าคอลัมน์ที่ 4, 5, 6 มีอยู่จริงก่อนคำนวณคะแนน
    matched_df["score"] = (
        matched_df.iloc[:, 4].fillna(0) * 3 + 
        matched_df.iloc[:, 5].fillna(0) * 2 +  
        matched_df.iloc[:, 6].fillna(0) * 1    
    )
    ranked_df = matched_df.sort_values(by="score", ascending=False)
    
    ranked_df = matched_df.sort_values(by="score", ascending=False)
    top_10 = ranked_df.iloc[:10, [1, 12]]

    return top_10

def upload_image(request):
    top_10_results = None
    image_url = None
    caption_text = None
    nlp_processed_data = None
    error_message = None

    if request.method == 'POST' and request.FILES['image_file']:
        if not MODELS_LOADED:
            error_message = "ไม่สามารถโหลดโมเดลหรือข้อมูลได้ กรุณาตรวจสอบการตั้งค่าและไฟล์."
        else:
            uploaded_file = request.FILES['image_file']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            image_url = fs.url(filename) # ได้ URL ของรูปภาพที่อัปโหลด

            # สร้างพาธเต็มสำหรับรูปภาพ
            image_path = os.path.join(settings.MEDIA_ROOT, filename)

            try:
                caption_text = predict_image(image_path)
                nlp_processed_data = NLP(caption_text)
                
                if "Error" in nlp_processed_data:
                    error_message = nlp_processed_data["Error"]
                else:
                    top_10 = matching(nlp_processed_data)
                    if not top_10.empty:
                        # แปลง DataFrame เป็น list ของ dictionaries เพื่อให้ส่งไปที่ template ได้ง่ายขึ้น
                        top_10.columns = ['MatchedName', 'URLs']
                        top_10_results = top_10.to_dict('records')
                    else:
                        error_message = "ไม่พบข้อมูลที่ตรงกันจากฐานข้อมูล"
            except Exception as e:
                error_message = f"เกิดข้อผิดพลาดในการประมวลผล: {e}"
    
    context = {
        'image_url': image_url,
        'caption_text': caption_text,
        'nlp_processed_data': nlp_processed_data,
        'top_10_results': top_10_results,
        'error_message': error_message,
    }
    return render(request, 'image_processor/upload.html', context)
# Create your views here.
