from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import spacy
from keybert import KeyBERT
import pandas as pd

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip2-opt-2.7b")
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

csv_file_keyword = pd.read_csv("C:\\Users\\AbelH\\Documents\\keywords.csv")

result = {}

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=100)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return caption

def NLP(caption):
    keywords = kw_model.extract_keywords(caption, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5)
    result["Keywords"] = [kw[0] for kw in keywords]

    if any(kw in caption.lower() for kw in csv_file_keyword.iloc[:,0]):
        result["Category"] = ['เสริมความงาม', 'คลินิกเสริมความงาม', 'clinic', 'ศัลยกรรม', 'ผู้หญิงสวย']
    elif any(kw in caption.lower() for kw in csv_file_keyword.iloc[:,2]):
        result["Category"] = ['อาหารเสริม', 'กินดีชีวิตดี', 'คอลลาเจน', 'วิตามิน', 'บำรุง']
    elif any(kw in caption.lower() for kw in csv_file_keyword.iloc[:,1]):
        result["Category"] = ['ออกกำลังกาย', 'สุขภาพดี', 'ลดพุง', 'ดูเเลตนเอง', 'ออกกำลังกายที่บ้าน']
    else:
        result["Category"] = ['other']
    return result

def matching():
    df = pd.read_csv("C:\\Users\\AbelH\\Documents\\Dataset.csv")
    matched_rows = []

    for index, row in df.iterrows():
        text = str(row[2]).lower()  
        if any(keyword.lower() in text for keyword in result.get("Category", []) + result.get("Keywords", [])):
            matched_rows.append(row)

    matched_df = pd.DataFrame(matched_rows)

    if matched_df.empty:
        print("ไม่พบข้อมูลที่แมตช์")
        return []

    matched_df["score"] = (
        matched_df.iloc[:, 4].fillna(0) * 3 + 
        matched_df.iloc[:, 5].fillna(0) * 2 +  
        matched_df.iloc[:, 6].fillna(0) * 1    
    )

    ranked_df = matched_df.sort_values(by="score", ascending=False)
    top_10 = ranked_df.iloc[:10, [1]]

    return top_10