from django.shortcuts import render
from .Backend import predict_image  
import os

def image_caption_view(request):
    caption = None

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_path = f"media/{image_file.name}"

        # บันทึกภาพชั่วคราว
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        # เรียกฟังก์ชัน AI
        caption = predict_image(image_path)

        # ส่ง path ของภาพไปแสดงบนเว็บด้วย
        image_url = f"/media/{image_file.name}"

        return render(request, 'result.html', {'caption': caption, 'image_url': image_url})

    return render(request, 'upload.html')
