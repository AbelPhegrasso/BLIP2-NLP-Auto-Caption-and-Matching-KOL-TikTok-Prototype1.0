from django.contrib import admin
from django.urls import path, include
from django.conf import settings # เพิ่ม
from django.conf.urls.static import static # เพิ่ม

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('image_processor.urls')), # รวม URL ของ image_processor
]

# สำหรับการเสิร์ฟไฟล์มีเดียในโหมดพัฒนา
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)