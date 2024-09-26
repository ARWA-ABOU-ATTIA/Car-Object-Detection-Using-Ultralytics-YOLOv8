import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# تحميل النموذج المدرب
@st.cache_resource
def load_model():
    return YOLO('car_detection_model.pt')

model = load_model()

st.title('Car Detection App')

# خيار تحميل صورة أو فيديو
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    # حفظ الملف في ملف مؤقت
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # تحقق من نوع الملف
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        # قراءة الصورة
        image = Image.open(tfile.name)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # تحويل الصورة إلى مصفوفة NumPy
        img_array = np.array(image)

        # إجراء الكشف
        results = model(img_array)

        # عرض النتائج
        st.subheader('Detection Results:')
        
        # تحويل الصورة إلى BGR (للرسم)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_bgr, f'Car: {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # تحويل الصورة مرة أخرى إلى RGB للعرض
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption='Detection Result', use_column_width=True)

        # عرض عدد السيارات المكتشفة
        num_cars = len(boxes)
        st.write(f'Number of cars detected: {num_cars}')

    elif uploaded_file.type in ["video/mp4", "video/quicktime", "video/x-msvideo"]:
        # قراءة الفيديو باستخدام OpenCV
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # إجراء الكشف على كل إطار
            results = model(frame)

            # رسم المستطيلات حول السيارات المكتشفة
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Car: {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # عرض الفيديو في واجهة المستخدم
            stframe.image(frame, channels="BGR")

        cap.release()

st.write('Upload an image or a video to detect cars.')