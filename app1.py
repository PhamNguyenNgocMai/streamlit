import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os

# Hàm tải mô hình từ Google Drive
def download_model_from_gdrive(model_url, output):
    gdown.download(model_url, output, quiet=False)

# Kiểm tra nếu mô hình đã tồn tại, nếu không thì tải về
model_path = 'model.h5'
if not os.path.exists(model_path):
    with st.spinner('Đang tải mô hình...'):
        model_url = 'https://drive.google.com/file/d/1NMugeY1zH0sJJ6Dl-DlGtxpRxR7s78zl/view?usp=drive_link'
        download_model_from_gdrive(model_url, model_path)
        st.success('Mô hình đã được tải về thành công!')

# Tải mô hình
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Hàm tiền xử lý hình ảnh đã tải lên
def preprocess_image(image):
    image = image.resize((150, 150))  
    image = np.array(image) / 255.0   # Chuẩn hóa hình ảnh
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch
    return image

# Ứng dụng Streamlit
st.title("Ứng Dụng Phân Loại Hình Ảnh X-quang ngực")
st.write("Tải lên một hình ảnh để phân loại là bình thường hay bất thường.")

# Tải lên tập tin
uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị hình ảnh đã tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption='Hình Ảnh Đã Tải Lên', use_column_width=True)
    st.write("Đang phân loại...")

    # Tiền xử lý hình ảnh
    preprocessed_image = preprocess_image(image)

    # Dự đoán
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Hiển thị kết quả
    if predicted_class == 0:
        st.write("Hình ảnh được phân loại là **Bình Thường**.")
    else:
        st.write("Hình ảnh được phân loại là **Viêm phổi**.")
