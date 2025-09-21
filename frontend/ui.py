import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import time

#Configuration
API_BASE_URL = "http://backend:8000"  # Service name in docker-compose

st.set_page_config(
    page_title="Gap Junction Segmentation",
    page_icon="🧬",
    layout="wide"
)

st.title("Gap Junction Segmentation Online Toolkit")
st.markdown("Upload a dataset to get started. Visualize, evaluate, and interact with results!")

#Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Upload Dataset", "Run Inference", "View Results"])

#Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'inference_complete' not in st.session_state:
    st.session_state.inference_complete = False
    
#Functions
def upload_files_to_backend(files):
    """Upload files to the backend API"""
    files_data = []
    for uploaded_file in files:
        files_data.append(('files', (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))
    
    try:
        response = requests.post(f"{API_BASE_URL}/upload-dataset", files=files_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload failed: {e}")
        return None

def run_inference(session_id):
    """Trigger inference on the backend"""
    try:
        response = requests.post(f"{API_BASE_URL}/run-inference/{session_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Inference failed: {e}")
        return None

def get_visualization(session_id, image_name=None):
    """Get visualization from backend"""
    try:
        params = {"image_name": image_name} if image_name else {}
        response = requests.get(f"{API_BASE_URL}/visualize/{session_id}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Visualization failed: {e}")
        return None

def get_evaluation(session_id):
    """Get evaluation metrics from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/evaluate/{session_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Evaluation failed: {e}")
        return None

#Page 1: Upload Dataset
if page == "Upload Dataset":
    st.header("📁 Upload Your Dataset")
    
    st.markdown("""
    ### Instructions:
    1. Upload your images and corresponding ground truth masks
    2. Image files: `image_name.png`
    3. Ground truth files: `image_name_label.png`
    4. Supported formats: PNG, JPG, JPEG
    """)
    
    uploaded_files = st.file_uploader(
        "Choose image and label files",
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_files:
        st.write(f"📊 **{len(uploaded_files)} files selected**")
        
        # Show file preview
        img_files = [f for f in uploaded_files if '_label' not in f.name]
        label_files = [f for f in uploaded_files if '_label' in f.name]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"🖼️ Images: {len(img_files)}")
        with col2:
            st.write(f"🎯 Labels: {len(label_files)}")
        
        if st.button("🚀 Upload Dataset", type="primary"):
            with st.spinner("Uploading files..."):
                result = upload_files_to_backend(uploaded_files)
                
            if result:
                st.success("✅ Dataset uploaded successfully!")
                st.session_state.session_id = result['session_id']
                st.json(result)
                st.info(f"Session ID: {st.session_state.session_id}")

#Page 2: Run Inference
elif page == "Run Inference":
    st.header("🔬 Run Model Inference")
    
    if not st.session_state.session_id:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        st.info(f"Current session: {st.session_state.session_id}")
        
        if st.button("🎯 Start Inference", type="primary"):
            with st.spinner("Running inference... This may take a few minutes."):
                result = run_inference(st.session_state.session_id)
            
            if result:
                st.success("✅ Inference completed!")
                st.session_state.inference_complete = True
                st.json(result)

# Page 3: View Results
elif page == "View Results":
    st.header("📊 View Results")
    
    if not st.session_state.session_id:
        st.warning("⚠️ Please upload a dataset first!")
    elif not st.session_state.inference_complete:
        st.warning("⚠️ Please run inference first!")
    else:
        st.info(f"Showing results for session: {st.session_state.session_id}")
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["🖼️ Visualizations", "📈 Evaluation Metrics"])
        
        with tab1:
            st.subheader("Prediction Visualizations")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("🎲 Random Visualization"):
                    with st.spinner("Generating visualization..."):
                        viz_result = get_visualization(st.session_state.session_id)
                    
                    if viz_result:
                        # Decode base64 image
                        img_data = base64.b64decode(viz_result['visualization'])
                        img = Image.open(BytesIO(img_data))
                        
                        with col2:
                            st.image(img, caption="Prediction vs Ground Truth", use_column_width=True)
        
        with tab2:
            st.subheader("Model Performance Metrics")
            
            if st.button("📊 Generate Evaluation"):
                with st.spinner("Calculating metrics..."):
                    eval_result = get_evaluation(st.session_state.session_id)
                
                if eval_result:
                    # Decode base64 image
                    img_data = base64.b64decode(eval_result['evaluation_plot'])
                    img = Image.open(BytesIO(img_data))
                    
                    st.image(img, caption="Performance Metrics", use_column_width=True)

# Sidebar status
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Status")
if st.session_state.session_id:
    st.sidebar.success(f"✅ Session: {st.session_state.session_id}")
else:
    st.sidebar.error("❌ No active session")

if st.session_state.inference_complete:
    st.sidebar.success("✅ Inference complete")
else:
    st.sidebar.warning("⏳ Inference pending")