import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
SEARCH_ENDPOINT = f"{API_URL}/search"

st.set_page_config(page_title="Visual Search Engine", layout="wide")

st.title("Visual Search Engine")
st.markdown("Upload an image to find visually similar images from our database.")

st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    'Choose AI Model',
    ('resnet', 'vit'),
)
top_k = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=5)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Query Image")
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Search Results")
        if st.button("Search"):
            with st.spinner(f"Searching with {model_choice} for matches..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {"top_k": top_k, "model" : model_choice}

                    response = requests.post(SEARCH_ENDPOINT, files=files, params=params)

                    if response.status_code == 200:
                        results = response.json().get("results", [])

                        if not results:
                            st.warning("No results found.")

                        cols = st.columns(3)
                        for idx, result in enumerate(results):
                            with cols[idx % 3]:
                                img_url = result.get("image_url")

                                st.image(img_url, caption=f"Rank {result['rank']} (Dist: {result['distance']})", use_container_width=True)

                    else:
                        st.error(f"Error {response.status_code}: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend API. Is it running?")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
