import streamlit as st
import os
from utils import get_answer, text_to_speech, autoplay_audio, speech_to_text
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

# Float feature initialization
float_init()

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """Welcome to Restaurant Insights! How can I assist you today? 

Here are some ways I can help:

1. **Sales Analysis**: Compare current sales to last year's sales, identify factors impacting sales differences, and analyze weekly and annual sales data.
2. **Cost Analysis**: Provide insights on Cost of Goods Sold (COGS) over the past few years, identify trends in costs, and suggest areas for improvement.
3. **Promotion Planning**: Calculate the additional traffic needed to break even on promotions.
4. **Staffing**: Forecast staffing needs for upcoming weeks.
5. **Inventory Ordering**: Recommend ordering quantities based on historical data and forecasts.
6. **General Operations**: Provide recommendations for improving sales, reducing costs, and optimizing staff scheduling based on available data.

Feel free to ask specific questions, and I'll provide you with the most relevant information and actionable insights."""}
        ]
    # if "audio_initialized" not in st.session_state:
    #     st.session_state.audio_initialized = False

initialize_session_state()

# # Display MOD Pizza logo and title in the same line
import base64

def image_to_data_url(image_path):
    with open(image_path, "rb") as img_file:
        return "data:image/png;base64," + base64.b64encode(img_file.read()).decode("utf-8")

# Convert your image
data_url = image_to_data_url("data/mod_logo.png")
md_logo = image_to_data_url("data/md_logo.png")


# Title
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="{data_url}" width="50" style="margin-right: 10px;">
        <h1 style="margin: 0;">Restaurant Insights</h1>
    </div>
    """,
    unsafe_allow_html=True
)



# Create footer container for the microphone
footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder()


pizza_path = "data/pizza.png"
# questoion_path = "data/question.png"
quest_path = "data/quest.png"

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=pizza_path if message["role"] == "assistant" else quest_path):
        st.write(message["content"])

if audio_bytes:
    # Write the audio bytes to a file
    with st.spinner("Transcribing..."):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)

        transcript = speech_to_text(webm_file_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            with st.chat_message("user", avatar=quest_path):
                st.write(transcript)
            os.remove(webm_file_path)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar=pizza_path):
        with st.spinner("Thinking🤔..."):
            final_response = get_answer(st.session_state.messages)
        # with st.spinner("Generating audio response..."):    
        #     audio_file = text_to_speech(final_response)
        #     autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        # os.remove(audio_file)

# Float the footer container and provide CSS to target it with
footer_container.float("bottom: 0rem;")