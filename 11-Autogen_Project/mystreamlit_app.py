import streamlit as st
import asyncio
import nest_asyncio

from myautogen_backend import run_litrev

nest_asyncio.apply()

st.set_page_config(page_title="Groq Literature Review Assistant")

st.title("📚 Multi-Agent Literature Review")
st.caption("Powered by AutoGen 0.4+ and Groq")

groq_api_key = st.text_input(
    "Enter your Groq API Key",
    type="password"
)

topic = st.text_input("Research Topic")

num_papers = st.slider("Number of Papers", 1, 10, 5)

if st.button("Generate Review"):
    if not groq_api_key:
        st.warning("Please enter your Groq API key.")
    elif not topic:
        st.warning("Please enter a topic.")
    else:
        output_area = st.empty()

        async def run():
            async for chunk in run_litrev(topic, num_papers, groq_api_key):
                output_area.write(chunk)

        asyncio.run(run())