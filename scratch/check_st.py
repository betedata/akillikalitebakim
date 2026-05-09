import streamlit as st
import inspect

try:
    print(f"Streamlit version: {st.__version__}")
    print(f"st.image signature: {inspect.signature(st.image)}")
except Exception as e:
    print(f"Error: {e}")
