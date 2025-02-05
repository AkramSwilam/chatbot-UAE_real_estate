import streamlit as st
import requests


API_URL = "http://localhost:8000/search"

st.title("Property Search")

query = st.text_input("Enter your property query:")

if st.button("Search"):
    if query:
        response = requests.post(API_URL, json={"query": query})

        if response.status_code == 200:
            result = response.json()
            st.write("Response from FastAPI:")
            st.write(result.get("response"))
        else:
            st.error(f"Error: {response.status_code}")
    else:
        st.warning("Please enter a query to search.")
