import streamlit as st
from graph import graph

st.title("Customer Support Assistant")

query = st.text_input("Ask your question")

if st.button("Submit"):

    result = graph.invoke({
        "question": query
    })

    st.success(result["answer"])