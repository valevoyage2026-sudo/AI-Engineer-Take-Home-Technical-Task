import tempfile

import pandas as pd
import streamlit as st

from graph_workflow.Batch_Loader import batch
from models.schema import batch_schema

st.set_page_config(page_title="NovaDeskAI", layout="wide")

st.title("🎫 NovaDeskAI Ticket Triage")

uploaded_file = st.file_uploader("Upload Ticket CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Input Tickets")

    st.dataframe(df)

    if st.button("Process Tickets"):
        with st.spinner("Processing tickets..."):
            # Save uploaded csv temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())

                temp_path = tmp.name

            batch_input = batch_schema(csv_path=temp_path)

            result = batch_schema.model_validate(batch.invoke(batch_input))

            output_file = result.output_csv_path

        st.success("Processing Complete!")

        processed_df = pd.read_csv(f"{output_file}")

        st.subheader("Results")

        st.dataframe(processed_df)

        with open(f"{output_file}", "rb") as f:
            st.download_button(
                label="Download Processed CSV",
                data=f,
                file_name="processed_tickets.csv",
                mime="text/csv",
            )
