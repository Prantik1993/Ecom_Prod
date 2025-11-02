# data_converter.py
import pandas as pd
from langchain_core.documents import Document

class DataConverter:
    """Converts CSV rows into LangChain Document objects for AstraDB ingestion."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def convert(self):
        df = pd.read_csv(self.file_path)
        for c in ["product_title", "summary", "review", "rating"]:
            if c not in df.columns:
                df[c] = ""

        docs = []
        for _, row in df.iterrows():
            text = (
        f"Product: {row['product_title']}. "
        f"Category: {row.get('category', '')}. "
        f"Summary: {row.get('summary', '')}. "
        f"Customer Review: {row.get('review', '')}. "
        f"Rating: {row.get('rating', '')}. "
        f"Price: {row.get('price', '')}."
        )

            meta = {
                "title": row["product_title"],
                "rating": row.get("rating", ""),
                "summary": row.get("summary", "")
            }
            docs.append(Document(page_content=text, metadata=meta))
        return docs
