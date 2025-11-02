# flipkart/data_ingestion.py
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from flipkart.data_converter import DataConverter
from flipkart.config import Config
import os


class DataIngestor:
    """Handles ingestion of product data into AstraDB Vector Store (optimized for reuse)."""

    def __init__(self, csv_path="data/Ecom_Product_Dataset.csv"):
        self.csv_path = csv_path
        self.embedding = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.store = AstraDBVectorStore(
            collection_name=Config.COLLECTION_NAME,
            embedding=self.embedding,
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )

    def ingest(self, update_existing=False):
        """
        Connects to AstraDB and uploads data *only if needed*.
        - update_existing=False → Fast startup, reuse existing collection.
        - update_existing=True → Re-ingest or update data.
        """

        print(f"🔗 Connecting to AstraDB collection '{Config.COLLECTION_NAME}'...")

        # Fast-path: reuse if collection already exists
        try:
            test_results = self.store.similarity_search("test", k=1)
            print(f"✅ Existing collection found in Astra DB. Skipping re-ingestion for fast startup.")
            return self.store
        except Exception as e:
            print(f"⚠️ Could not verify collection existence: {e}")
            print(f"🆕 Creating new Astra DB collection and ingesting data...")

        # Only re-ingest if explicitly requested or collection is missing
        if update_existing or not os.path.exists(self.csv_path):
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"❌ CSV file not found at {self.csv_path}")
            print(f"📂 Loading dataset: {self.csv_path}")

            converter = DataConverter(self.csv_path)
            docs = converter.convert()
            print(f"📦 Prepared {len(docs)} product documents...")

            try:
                self.store.add_documents(docs)
                print(f"✅ Successfully ingested {len(docs)} records into Astra DB.")
            except Exception as e:
                print(f"⚠️ AstraDB ingestion error: {e}")

        return self.store
