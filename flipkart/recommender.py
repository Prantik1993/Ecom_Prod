from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flipkart.config import Config
import re
from operator import itemgetter


class ProductRecommender:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 50})
   
   
    def classify_intent(self, query: str):
        prompt = ChatPromptTemplate.from_template("""
Classify user intent into one of these categories:
1. shopping - asking about products, prices, deals, ratings, comparisons, etc.
2. greeting - greeting messages like hi, hello, hey.
3. chat - non-shopping talk or general questions.
4. unknown - anything unclear.

User: "{query}"
Answer with only one word: shopping / greeting / chat / unknown
""")
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query}).strip().lower()

    def small_talk(self, query: str):
        prompt = ChatPromptTemplate.from_template("""
You are Flipkart's friendly AI shopping assistant.

If the question is NOT related to e-commerce or products,
do NOT answer it directly.
Instead, say politely:
"I'm your Flipkart AI shopping assistant — I can help you find products, compare prices, or suggest top deals 😊"

If it IS product-related (like phones, deals, or shopping questions),
reply politely but concisely.

User query: {query}
""")
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query}).strip()


    def extract_category(self, query: str):
        query_lower = query.lower()
        categories = {
            "phone": ["phone", "mobile", "smartphone"],
            "laptop": ["laptop", "notebook"],
            "tv": ["tv", "television", "led", "smart tv"],
            "earbuds": ["earbud", "earphone", "headphone"],
            "camera": ["camera", "dslr"],
            "watch": ["watch", "smartwatch"],
            "ac": ["ac", "air conditioner"],
            "fridge": ["fridge", "refrigerator"]
        }
        for cat, kws in categories.items():
            if any(k in query_lower for k in kws):
                return cat
        return None


    def extract_brand(self, query: str):
        known_brands = [
            "samsung", "apple", "realme", "redmi", "mi", "poco", "vivo", "oppo", "oneplus",
            "asus", "lenovo", "dell", "hp", "acer", "boat", "noise", "boult", "panasonic",
            "lg", "sony", "philips", "whirlpool", "godrej"
        ]
        query_lower = query.lower()
        for brand in known_brands:
            if brand in query_lower:
                return brand
        return None

    
    def recommend(self, query: str, top_k: int = 5):
        """
        Industry-grade recommender with:
        - Category + brand + price filtering
        - Smart count detection
        - Confidence-based graceful fallback
        """
        category = self.extract_category(query)
        brand = self.extract_brand(query)
        print(f"🧩 Detected category: {category} | brand: {brand}")

        # Detect numeric request like "5 phones"
        match = re.search(r"\b(\d+)\b", query)
        if match:
            top_k = int(match.group(1))
        print(f"📦 Requested top_k: {top_k}")

        # Detect price filters
        price_limit = None
        min_price = None
        between_match = re.search(r"between\s*(\d+)\s*(?:and|to)\s*(\d+)", query.lower())
        under_match = re.search(r"under\s*(\d+)", query.lower())
        above_match = re.search(r"above\s*(\d+)", query.lower())

        if between_match:
            min_price, price_limit = map(float, between_match.groups())
        elif under_match:
            price_limit = float(under_match.group(1))
        elif above_match:
            min_price = float(above_match.group(1))

        docs = self.retriever.invoke(query)
        if not docs:
            return {"explanation": "No matching products found.", "products": []}

        products, seen = [], set()
        for d in docs:
            title = d.metadata.get("title", "")
            summary = d.metadata.get("summary", "")
            rating_str = d.metadata.get("rating", "0") or "0"
            price_str = d.metadata.get("price", "0") or "0"

            try:
                rating = float(rating_str)
            except:
                rating = 0.0
            try:
                price = float(price_str)
            except:
                price = 0.0

            lower_title = title.lower()

            # Category + brand filtering
            if category and category not in lower_title and category not in summary.lower():
                continue
            if brand and brand not in lower_title and brand not in summary.lower():
                continue

            # Price filtering
            if price_limit and price > price_limit:
                continue
            if min_price and price < min_price:
                continue
            if min_price and price_limit and not (min_price <= price <= price_limit):
                continue

            # Deduplicate
            if lower_title in seen:
                continue
            seen.add(lower_title)

            products.append({
                "title": title,
                "rating": rating,
                "price": price,
                "summary": summary[:150],
                "image_url": d.metadata.get("image_url", "")
            })

        # Ranking logic
        if re.search(r"\b(best|top|good|high|recommended)\b", query.lower()):
            products.sort(key=itemgetter("rating"), reverse=True)
        elif re.search(r"\b(cheapest|lowest|budget)\b", query.lower()):
            products.sort(key=itemgetter("price"))
        else:
            products.sort(key=itemgetter("rating"), reverse=True)

        total_found = len(products)

        # Smart fallback logic
        if total_found == 0:
            return {"explanation": "I couldn’t find any products matching your filters 😅", "products": []}

        if total_found < top_k:
            explanation = (
                f"I could only find {total_found} matching "
                f"{category or 'products'}"
                + (f" from {brand}" if brand else "")
                + (f" under ₹{int(price_limit)}" if price_limit else "")
                + (f" above ₹{int(min_price)}" if min_price else "")
                + ". 😊"
            )
            return {"explanation": explanation, "products": products[:total_found]}

        # If enough items, show exactly N
        return {
            "explanation": "",
            "products": products[:top_k]
        }
