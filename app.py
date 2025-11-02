from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import re

from flipkart.data_ingestion import DataIngestor
from flipkart.recommender import ProductRecommender

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "flipkart-secret-key")
CORS(app)

data_ingestor = None
vector_store = None
recommender = None

def init_recommender():
    """Initialize LLM Recommender (only once to save Astra tokens)."""
    global data_ingestor, vector_store, recommender
    if recommender:
        return
    print("Initializing LLM Recommender")
    data_ingestor = DataIngestor()
    vector_store = data_ingestor.ingest(update_existing=False)
    recommender = ProductRecommender(vector_store)
    print("Recommender ready.")


@app.route("/")
def home():
    """Render main chat interface."""
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend_products():
    """Main route for handling user queries and returning results."""
    user_query = request.form.get("msg", "").strip()
    if not user_query:
        return jsonify({"intent": "unknown", "explanation": "Please enter a query.", "products": []})
    init_recommender()
    try:
        intent = recommender.classify_intent(user_query)
        print(f"Intent detected: {intent}")
    except Exception as e:
        print(f"Intent classification failed: {e}")
        intent = "unknown"

    
    if intent == "shopping":
        # Extract numeric value for top_k (default = 5)
        match = re.search(r"\b(\d+)\b", user_query)
        num_requested = int(match.group(1)) if match else 5

        try:
            result = recommender.recommend(user_query, top_k=num_requested)
            # result already includes "explanation" and "products"
            result["intent"] = intent
            return jsonify(result)
        except Exception as e:
            print(f"Error during recommendation: {e}")
            return jsonify({
                "intent": intent,
                "explanation": "Sorry, something went wrong while fetching product recommendations 😅",
                "products": []
            })

    
    elif intent == "greeting":
        reply = "Hi there! I’m your Flipkart AI shopping assistant. How can I help you today?"
        return jsonify({"intent": intent, "explanation": reply, "products": []})

    
    elif intent == "chat":
        try:
            reply = recommender.small_talk(user_query)
        except Exception:
            reply = "I'm your Flipkart shopping assistant — I can help you find products, deals, or comparisons 😊"
        return jsonify({"intent": intent, "explanation": reply, "products": []})
    else:
        reply = (
            "I’m your Flipkart AI shopping assistant — I can help you find products, compare prices, "
            "or suggest top deals"
        )
        return jsonify({"intent": intent, "explanation": reply, "products": []})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
