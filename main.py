import os
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import shopify
import openai
import json
import difflib

# For embeddings
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# For Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#########################################################
# Global conversation state (to track last search)
#########################################################
conversation_state: Dict[str, Dict[str, Any]] = {}

#########################################################
# Pinecone Initialization
#########################################################
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "products-index")
pinecone_region = os.getenv("PINECONE_REGION", "us-west-2")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")

if pinecone_api_key:
    pc = Pinecone(api_key=pinecone_api_key)
    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name,
            dimension=384,  # For all-MiniLM-L6-v2 model
            metric="cosine",
            spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)
        )
    pinecone_index = pc.Index(pinecone_index_name)
    logger.info("Pinecone initialized")
else:
    logger.error("Pinecone API key not set")
    pinecone_index = None

#########################################################
# FastAPI Initialization and Middleware
#########################################################
app = FastAPI(title="Chatbot Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#########################################################
# MongoDB Connection
#########################################################
class MongoDB:
    client = None
    db = None

    @classmethod
    async def connect_db(cls, mongodb_url: str, database_name: str):
        cls.client = AsyncIOMotorClient(mongodb_url)
        cls.db = cls.client[database_name]
        await cls.db.users.create_index("user_id", unique=True)
        await cls.db.users.create_index("email", unique=True)
        await cls.db.onboarding.create_index("user_id", unique=True)
        await cls.db.products.create_index([("user_id", 1), ("product_id", 1)], unique=True)

    @classmethod
    def get_db(cls):
        if cls.db is None:
            raise Exception("Database not initialized. Call connect_db first.")
        return cls.db

    @classmethod
    async def close_db(cls):
        if cls.client is not None:
            cls.client.close()

@app.get("/products/{user_id}")
async def get_products(user_id: str):
    try:
        db = MongoDB.get_db()
        cursor = db.products.find({"user_id": user_id})
        products = await cursor.to_list(length=None)
        for product in products:
            if '_id' in product:
                product['_id'] = str(product['_id'])
        return products or []
    except Exception as e:
        logger.error(f"Error fetching products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

#########################################################
# Pydantic Models
#########################################################
class UserBase(BaseModel):
    user_id: str
    company_name: str
    user_name: str
    email: EmailStr

class OnboardingData(BaseModel):
    user_id: str
    industry: str
    goal: str
    support_link: str
    meeting_link: Optional[str]
    general_info: str
    shopify_store: str
    shopify_token: str

class ChatQuery(BaseModel):
    query: str
    user_id: str

#########################################################
# Startup and Shutdown Events
#########################################################
@app.on_event("startup")
async def startup_event():
    try:
        await MongoDB.connect_db("mongodb://localhost:27017", "chatbot_db")
        db = MongoDB.get_db()
        await db.command("ping")
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    await MongoDB.close_db()

#########################################################
# Utility Functions
#########################################################
def measure_time(func):
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"Function {func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper

def clean_html(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ", strip=True)

# Language detection and translation helpers
async def detect_language_openai(text: str) -> str:
    try:
        messages = [
            {"role": "system", "content": "You are a language detection assistant. Reply with only the 2-letter ISO code."},
            {"role": "user", "content": text}
        ]
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0,
            max_tokens=2
        )
        lang_code = response.choices[0].message["content"].strip().lower()
        return lang_code if len(lang_code) == 2 else "en"
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        return "en"

async def translate_from_english(text: str, target_lang: str) -> str:
    try:
        messages = [
            {"role": "system", "content": f"Translate the following text from English to {target_lang}."},
            {"role": "user", "content": text}
        ]
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        translated = response.choices[0].message["content"].strip()
        return translated
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return text

#########################################################
# Embedding Model Helper (using lowercased text)
#########################################################
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def get_embedding(text: str) -> List[float]:
    return embedding_model.encode(text.lower()).tolist()

#########################################################
# LLM-based Classification of User Query (expects lowercase values)
#########################################################
async def classify_user_query(query: str) -> Dict:
    system_instructions = (
        "You are a product classification assistant. Parse the user's query into a JSON object with these keys:\n"
        "  - direct_product: boolean (set to true if the query indicates intent to buy or references a specific product/brand)\n"
        "  - category: string (e.g., 'shoes', 'electronics') in lowercase\n"
        "  - brand: string (e.g., 'adidas', 'apple') in lowercase\n"
        "  - color: string (if mentioned, else empty) in lowercase\n"
        "Return only valid JSON with no extra text."
    )
    user_prompt = f"User query: \"{query}\""
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        content = response.choices[0].message["content"].strip()
        classification = {}
        try:
            classification = json.loads(content)
            if "category" in classification and isinstance(classification["category"], str):
                classification["category"] = classification["category"].strip().lower()
            if "brand" in classification and isinstance(classification["brand"], str):
                classification["brand"] = classification["brand"].strip().lower()
            if "color" in classification and isinstance(classification["color"], str):
                classification["color"] = classification["color"].strip().lower()
        except Exception as e:
            logger.warning("Could not parse JSON from LLM, returning empty classification.")
        return classification
    except Exception as e:
        logger.error(f"Error classifying query: {str(e)}")
        return {}

#########################################################
# Automatic Category Mapping using GPT
#########################################################
async def map_category_to_available(input_cat: str, available: List[str]) -> Optional[str]:
    """
    Maps search term to available categories, optimizing for finding matches
    to avoid unnecessary deep searches.
    """
    prompt = (
        f"Given a search for '{input_cat}' and available categories: {', '.join(available)}\n"
        f"Which single category would be most relevant for this search?\n"
        f"Respond with only the category name in lowercase."
    )
    
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "Choose the most relevant category. Respond with single word only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=20,
        )
        
        mapped = response.choices[0].message["content"].strip().lower()
        
        # Return the mapped category if valid
        if mapped in available:
            return mapped
            
        # Only return None if we absolutely can't map to an available category
        return None
        
    except Exception as e:
        logger.error(f"Error in category mapping: {str(e)}")
        return None

#########################################################
# Shopify Service (Sync Products with Lowercase Metadata)
#########################################################
class ShopifyService:
    def __init__(self, shop_name: str, access_token: str):
        try:
            shop_name = shop_name.replace('https://', '').replace('http://', '')
            shop_name = shop_name.replace('.myshopify.com', '')
            self.shop_url = f"{shop_name}.myshopify.com"
            api_version = '2024-01'
            shop_url = f"https://{self.shop_url}/admin/api/{api_version}"
            logger.info(f"Creating Shopify session for shop: {shop_url}")
            session = shopify.Session(shop_url, api_version, access_token)
            shopify.ShopifyResource.activate_session(session)
            shop = shopify.Shop.current()
            if not shop:
                raise ValueError("Could not connect to Shopify")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info(f"Successfully initialized ShopifyService for {shop.name}")
        except Exception as e:
            logger.error(f"Error initializing ShopifyService: {str(e)}")
            raise

    def clean_html(self, html_text: str) -> str:
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def generate_product_text(self, product) -> str:
        clean_description = self.clean_html(product.body_html or '')
        text_parts = [
            str(product.title or ''),
            clean_description,
            str(product.product_type or ''),
            str(product.vendor or ''),
            str(product.tags or '')
        ]
        for variant in product.variants:
            variant_parts = [
                f"SKU: {variant.sku or ''}",
                f"Title: {variant.title or ''}",
                f"Option: {' '.join(str(option) for option in getattr(variant, 'option_values', []))}"
            ]
            text_parts.extend(variant_parts)
        return ' '.join(filter(None, text_parts))

    async def sync_products(self, db: Any, user_id: str):
        try:
            processed_count = 0
            failed_count = 0
            start_time = time.time()
            current_batch = shopify.Product.find(limit=250)
            logger.info("Starting product sync...")

            while current_batch:
                logger.info(f"Processing batch of products (processed so far: {processed_count})")
                for product in current_batch:
                    try:
                        product_text = " ".join([
                            str(product.title or '').lower(),
                            self.clean_html(product.body_html or '').lower(),
                            str(product.product_type or '').lower(),
                            str(product.vendor or '').lower(),
                            str(product.tags or '').lower()
                        ])
                        title_text = str(product.title or '').lower()
                        title_embedding = self.model.encode(title_text).tolist()
                        description_embedding = self.model.encode(product_text).tolist()

                        try:
                            metafields = shopify.Metafield.find(resource='products', resource_id=product.id)
                            metafields_data = [{
                                'namespace': field.namespace,
                                'key': field.key,
                                'value': field.value,
                                'type': field.type if hasattr(field, 'type') else None
                            } for field in metafields] if metafields else []
                        except Exception as e:
                            logger.warning(f"Error fetching metafields for product {product.id}: {str(e)}")
                            metafields_data = []

                        variants = []
                        if hasattr(product, 'variants'):
                            for variant in product.variants:
                                variant_data = {
                                    'id': str(variant.id),
                                    'sku': str(variant.sku or '').lower(),
                                    'price': float(variant.price or 0),
                                    'inventory_quantity': int(variant.inventory_quantity or 0),
                                    'title': str(variant.title or '').lower()
                                }
                                variants.append(variant_data)

                        images = []
                        if hasattr(product, 'images'):
                            for image in product.images:
                                image_data = {
                                    'id': str(image.id),
                                    'src': str(image.src),
                                    'position': int(image.position or 0),
                                    'alt': str(image.alt or '').lower(),
                                    'width': int(image.width) if hasattr(image, 'width') else None,
                                    'height': int(image.height) if hasattr(image, 'height') else None,
                                    'variant_ids': [str(vid) for vid in image.variant_ids] if hasattr(image, 'variant_ids') else []
                                }
                                images.append(image_data)

                        product_doc = {
                            "user_id": str(user_id),
                            "product_id": str(product.id),
                            "title": str(product.title or '').lower(),
                            "description": self.clean_html(str(product.body_html or '')).lower(),
                            "vendor": str(product.vendor or '').lower(),
                            "product_type": str(product.product_type or '').lower(),
                            "status": str(product.status or '').lower(),
                            "tags": [tag.strip().lower() for tag in str(product.tags or '').split(',')] if product.tags else [],
                            "variants": variants,
                            "images": images,
                            "metafields": metafields_data,
                            "title_embedding": title_embedding,
                            "description_embedding": description_embedding,
                            "last_sync": datetime.utcnow().isoformat()
                        }

                        try:
                            await db.products.replace_one(
                                {"user_id": str(user_id), "product_id": str(product.id)},
                                product_doc,
                                upsert=True
                            )
                            processed_count += 1
                            logger.info(f"Processed product {product.id}")

                            if pinecone_index is not None:
                                vector_id = f"{user_id}_{product.id}"
                                price = 0
                                if product_doc.get("variants") and len(product_doc["variants"]) > 0:
                                    price = product_doc["variants"][0].get("price", 0)
                                first_image = ""
                                if images and len(images) > 0:
                                    first_image = images[0].get("src", "")
                                metadata = {
                                    "user_id": str(user_id),
                                    "product_id": str(product.id),
                                    "title": product_doc["title"],
                                    "price": float(price),
                                    "description": product_doc["description"],
                                    "image": first_image,
                                    "vendor": product_doc["vendor"],
                                    "category": product_doc["product_type"]
                                }
                                try:
                                    pinecone_index.upsert(
                                        vectors=[(vector_id, description_embedding, metadata)]
                                    )
                                    logger.info(f"Upserted product {product.id} to Pinecone")
                                except Exception as pe:
                                    logger.error(f"Error upserting product {product.id} to Pinecone: {str(pe)}")
                        except Exception as e:
                            logger.error(f"Error saving product {product.id} to MongoDB: {str(e)}")
                            failed_count += 1
                            continue
                    except Exception as e:
                        logger.error(f"Error processing product {product.id}: {str(e)}")
                        failed_count += 1
                        continue

                if current_batch:
                    last_id = current_batch[-1].id
                    current_batch = shopify.Product.find(limit=250, since_id=last_id)
                else:
                    break
                time.sleep(0.5)

            end_time = time.time()
            sync_duration = end_time - start_time
            final_count = await db.products.count_documents({"user_id": str(user_id)})
            logger.info(f"Sync completed. Total products in database: {final_count}")
            return {
                "processed": processed_count,
                "failed": failed_count,
                "total": processed_count + failed_count,
                "duration_seconds": round(sync_duration, 2),
                "final_product_count": final_count
            }
        except Exception as e:
            logger.error(f"Error during sync: {str(e)}")
            raise

#########################################################
# Setup Endpoints
#########################################################
@app.post("/setup/user")
async def setup_user(user: UserBase):
    try:
        result = await MongoDB.db.users.insert_one(user.dict())
        return {"status": "success", "user_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setup/onboarding")
async def setup_onboarding(data: OnboardingData):
    try:
        shopify_service = ShopifyService(data.shopify_store, data.shopify_token)
        store_info = shopify_service.get_store_info()
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L12-v2')
        company_text = f"{data.general_info} {data.industry} {data.goal}"
        company_embedding = model.encode(company_text).tolist()

        onboarding_dict = data.dict()
        onboarding_dict["store_info"] = store_info
        onboarding_dict["embedding"] = company_embedding

        await MongoDB.db.onboarding.insert_one(onboarding_dict)
        sync_results = await shopify_service.sync_products(MongoDB.db, data.user_id)
        return {
            "status": "success",
            "store_info": store_info,
            "sync_results": sync_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync/{user_id}")
async def sync_products_endpoint(user_id: str):
    try:
        logger.info(f"Starting manual sync for user {user_id}")
        db = MongoDB.get_db()
        onboarding = await db.onboarding.find_one({"user_id": user_id})
        if not onboarding:
            raise HTTPException(status_code=404, detail="User onboarding data not found")
        await db.products.delete_many({"user_id": user_id})
        shopify_service = ShopifyService(onboarding["shopify_store"], onboarding["shopify_token"])
        sync_results = await shopify_service.sync_products(db, user_id)
        product_count = await db.products.count_documents({"user_id": user_id})
        return {
            "status": "success",
            "sync_results": sync_results,
            "product_count": product_count
        }
    except Exception as e:
        logger.error(f"Sync error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/{user_id}")
async def get_products(user_id: str):
    try:
        db = MongoDB.get_db()
        cursor = db.products.find({"user_id": user_id})
        products = await cursor.to_list(length=None)
        for product in products:
            if '_id' in product:
                product['_id'] = str(product['_id'])
        return products or []
    except Exception as e:
        logger.error(f"Error fetching products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

#########################################################
# Process Query: LLM Classification + Auto Category Mapping via GPT + Pinecone Filtering + Semantic Search
#########################################################
@measure_time
async def process_query(user_query: str, user_id: str) -> Tuple[str, List[Dict], str]:
    """Initial search with category mapping and filtering"""
    # 1) Classify the query using LLM
    classification = await classify_user_query(user_query)
    logger.info(f"Classification: {classification}")

    # Build Pinecone filter starting with user_id
    pinecone_filter = {"user_id": user_id}
    
    # Check available categories
    db = MongoDB.get_db()
    available_categories = await db.products.distinct("product_type", {"user_id": user_id})
    available_categories = [cat.strip().lower() for cat in available_categories if cat]

    # Try to map the category if classification provided one
    if classification.get("category"):
        mapped_category = await map_category_to_available(classification["category"], available_categories)
        if mapped_category:
            logger.info(f"Mapped category '{classification['category']}' to '{mapped_category}'")
            pinecone_filter["category"] = mapped_category
            classification["category"] = mapped_category
        else:
            logger.info(f"No category mapping found for '{classification['category']}'")
    
    # Generate embedding for the full user query (lowercase)
    query_embedding = get_embedding(user_query.lower())
    
    # Log search parameters
    logger.info(f"Initial search with filter: {pinecone_filter}")
    
    search_results = []
    try:
        # Search with category filter
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=10,
            filter=pinecone_filter,
            include_metadata=True
        )
        
        logger.info(f"Found {len(pine_res.get('matches', []))} matches with category filter")
        
        for match in pine_res.get("matches", []):
            metadata = match["metadata"]
            score = match["score"]
            
            search_results.append({
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "price": metadata.get("price", 0),
                "image": metadata.get("image", ""),
                "product_id": metadata.get("product_id", ""),
                "vendor": metadata.get("vendor", ""),
                "category": metadata.get("category", ""),
                "similarity": round(score, 3)
            })
            
        # Sort by similarity
        search_results = sorted(search_results, key=lambda x: x["similarity"], reverse=True)
        
    except Exception as e:
        logger.error(f"Pinecone query error: {str(e)}")

    # Save state for potential follow-up
    conversation_state[user_id] = {
        "filter": pinecone_filter,
        "embedding": query_embedding,
        "query": user_query,
        "classification": classification
    }

    if search_results:
        return (
            "Here are some products that match your request. Did you find what you were looking for? (yes/no)",
            search_results,
            "product_query"
        )
    else:
        # If no results with category filter, inform user we'll try a broader search
        return await process_deep_query(user_id)

async def process_deep_query(user_id: str) -> Tuple[str, List[Dict], str]:
    """Deep search across all categories when category mapping fails or user requests more results"""
    state = conversation_state.get(user_id)
    if not state:
        return ("I don't have your previous query. Could you please rephrase?", [], "need_more_info")

    # Use only user_id filter for deep search
    deep_filter = {"user_id": user_id}
    query_embedding = state.get("embedding")
    original_query = state.get("query", "")
    
    logger.info(f"Performing deep search for query '{original_query}' without category filter")
    
    search_results = []
    try:
        # Deep search across all categories
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=15,  # Increased for broader search
            filter=deep_filter,
            include_metadata=True
        )
        
        logger.info(f"Found {len(pine_res.get('matches', []))} matches in deep search")
        
        for match in pine_res.get("matches", []):
            metadata = match["metadata"]
            score = match["score"]
            
            search_results.append({
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "price": metadata.get("price", 0),
                "image": metadata.get("image", ""),
                "product_id": metadata.get("product_id", ""),
                "vendor": metadata.get("vendor", ""),
                "category": metadata.get("category", ""),
                "similarity": round(score, 3)
            })
            
        # Sort by similarity
        search_results = sorted(search_results, key=lambda x: x["similarity"], reverse=True)
        
    except Exception as e:
        logger.error(f"Deep search error: {str(e)}")

    # Clear conversation state
    conversation_state.pop(user_id, None)

    if search_results:
        return (
            "Here are more products I found across all categories. Let me know if you need anything else.",
            search_results,
            "deep_search"
        )
    else:
        return (
            "I couldn't find any products matching your search. Please try with different keywords.",
            [],
            "no_results"
        )
@measure_time
@app.post("/chat/{user_id}")
async def chat(user_id: str, query: ChatQuery, db: Any = Depends(MongoDB.get_db)):
    """Chat endpoint handling both initial and follow-up queries"""
    # Verify user
    user = await db.users.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_input = query.query.strip().lower()
    
    # Handle follow-up responses
    if user_input in ["yes", "no"] and user_id in conversation_state:
        if user_input == "yes":
            # User found what they wanted
            conversation_state.pop(user_id, None)
            return {"type": "confirmation", "message": "Great! Let me know if you need anything else."}
        else:
            # User wants more results - do deep search
            message, products, intent = await process_deep_query(user_id)
            return {
                "type": intent,
                "message": message,
                "has_products": len(products) > 0,
                "results": products
            }

    # Handle new query
    message, products, intent = await process_query(query.query, user_id)

    # Handle language translation if needed
    detected_lang = await detect_language_openai(query.query)
    if detected_lang != "en":
        lang_map = {"fr": "French", "es": "Spanish", "de": "German", "nl": "Dutch", "it": "Italian"}
        target_language = lang_map.get(detected_lang, "English")
        message = await translate_from_english(message, target_language)

    return {
        "type": intent,
        "message": message,
        "has_products": len(products) > 0,
        "results": products
    }

#########################################################
# Run the Application
#########################################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)