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
# Global conversation state to track follow-up queries
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
        # Create indexes
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
    prompt = (f"Given the input category '{input_cat}', and the available categories: {', '.join(available)}, "
              "choose the best matching available category. Return only the matching category in lowercase.")
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant for mapping product categories automatically."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=20,
        )
        mapped = response.choices[0].message["content"].strip().lower()
        if mapped in available:
            return mapped
        else:
            # Fallback using difflib
            candidates = difflib.get_close_matches(input_cat, available, n=1, cutoff=0.0)
            return candidates[0] if candidates else None
    except Exception as e:
        logger.error(f"Error mapping category: {str(e)}")
        return None

#########################################################
# Interpret Follow-up Response using LLM
#########################################################
async def interpret_followup(user_input: str) -> str:
    prompt = (f"Determine if the following response expresses a positive confirmation, a negative response (not satisfied), "
              f"a request to show more recommendations, or a vendor inquiry. "
              f"Respond with one word: 'confirmation', 'deep_search', 'show_more', or 'vendor_query'.\n"
              f"Response: {user_input}")
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        intent = response.choices[0].message["content"].strip().lower()
        if intent in ["confirmation", "deep_search", "show_more", "vendor_query"]:
            return intent
        else:
            return "unknown"
    except Exception as e:
        logger.error(f"Error interpreting followup: {str(e)}")
        return "unknown"

#########################################################
# Process Deep Query (when follow-up indicates negative)
#########################################################
@measure_time
async def process_deep_query(user_id: str) -> Tuple[str, List[Dict], str]:
    state = conversation_state.get(user_id)
    if not state:
        return ("I don't have your previous query. Could you please rephrase?", [], "need_more_info")
    # Remove category filter for deep search
    deep_filter = {"user_id": user_id}
    query_embedding = state.get("embedding")
    search_results = []
    try:
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            filter=deep_filter,
            include_metadata=True
        )
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
    except Exception as e:
        logger.error(f"Deep search error: {str(e)}")
    conversation_state.pop(user_id, None)
    if search_results:
        return (
            "Let me do a deep analysis of your query. Here are additional products:",
            search_results,
            "deep_search"
        )
    else:
        return (
            "I still couldn't find any products matching your query. Please try refining your search further.",
            [],
            "no_results"
        )

#########################################################
# Process Show More Recommendations
#########################################################
@measure_time
async def process_show_more(user_id: str) -> Tuple[str, List[Dict], str]:
    state = conversation_state.get(user_id)
    if not state:
        return ("I don't have your previous query. Could you please rephrase?", [], "need_more_info")
    filter_ = state.get("filter", {"user_id": user_id})
    query_embedding = state.get("embedding")
    shown_ids = state.get("shown_ids", [])
    search_results = []
    try:
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=10,
            filter=filter_,
            include_metadata=True
        )
        for match in pine_res.get("matches", []):
            metadata = match["metadata"]
            pid = metadata.get("product_id", "")
            if pid not in shown_ids:
                search_results.append({
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "price": metadata.get("price", 0),
                    "image": metadata.get("image", ""),
                    "product_id": pid,
                    "vendor": metadata.get("vendor", ""),
                    "category": metadata.get("category", ""),
                    "similarity": round(match["score"], 3)
                })
                shown_ids.append(pid)
                if len(search_results) >= 5:
                    break
        state["shown_ids"] = shown_ids
    except Exception as e:
        logger.error(f"Show more query error: {str(e)}")
    if search_results:
        return (
            "Here are more products for your query:",
            search_results,
            "show_more"
        )
    else:
        return (
            "No additional products found.",
            [],
            "no_more"
        )

#########################################################
# Process Vendor Query: Retrieve Distinct Vendors for the Category
#########################################################
async def process_vendor_query(user_id: str) -> Tuple[str, List[Dict], str]:
    db = MongoDB.get_db()
    state = conversation_state.get(user_id, {})
    category = state.get("filter", {}).get("category", None)
    if not category:
        return ("Please specify a product category first.", [], "need_more_info")
    vendors = await db.products.distinct("vendor", {"user_id": user_id, "product_type": category})
    vendors = [v for v in vendors if v]
    if vendors:
        vendor_list = [{"vendor": v} for v in vendors]
        return (
            "These are the available vendors for the product category:",
            vendor_list,
            "vendor_query"
        )
    else:
        return ("No vendors found for the given product category.", [], "no_results")

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
        sync_results = await shopify_service.sync_products(MongoDB.db, user_id)
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
    classification = await classify_user_query(user_query)
    logger.info(f"Classification: {classification}")

    if ("buy" in user_query.lower() or "purchase" in user_query.lower()) and not classification.get("direct_product", False):
        classification["direct_product"] = True

    if not classification.get("category"):
        return (
            "Could you please provide more details about the product you're interested in? (e.g., specify category or brand)",
            [],
            "need_more_info"
        )

    db = MongoDB.get_db()
    available_categories = await db.products.distinct("product_type", {"user_id": user_id})
    available_categories = [cat.strip().lower() for cat in available_categories if cat]

    cat = classification.get("category")
    if cat not in available_categories:
        mapped_cat = await map_category_to_available(cat, available_categories)
        if mapped_cat:
            logger.info(f"Mapped category '{cat}' to '{mapped_cat}' using GPT")
            classification["category"] = mapped_cat
        else:
            logger.info("No suitable mapping found; removing category filter for deep search.")
            classification.pop("category", None)

    pinecone_filter = {"user_id": user_id}
    if classification.get("category"):
        pinecone_filter["category"] = classification["category"].strip()
    if classification.get("brand"):
        pinecone_filter["vendor"] = classification["brand"].strip()

    conversation_state[user_id] = {
        "filter": pinecone_filter,
        "embedding": get_embedding(user_query),
        "query": user_query,
        "shown_ids": []
    }

    query_embedding = get_embedding(user_query)
    search_results = []
    try:
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            filter=pinecone_filter,
            include_metadata=True
        )
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
            conversation_state[user_id].setdefault("shown_ids", []).append(metadata.get("product_id", ""))
    except Exception as e:
        logger.error(f"Pinecone query error: {str(e)}")

    if search_results:
        return (
            "Here are some products that match your request. Please let me know if these meet your needs.",
            search_results,
            "product_query"
        )
    else:
        return (
            "I couldn't find any products matching that query. Please try to refine your search with more details.",
            [],
            "no_results"
        )

#########################################################
# Process Deep Query (when follow-up indicates negative)
#########################################################
@measure_time
async def process_deep_query(user_id: str) -> Tuple[str, List[Dict], str]:
    state = conversation_state.get(user_id)
    if not state:
        return ("I don't have your previous query. Could you please rephrase?", [], "need_more_info")
    deep_filter = {"user_id": user_id}
    query_embedding = state.get("embedding")
    search_results = []
    try:
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            filter=deep_filter,
            include_metadata=True
        )
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
    except Exception as e:
        logger.error(f"Deep search error: {str(e)}")
    conversation_state.pop(user_id, None)
    if search_results:
        return (
            "Let me do a deep analysis of your query. Here are additional products:",
            search_results,
            "deep_search"
        )
    else:
        return (
            "I still couldn't find any products matching your query. Please try refining your search further.",
            [],
            "no_results"
        )

#########################################################
# Process Show More Recommendations
#########################################################
@measure_time
async def process_show_more(user_id: str) -> Tuple[str, List[Dict], str]:
    state = conversation_state.get(user_id)
    if not state:
        return ("I don't have your previous query. Could you please rephrase?", [], "need_more_info")
    filter_ = state.get("filter", {"user_id": user_id})
    query_embedding = state.get("embedding")
    shown_ids = state.get("shown_ids", [])
    search_results = []
    try:
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=10,
            filter=filter_,
            include_metadata=True
        )
        for match in pine_res.get("matches", []):
            metadata = match["metadata"]
            pid = metadata.get("product_id", "")
            if pid not in shown_ids:
                search_results.append({
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "price": metadata.get("price", 0),
                    "image": metadata.get("image", ""),
                    "product_id": pid,
                    "vendor": metadata.get("vendor", ""),
                    "category": metadata.get("category", ""),
                    "similarity": round(match["score"], 3)
                })
                shown_ids.append(pid)
                if len(search_results) >= 5:
                    break
        state["shown_ids"] = shown_ids
    except Exception as e:
        logger.error(f"Show more query error: {str(e)}")
    if search_results:
        return (
            "Here are more products for your query:",
            search_results,
            "show_more"
        )
    else:
        return (
            "No additional products found.",
            [],
            "no_more"
        )

#########################################################
# Process Vendor Query: Retrieve Distinct Vendors for the Category
#########################################################
async def process_vendor_query(user_id: str) -> Tuple[str, List[Dict], str]:
    db = MongoDB.get_db()
    state = conversation_state.get(user_id, {})
    category = state.get("filter", {}).get("category", None)
    if not category:
        return ("Please specify a product category first.", [], "need_more_info")
    vendors = await db.products.distinct("vendor", {"user_id": user_id, "product_type": category})
    vendors = [v for v in vendors if v]
    if vendors:
        vendor_list = [{"vendor": v} for v in vendors]
        return (
            "These are the available vendors for the product category:",
            vendor_list,
            "vendor_query"
        )
    else:
        return ("No vendors found for the given product category.", [], "no_results")

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
        sync_results = await shopify_service.sync_products(MongoDB.db, user_id)
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
    classification = await classify_user_query(user_query)
    logger.info(f"Classification: {classification}")

    if ("buy" in user_query.lower() or "purchase" in user_query.lower()) and not classification.get("direct_product", False):
        classification["direct_product"] = True

    if not classification.get("category"):
        return (
            "Could you please provide more details about the product you're interested in? (e.g., specify category or brand)",
            [],
            "need_more_info"
        )

    db = MongoDB.get_db()
    available_categories = await db.products.distinct("product_type", {"user_id": user_id})
    available_categories = [cat.strip().lower() for cat in available_categories if cat]

    cat = classification.get("category")
    if cat not in available_categories:
        mapped_cat = await map_category_to_available(cat, available_categories)
        if mapped_cat:
            logger.info(f"Mapped category '{cat}' to '{mapped_cat}' using GPT")
            classification["category"] = mapped_cat
        else:
            logger.info("No suitable mapping found; removing category filter for deep search.")
            classification.pop("category", None)

    pinecone_filter = {"user_id": user_id}
    if classification.get("category"):
        pinecone_filter["category"] = classification["category"].strip()
    if classification.get("brand"):
        pinecone_filter["vendor"] = classification["brand"].strip()

    conversation_state[user_id] = {
        "filter": pinecone_filter,
        "embedding": get_embedding(user_query),
        "query": user_query,
        "shown_ids": []
    }

    query_embedding = get_embedding(user_query)
    search_results = []
    try:
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            filter=pinecone_filter,
            include_metadata=True
        )
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
            conversation_state[user_id].setdefault("shown_ids", []).append(metadata.get("product_id", ""))
    except Exception as e:
        logger.error(f"Pinecone query error: {str(e)}")

    if search_results:
        return (
            "Here are some products that match your request. Please let me know if these meet your needs.",
            search_results,
            "product_query"
        )
    else:
        return (
            "I couldn't find any products matching that query. Please try to refine your search with more details.",
            [],
            "no_results"
        )

#########################################################
# Process Deep Query (when follow-up indicates negative)
#########################################################
@measure_time
async def process_deep_query(user_id: str) -> Tuple[str, List[Dict], str]:
    state = conversation_state.get(user_id)
    if not state:
        return ("I don't have your previous query. Could you please rephrase?", [], "need_more_info")
    deep_filter = {"user_id": user_id}  # Remove category filter
    query_embedding = state.get("embedding")
    search_results = []
    try:
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            filter=deep_filter,
            include_metadata=True
        )
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
    except Exception as e:
        logger.error(f"Deep search error: {str(e)}")
    conversation_state.pop(user_id, None)
    if search_results:
        return (
            "Let me do a deep analysis of your query. Here are additional products:",
            search_results,
            "deep_search"
        )
    else:
        return (
            "I still couldn't find any products matching your query. Please try refining your search further.",
            [],
            "no_results"
        )

#########################################################
# Process Show More Recommendations
#########################################################
@measure_time
async def process_show_more(user_id: str) -> Tuple[str, List[Dict], str]:
    state = conversation_state.get(user_id)
    if not state:
        return ("I don't have your previous query. Could you please rephrase?", [], "need_more_info")
    filter_ = state.get("filter", {"user_id": user_id})
    query_embedding = state.get("embedding")
    shown_ids = state.get("shown_ids", [])
    search_results = []
    try:
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=10,
            filter=filter_,
            include_metadata=True
        )
        for match in pine_res.get("matches", []):
            metadata = match["metadata"]
            pid = metadata.get("product_id", "")
            if pid not in shown_ids:
                search_results.append({
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "price": metadata.get("price", 0),
                    "image": metadata.get("image", ""),
                    "product_id": pid,
                    "vendor": metadata.get("vendor", ""),
                    "category": metadata.get("category", ""),
                    "similarity": round(match["score"], 3)
                })
                shown_ids.append(pid)
                if len(search_results) >= 5:
                    break
        state["shown_ids"] = shown_ids
    except Exception as e:
        logger.error(f"Show more query error: {str(e)}")
    if search_results:
        return (
            "Here are more products for your query:",
            search_results,
            "show_more"
        )
    else:
        return (
            "No additional products found.",
            [],
            "no_more"
        )

#########################################################
# Process Vendor Query: Retrieve Distinct Vendors for the Category
#########################################################
async def process_vendor_query(user_id: str) -> Tuple[str, List[Dict], str]:
    db = MongoDB.get_db()
    state = conversation_state.get(user_id, {})
    category = state.get("filter", {}).get("category", None)
    if not category:
        return ("Please specify a product category first.", [], "need_more_info")
    vendors = await db.products.distinct("vendor", {"user_id": user_id, "product_type": category})
    vendors = [v for v in vendors if v]
    if vendors:
        vendor_list = [{"vendor": v} for v in vendors]
        return (
            "These are the available vendors for the product category:",
            vendor_list,
            "vendor_query"
        )
    else:
        return ("No vendors found for the given product category.", [], "no_results")

#########################################################
# Chat Endpoint
#########################################################
@app.post("/chat/{user_id}")
async def chat(user_id: str, query: ChatQuery, db: Any = Depends(MongoDB.get_db)):
    user = await db.users.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_input = query.query.strip().lower()

    # Interpret follow-up using LLM if previous state exists
    if user_id in conversation_state:
        followup_intent = await interpret_followup(user_input)
        if followup_intent == "confirmation":
            conversation_state.pop(user_id, None)
            return {"type": "confirmation", "message": "Great, I'm glad you found what you were looking for!"}
        elif followup_intent == "deep_search":
            message, products, intent = await process_deep_query(user_id)
            return {"type": intent, "message": message, "has_products": len(products) > 0, "results": products}
        elif followup_intent == "show_more":
            message, products, intent = await process_show_more(user_id)
            return {"type": intent, "message": message, "has_products": len(products) > 0, "results": products}
        elif followup_intent == "vendor_query":
            message, vendors, intent = await process_vendor_query(user_id)
            return {"type": intent, "message": message, "has_products": len(vendors) > 0, "results": vendors}
        else:
            conversation_state.pop(user_id, None)
            # Process as new query if follow-up intent is unknown

    message, products, intent = await process_query(query.query, user_id)
    detected_lang = await detect_language_openai(query.query)
    lang_map = {"en": "English", "fr": "French", "nl": "Dutch", "de": "German", "es": "Spanish", "it": "Italian"}
    if detected_lang != "en":
        target_language = lang_map.get(detected_lang, "English")
        message = await translate_from_english(message, target_language)
    response_data = {
        "type": intent,
        "message": message,
        "has_products": len(products) > 0,
        "results": products
    }
    return response_data

#########################################################
# Run the Application
#########################################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
