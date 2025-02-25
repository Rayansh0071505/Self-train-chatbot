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
    def get_store_info(self) -> Dict:
        """Fetch store info from Shopify."""
        try:
            shop = shopify.Shop.current()
            if not shop:
                raise ValueError("Could not retrieve shop information")
            return {
                'name': str(shop.name) if hasattr(shop, 'name') else '',
                'email': str(shop.email) if hasattr(shop, 'email') else '',
                'domain': str(shop.domain) if hasattr(shop, 'domain') else '',
                'country': str(shop.country) if hasattr(shop, 'country') else '',
                'currency': str(shop.currency) if hasattr(shop, 'currency') else '',
                'timezone': str(shop.timezone) if hasattr(shop, 'timezone') else ''
            }
        except Exception as e:
            logger.error(f"Error getting store info: {str(e)}")
            raise

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

async def process_show_more(user_id: str) -> Tuple[str, List[Dict], str]:
    """Process a request to show more products from the same category"""
    state = conversation_state.get(user_id)
    if not state:
        return ("I don't have your previous search. Could you please make a new search request?", [], "need_more_info")

    # Get the current filter and query information
    current_filter = state.get("filter", {"user_id": user_id})
    query_embedding = state.get("embedding")
    original_query = state.get("query", "")
    
    # Initialize last_results if it doesn't exist
    if "last_results" not in state:
        state["last_results"] = []
        
    last_results = state.get("last_results", [])
    current_page = state.get("page", 1)
    
    # Log what we're doing
    filter_desc = " and ".join([f"{k}: {v}" for k, v in current_filter.items() if k != "user_id"])
    logger.info(f"Showing more results for query '{original_query}' with filter: {filter_desc}, page {current_page+1}")
    logger.info(f"Excluding previously shown products: {last_results}")
    
    # Get category and brand info for better messaging
    category = current_filter.get("category", "")
    vendor = current_filter.get("vendor", "")
    
    search_results = []
    already_shown_ids = set(last_results)  # Convert to set for faster lookups
    
    try:
        # Request a larger number of results to account for filtering
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=50,  # Request more to ensure we have enough after filtering
            filter=current_filter,
            include_metadata=True
        )
        
        logger.info(f"Found {len(pine_res.get('matches', []))} total matches before filtering")
        
        # Filter out products we've already shown
        new_results = []
        for match in pine_res.get("matches", []):
            metadata = match["metadata"]
            product_id = metadata.get("product_id", "")
            
            # Only include if not in last_results
            if product_id and product_id not in already_shown_ids:
                score = match["score"]
                new_results.append({
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "price": metadata.get("price", 0),
                    "image": metadata.get("image", ""),
                    "product_id": product_id,
                    "vendor": metadata.get("vendor", ""),
                    "category": metadata.get("category", ""),
                    "similarity": round(score, 3)
                })
                
                # Keep track of what we've shown
                already_shown_ids.add(product_id)
                
                # Cap at 10 new results
                if len(new_results) >= 10:
                    break
        
        # Sort by similarity
        new_results = sorted(new_results, key=lambda x: x["similarity"], reverse=True)
        search_results = new_results
        
        logger.info(f"Found {len(search_results)} new matches for 'show more'")
        
    except Exception as e:
        logger.error(f"Show more query error: {str(e)}")

    # Update conversation state with new results
    if user_id in conversation_state:
        # Update the page number
        conversation_state[user_id]["page"] = current_page + 1
        
        # Add new product IDs to last_results
        conversation_state[user_id]["last_results"] = list(already_shown_ids)
        
        # Log the updated tracking
        logger.info(f"Updated last_results tracking: {conversation_state[user_id]['last_results']}")
    
    if search_results:
        # Construct appropriate message based on filter
        if vendor and category:
            message = f"Here are more {vendor} {category} products (page {current_page+1}). Did you find what you were looking for?"
        elif category:
            message = f"Here are more {category} products (page {current_page+1}). Did you find what you were looking for?"
        else:
            message = f"Here are more products that match your request (page {current_page+1}). Did you find what you were looking for?"
        
        return (message, search_results, "product_query")
    else:
        # No more results found
        if vendor and category:
            message = f"I don't have any more {vendor} {category} products to show you. Would you like to see products from other brands?"
        elif category:
            message = f"I don't have any more {category} products to show you. Would you like to see products from other categories?"
        else:
            message = "I don't have any more products that match your request. Would you like to try a different search?"
        
        return (message, [], "no_more_results")
    
async def detect_user_intent(query: str) -> Dict:
    """
    Uses LLM to detect the intent of the user query.
    Returns a dictionary with intent classification.
    """
    system_instructions = (
        "You are a user intent classifier. Parse the user's query into a JSON object with these keys:\n"
        "  - intent: string (one of: 'greeting', 'show_more', 'product_search', 'brand_selection', 'confirmation', 'help')\n"
        "  - confidence: float (between 0 and 1)\n"
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
            max_tokens=100
        )
        content = response.choices[0].message["content"].strip()
        intent_data = {}
        try:
            intent_data = json.loads(content)
            logger.info(f"Detected intent: {intent_data.get('intent', 'unknown')} with confidence: {intent_data.get('confidence', 0)}")
        except Exception as e:
            logger.warning(f"Could not parse JSON from LLM intent detection: {str(e)}")
        return intent_data
    except Exception as e:
        logger.error(f"Error detecting user intent: {str(e)}")
        return {"intent": "unknown", "confidence": 0.0}
    
@measure_time
async def process_query(user_query: str, user_id: str) -> Tuple[str, List[Dict], str]:
    """Initial search with category mapping and filtering and brand selection"""
    # First, detect user intent using LLM
    intent_data = await detect_user_intent(user_query)
    user_intent = intent_data.get("intent", "unknown")
    confidence = intent_data.get("confidence", 0.0)
    
    # Handle different intents
    if user_intent == "greeting" and confidence > 0.7:
        # Return a greeting response instead of product search
        return (
            f"Hello! How can I help you today? You can ask about products or search for specific items.",
            [],
            "greeting"
        )
    
    # Check if this is a "show more" request
    if user_intent == "show_more" and confidence > 0.7 and user_id in conversation_state:
        return await process_show_more(user_id)
    
    # Check if this is a brand selection from a previous query
    if user_id in conversation_state and conversation_state[user_id].get("awaiting_brand_selection", False):
        return await process_brand_selection(user_query, user_id)
    
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
    category_found = False
    if classification.get("category"):
        mapped_category = await map_category_to_available(classification["category"], available_categories)
        if mapped_category:
            logger.info(f"Mapped category '{classification['category']}' to '{mapped_category}'")
            pinecone_filter["category"] = mapped_category
            classification["category"] = mapped_category
            category_found = True
        else:
            logger.info(f"No category mapping found for '{classification['category']}'")
    
    # Handle brand selection if category is found but no brand is specified
    if category_found and not classification.get("brand"):
        # Get available brands/vendors for this category
        available_brands = await db.products.distinct(
            "vendor", 
            {"user_id": user_id, "product_type": classification["category"]}
        )
        available_brands = [brand.strip() for brand in available_brands if brand]
        
        if available_brands:
            # Save state for the brand selection follow-up
            conversation_state[user_id] = {
                "category": classification["category"],
                "query": user_query,
                "awaiting_brand_selection": True,
                "available_brands": available_brands
            }
            
            # Format brands as a numbered list
            brand_list = "\n".join([f"{i+1}. {brand}" for i, brand in enumerate(available_brands)])
            
            return (
                f"For {classification['category']}, we have the following brands available. Please select one:\n\n{brand_list}",
                [],
                "brand_selection"
            )
    
    # If we have both category and brand, add brand to filter
    if classification.get("brand"):
        pinecone_filter["vendor"] = classification["brand"]
    
    # Generate embedding for the full user query (lowercase)
    query_embedding = get_embedding(user_query.lower())
    
    # Log search parameters
    logger.info(f"Initial search with filter: {pinecone_filter}")
    
    search_results = []
    try:
        # Search with category and potentially brand filter
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=10,
            filter=pinecone_filter,
            include_metadata=True
        )
        
        logger.info(f"Found {len(pine_res.get('matches', []))} matches with filter")
        
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

    # Save state for potential follow-up (but not overwriting brand selection state)
    if not conversation_state.get(user_id, {}).get("awaiting_brand_selection", False):
        # Save all product IDs we've shown to the user
        shown_product_ids = [r["product_id"] for r in search_results if "product_id" in r]
        
        conversation_state[user_id] = {
            "filter": pinecone_filter,
            "embedding": query_embedding,
            "query": user_query,
            "classification": classification,
            "last_results": shown_product_ids,  # Track shown product IDs
            "page": 1
        }
        
        # Log the tracked products
        logger.info(f"Tracking {len(shown_product_ids)} shown products: {shown_product_ids}")

    if search_results:
        return (
            "Here are some products that match your request. Did you find what you were looking for? (yes/no) You can also ask to see more products if you need additional options.",
            search_results,
            "product_query"
        )
    else:
        # If no results with category filter, inform user we'll try a broader search
        return await process_deep_query(user_id)

async def process_brand_selection(user_input: str, user_id: str) -> Tuple[str, List[Dict], str]:
    """Process a brand selection from the user"""
    state = conversation_state.get(user_id, {})
    if not state.get("awaiting_brand_selection"):
        return ("I don't understand. Could you please rephrase your question?", [], "need_more_info")
    
    available_brands = state.get("available_brands", [])
    category = state.get("category", "")
    original_query = state.get("query", "")
    
    # Try to match the user input to a brand
    selected_brand = None
    
    # Check if user entered a number
    try:
        selection_idx = int(user_input.strip()) - 1
        if 0 <= selection_idx < len(available_brands):
            selected_brand = available_brands[selection_idx]
    except ValueError:
        # Not a number, try to match by name
        user_input_lower = user_input.lower().strip()
        
        # Exact match
        for brand in available_brands:
            if brand.lower() == user_input_lower:
                selected_brand = brand
                break
        
        # Partial match if no exact match found
        if not selected_brand:
            for brand in available_brands:
                if user_input_lower in brand.lower():
                    selected_brand = brand
                    break
    
    if not selected_brand:
        # Could not determine brand, ask again
        brand_list = "\n".join([f"{i+1}. {brand}" for i, brand in enumerate(available_brands)])
        return (
            f"I couldn't match that to any of our available brands. Please try again by selecting a number or typing the brand name:\n\n{brand_list}",
            [],
            "brand_selection"
        )
    
    # Brand identified, now search with this brand
    logger.info(f"Selected brand: {selected_brand} for category: {category}")
    
    # Build filter with user_id, category and brand
    pinecone_filter = {
        "user_id": user_id,
        "category": category,
        "vendor": selected_brand
    }
    
    # Generate embedding for the original query
    query_embedding = get_embedding(original_query.lower())
    
    search_results = []
    try:
        # Search with category and brand filter
        pine_res = pinecone_index.query(
            vector=query_embedding,
            top_k=10,
            filter=pinecone_filter,
            include_metadata=True
        )
        
        logger.info(f"Found {len(pine_res.get('matches', []))} matches with category and brand filter")
        
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
    
    # Reset state
    conversation_state[user_id] = {
        "filter": pinecone_filter,
        "embedding": query_embedding,
        "query": original_query
    }
    
    if search_results:
        return (
            f"Here are {selected_brand} {category} products that match your request. Did you find what you were looking for? (yes/no)",
            search_results,
            "product_query"
        )
    else:
        # No results with brand filter, try just category
        return (
            f"I couldn't find any {selected_brand} {category} products. Let me show you all {category} products instead.",
            [],
            "no_brand_results"
        )

async def process_deep_query(user_id: str) -> Tuple[str, List[Dict], str]:
    """Deep search across all categories when category mapping fails or user requests more results"""
    state = conversation_state.get(user_id)
    if not state:
        return ("I don't have your previous query. Could you please rephrase?", [], "need_more_info")

    # Skip deep query if awaiting brand selection
    if state.get("awaiting_brand_selection", False):
        brand_list = "\n".join([f"{i+1}. {brand}" for i, brand in enumerate(state.get("available_brands", []))])
        return (
            f"Please select a brand from the list:\n\n{brand_list}",
            [],
            "brand_selection"
        )

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
    if not state.get("awaiting_brand_selection", False):
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

    user_input = query.query.strip()
    
    # First detect intent for yes/no responses to avoid unnecessary LLM calls
    if user_input.lower() in ["yes", "no", "y", "n"] and user_id in conversation_state and not conversation_state[user_id].get("awaiting_brand_selection", False):
        if user_input.lower() in ["yes", "y"]:
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

    # For other inputs, use the LLM to determine intent
    intent_data = await detect_user_intent(user_input)
    user_intent = intent_data.get("intent", "unknown")
    confidence = intent_data.get("confidence", 0.0)
    
    # Handle show more intent
    if user_intent == "show_more" and confidence > 0.7 and user_id in conversation_state:
        message, products, intent = await process_show_more(user_id)
        return {
            "type": intent,
            "message": message,
            "has_products": len(products) > 0,
            "results": products
        }

    # Handle new query or brand selection
    message, products, intent = await process_query(query.query, user_id)

    # Handle no brand results
    if intent == "no_brand_results":
        # Try to show all products in the category without brand filter
        category = conversation_state.get(user_id, {}).get("filter", {}).get("category")
        if category:
            db = MongoDB.get_db()
            pinecone_filter = {"user_id": user_id, "category": category}
            query_embedding = conversation_state[user_id].get("embedding")
            
            try:
                pine_res = pinecone_index.query(
                    vector=query_embedding,
                    top_k=10,
                    filter=pinecone_filter,
                    include_metadata=True
                )
                
                for match in pine_res.get("matches", []):
                    metadata = match["metadata"]
                    score = match["score"]
                    
                    products.append({
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
                products = sorted(products, key=lambda x: x["similarity"], reverse=True)
                
                # Update conversation state with these results
                conversation_state[user_id] = {
                    "filter": pinecone_filter,
                    "embedding": query_embedding,
                    "query": conversation_state[user_id].get("query", ""),
                    "last_results": [p["product_id"] for p in products],
                    "page": 1
                }
                
                intent = "product_query"
                message = f"Here are {category} products that match your request. Did you find what you were looking for? (yes/no) You can also ask to see more products if you need additional options."
                
            except Exception as e:
                logger.error(f"Fallback query error: {str(e)}")

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
        "results": products if products else []
    }
#########################################################
# Run the Application
#########################################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)