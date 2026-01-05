from typing import List
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, Request
from app.core.database import db
import random
import json
from openai import OpenAI
from app.models.product_model import ProductModel
import requests
import base64
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from bson import ObjectId
from fastapi import Body

feed_route = APIRouter(prefix="/feed", tags=["Products Feed"])

brandsCollection = db["brands"]
productsCollection = db["products_scrapped"]
usersCollection = db["users"]



def serializeItem(item):
    item["_id"] = str(item["_id"])
    return item


'''
Fetches and sends list of available brands in our app
'''
@feed_route.get("/products/random")
async def getRandomProducts(count: int):
    # ✅ Use MongoDB aggregation for efficient random sampling
    products_cursor = productsCollection.aggregate([{"$sample": {"size": count}}])
    products = [serializeItem(p) async for p in products_cursor]
    return products


# --- STEP 1: Prefilter products (optimized) ---
def prefilter_products(products, user):
    viewed_ids = {p.get("_id", "") for p in user.get("productsViewed", [])}
    dislike_ids = {p.get("_id", "") for p in user.get("dislikes", [])}
    liked_ids = {p.get("_id", "") for p in user.get("likes", [])}
    user_gender = user.get("gender", "").lower()
    return [
        p for p in products
        if p.get("gender", "").lower() == user_gender
        and p.get("_id", "") not in viewed_ids
        and p.get("_id", "") not in dislike_ids
        and p.get("_id", "") not in liked_ids
    ]


def prefilter_products_guest(products, dislikes, productsViewed, gender):
    viewed_ids = {p["_id"] for p in productsViewed}
    dislike_ids = {p["_id"] for p in dislikes}
    user_gender = gender.lower()

    return [
        p for p in products
        if p.get("gender", "").lower() == user_gender
        and p["_id"] not in viewed_ids
        and p["_id"] not in dislike_ids
    ]


# --- STEP 2: Extract keywords from likes ---
def extract_keywords(items):
    words = []
    for item in items:
        for h in item.get("highlights", []):
            words.extend(re.findall(r"[a-zA-Z]+", h.lower()))
        words.extend(re.findall(r"[a-zA-Z]+", item.get("title", "").lower()))
    return Counter(words)


# --- STEP 3: Score a single product ---
def score_product(product, keyword_counter):
    score = 0
    for h in product.get("highlights", []):
        for word in re.findall(r"[a-zA-Z]+", h.lower()):
            score += keyword_counter[word] * 2
    for word in re.findall(r"[a-zA-Z]+", product.get("title", "").lower()):
        score += keyword_counter[word]
    return score


# --- STEP 4: Mix in randomness ---
def inject_randomness(scored_products, accessories, interval=2):
    """
    Insert random accessories (perfumes, bags, belts, ties, etc.)
    after every `interval` products.
    """
    final = []
    for i, (prod, _) in enumerate(scored_products):
        final.append(prod)
        if (i + 1) % interval == 0 and accessories:
            final.append(random.choice(accessories))
    return final

def sanitize_product(item: dict) -> dict:
    """
    Removes non-JSON-safe or unwanted fields from product objects.
    Works even if fields are missing.
    """
    item.pop("created_at", None)
    item.pop("updated_at", None)
    item.pop("createdAt", None)
    item.pop("updatedAt", None)
    return item

def safe_obj_ids(items):
    """
    Safely extract ObjectIds from a list of dicts.
    """
    return [
        ObjectId(p["_id"])
        for p in items
        if isinstance(p, dict) and p.get("_id")
    ]


@feed_route.post("/get_recommendations")
async def get_recommendations(data: dict = Body(...)):
    email = data.get("email", "")
    gender = data.get("gender", "").strip().lower()

    # 🔹 Fetch user
    user = await usersCollection.find_one({"email": email})
    if not user:
        return []

    user = sanitize_product(serializeItem(user))

    # 🔹 Collect excluded product IDs
    excluded_ids = list({
        *safe_obj_ids(user.get("productsViewed", [])),
        *safe_obj_ids(user.get("dislikes", [])),
        *safe_obj_ids(user.get("likes", [])),
    })

    # 🔹 Strict gender filter
    query = {
        "gender": gender,
        "_id": {"$nin": excluded_ids},
    }

    # 🔹 Fetch candidate products
    candidates_cursor = productsCollection.find(query)
    candidates = [
        sanitize_product(serializeItem(p))
        async for p in candidates_cursor
    ]

    if not candidates:
        return []

    # 🔹 Load accessories once
    accessories_cursor = productsCollection.find({
        "gender": gender,
        "highlights": {
            "$elemMatch": {
                "$regex": "(bag|perfume|belt|watch|tie|fragrance|accessory)",
                "$options": "i",
            }
        }
    })

    accessories = [
        sanitize_product(serializeItem(p))
        async for p in accessories_cursor
    ]

    # 🔹 Scored recommendations
    if user.get("likes"):
        keywords = extract_keywords(user["likes"])

        scored_products = [
            (product, score_product(product, keywords))
            for product in candidates
        ]

        scored_products.sort(key=lambda x: x[1], reverse=True)

        recommendations = inject_randomness(
            scored_products,
            accessories,
            interval=4
        )

        # 🔹 Deduplicate by _id
        seen_ids = set()
        unique_recommendations = []

        for product in recommendations:
            pid = str(product.get("_id"))
            if pid and pid not in seen_ids:
                unique_recommendations.append(product)
                seen_ids.add(pid)

        return unique_recommendations[:10]

    # 🔹 Random fallback (no likes)
    unique_candidates = list(
        {str(p["_id"]): p for p in candidates}.values()
    )

    return random.sample(
        unique_candidates,
        min(10, len(unique_candidates))
    )


# --- Guest Recommendations (Fixed) ---
@feed_route.post("/get_recommendations_guest")
async def get_recommendations_guest(data: dict = Body(...)):
    likes = data.get("likes", [])
    dislikes = data.get("dislikes", [])
    products_viewed = data.get("productsViewed", [])
    gender = data.get("gender", "").strip().lower()

    # 🔹 Exclude already seen / disliked products
    excluded_ids = [
        ObjectId(p["_id"])
        for p in products_viewed + dislikes
        if p.get("_id")
    ]

    # 🔹 Case-insensitive gender filter
    query = {
        "gender": {"$regex": f"^{gender}$", "$options": "i"},
        "_id": {"$nin": excluded_ids},
    }

    # 🔹 Fetch candidate products
    candidates_cursor = productsCollection.find(query)
    candidates = [
        sanitize_product(serializeItem(p))
        async for p in candidates_cursor
    ]

    if not candidates:
        return []

    # 🔹 Preload accessories
    accessories_cursor = productsCollection.find({
        "highlights": {
            "$elemMatch": {
                "$regex": "(bag|perfume|belt|watch|tie|fragrance|accessory)",
                "$options": "i",
            }
        }
    })

    accessories = [
        sanitize_product(serializeItem(p))
        async for p in accessories_cursor
    ]

    # 🔹 Recommendation logic
    if likes:
        keywords = extract_keywords(likes)

        scored_products = [
            (product, score_product(product, keywords))
            for product in candidates
        ]

        scored_products.sort(key=lambda x: x[1], reverse=True)

        recommendations = inject_randomness(
            scored_products,
            accessories,
            interval=2
        )

        return recommendations[:10]

    # 🔹 Fallback: random recommendations
    return random.sample(candidates, min(10, len(candidates)))


# @feed_route.get("/get_recommendations")
# async def get_recommendations(email :str):
#     productsList = await productsCollection.find().to_list(length=None)
#     userData = await usersCollection.find_one({"email": email})
#     serialUserData = serializeItem(userData)
#     serialProductsData = [serializeItem(item) for item in productsList]
#     prompt = f"""
#         You are given user data: {json.dumps(serialUserData)} 
#         and product data: {json.dumps(serialProductsData)}.

#         Your task:
#         1. Recommend exactly 4 products strictly based on the user's likes and dislikes. 
#         - Only use product features (title, composition, highlights) to measure relevance.
#         - Do NOT select random products if likes or dislikes exist.
#         2. If likes and dislikes are both empty:
#         - Return exactly 10 random unique product _id values.
#         3. Never include any product that appears in productsViewed from the user data.
#         4. The output must be a valid JSON array containing only the product _id values, 
#         with no extra text, no explanations, no new lines.

#         Output format (strictly):
#         [123, 456, 789, ...]
#     """

#     open_ai_client = OpenAI(
# api_key="YOUR_API_KEY_HERE"
#     )
#     try:
#         print(serialProductsData)
#         print("Generating recommendation...")
#         response = open_ai_client.chat.completions.create(
#             model="gpt-4.1-mini",
#             messages=[
#                 {
#                     "role": "system", 
#                     "content": (
#                         "You are a personalized fashion recommendation assistant. "
#                         "Your goal is to suggest products that closely match the user's preferences. "
#                         "Follow these rules strictly:"
#                         "\n1. Do not recommend any product already present in `productsViewed`."
#                         "\n2. Only recommend products that match the user's likes and avoid their dislikes."
#                         "\n3. Recommendations should be relevant and not random."
#                         "\n4. Suggest a maximum of 5 items per request."
#                         "\n5. Provide recommendations as a clean JSON array with product IDs only."
#                     )
#                 },
#                 {"role": "user", "content": prompt},
#                 {"role": "system", "content": "Always output your answer as valid JSON only. No extra text."}

#             ],
#             max_tokens=300,
#             temperature=1.0,   # controls randomness (0 = deterministic, 2 = very random)
#             top_p=1.0,         # nucleus sampling
#             presence_penalty=0.6,  # encourage diversity
#             frequency_penalty=0.5  # avoid repeating same IDs
#         )

#         filteredRecommendations = response.choices[0].message.content.strip().strip("\"[\\").strip("\"]\\").replace("\"", "").replace("\\", "").split(",")
#         responseProducts = []
#         for i in serialProductsData:
#             for j in filteredRecommendations:
#                 if(j == i["_id"]):
#                     responseProducts.append(i)
#                     break
#         return responseProducts
#     except Exception as e:
#         print("Couldn't generate recommendation")
#         print(e)

async def like_product_db(email: str, product: dict) -> bool:
    result = await usersCollection.update_one(
        {"email": email},
        {"$addToSet": {"likes": product}, "$pull": {"dislikes": product}}
    )
    return bool(result.modified_count)

async def dislike_product_db(email: str, product: dict) -> bool:
    result = await usersCollection.update_one(
        {"email": email},
        {"$addToSet": {"dislikes": product}, "$pull": {"likes": product}}
    )
    return bool(result.modified_count)

async def add_to_watched_db(email: str, product: dict) -> bool:
    result = await usersCollection.update_one(
        {"email": email},
        {"$addToSet": {"productsViewed": product}}
    )
    return bool(result.modified_count)

@feed_route.post("/add_to_likes")
async def likeProduct(email: str, product: dict):
    if not await like_product_db(email, product):
        raise HTTPException(500, "Couldn't like product")
    return {"message": "SUCCESS"}

@feed_route.post("/add_to_dislikes")
async def dislikeProduct(email: str, product: dict):
    if not await dislike_product_db(email, product):
        raise HTTPException(500, "Couldn't dislike product")
    return {"message": "SUCCESS"}

@feed_route.post("/add_to_watched")
async def addToWatched(email: str, product: dict):
    if not await add_to_watched_db(email, product):
        raise HTTPException(500, "Couldn't view product")
    return {"message": "SUCCESS"}

async def get_recommendations_with_genders(data: dict):
    """
    Helper function to handle recommendations with genders array.
    If genders is an array, fetches recommendations for each gender and combines them.
    """
    request_data = data.get("data", {})
    genders = request_data.get("genders", [])
    
    # Normalize genders to list
    if isinstance(genders, str):
        genders = [genders]
    elif not isinstance(genders, list):
        genders = []
    
    # If no genders array, fallback to single gender or fetch from user
    if not genders:
        single_gender = request_data.get("gender", "")
        if single_gender:
            genders = [single_gender]
        else:
            # Fetch user to get genders
            email = request_data.get("email", "")
            if email:
                user = await usersCollection.find_one({"email": email})
                if user:
                    user = serializeItem(user)
                    user_genders = user.get("genders", [])
                    if isinstance(user_genders, list) and user_genders:
                        genders = user_genders
                    elif user_genders:
                        genders = [user_genders]
                    else:
                        # Fallback to old gender field
                        old_gender = user.get("gender", "")
                        if old_gender:
                            genders = [old_gender]
    
    if not genders:
        return []
    
    # Normalize genders to lowercase strings
    genders = [g.strip().lower() if isinstance(g, str) else str(g).lower() for g in genders if g]
    
    if not genders:
        return []
    
    # If single gender, use existing function
    if len(genders) == 1:
        modified_data = request_data.copy()
        modified_data["gender"] = genders[0]
        return await get_recommendations(modified_data)
    
    # Multiple genders: fetch recommendations for each and combine
    all_recommendations = []
    seen_ids = set()
    
    for gender in genders:
        modified_data = request_data.copy()
        modified_data["gender"] = gender
        recommendations = await get_recommendations(modified_data)
        
        for product in recommendations:
            product_id = str(product.get("_id", ""))
            if product_id and product_id not in seen_ids:
                all_recommendations.append(product)
                seen_ids.add(product_id)
    
    # Return up to 10 recommendations
    return all_recommendations[:10]


async def get_recommendations_guest_with_genders(data: dict):
    """
    Helper function to handle guest recommendations with genders array.
    """
    request_data = data.get("data", {})
    genders = request_data.get("genders", [])
    
    # Normalize genders to list
    if isinstance(genders, str):
        genders = [genders]
    elif not isinstance(genders, list):
        genders = []
    
    # Fallback to single gender
    if not genders:
        single_gender = request_data.get("gender", "")
        if single_gender:
            genders = [single_gender]
    
    if not genders:
        return []
    
    # Normalize genders to lowercase strings
    genders = [g.strip().lower() if isinstance(g, str) else str(g).lower() for g in genders if g]
    
    if not genders:
        return []
    
    # If single gender, use existing function
    if len(genders) == 1:
        modified_data = request_data.copy()
        modified_data["gender"] = genders[0]
        return await get_recommendations_guest(modified_data)
    
    # Multiple genders: fetch recommendations for each and combine
    all_recommendations = []
    seen_ids = set()
    
    for gender in genders:
        modified_data = request_data.copy()
        modified_data["gender"] = gender
        recommendations = await get_recommendations_guest(modified_data)
        
        for product in recommendations:
            product_id = str(product.get("_id", ""))
            if product_id and product_id not in seen_ids:
                all_recommendations.append(product)
                seen_ids.add(product_id)
    
    # Return up to 10 recommendations
    return all_recommendations[:10]


'''
Get Feed API
'''
@feed_route.websocket("/get_feed/")
async def feed_socket(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_text("Connected to socket")
        while True:
            data = await websocket.receive_json()
            print(data)

            if data["req_type"] == "LIKE":
                if await like_product_db(data["email"], data["product"]):
                    await websocket.send_json({"message": "SUCCESS"})
                else:
                    await websocket.send_json({"error": "Couldn't like product"})

            elif data["req_type"] == "DISLIKE":
                if await dislike_product_db(data["email"], data["product"]):
                    await websocket.send_json({"message": "SUCCESS"})
                else:
                    await websocket.send_json({"error": "Couldn't dislike product"})

            elif data["req_type"] == "WATCHED":
                if await add_to_watched_db(data["email"], data["product"]):
                    await websocket.send_json({"message": "SUCCESS"})
                else:
                    await websocket.send_json({"error": "Couldn't view product"})

            elif data["req_type"] == "GET_RECOMMENDATIONS":
                response = await get_recommendations_with_genders(data)
                if len(response) == 0:
                    response = await get_recommendations_with_genders(data)
                await websocket.send_json(response)
            elif data["req_type"] == "GET_RECOMMENDATIONS_GUEST":
                response = await get_recommendations_guest_with_genders(data)
                if len(response) == 0:
                    response = await get_recommendations_guest_with_genders(data)
                await websocket.send_json(response)

            elif data["req_type"] == "PING":
                await websocket.send_json({"message": "Hey there from server"})

    except WebSocketDisconnect:
        print("Client disconnected")

@feed_route.get("/get_liked_products")
async def getLikedProducts(email :str):
    userData = await usersCollection.find_one({"email": email})
    userData = serializeItem(userData)
    return userData["likes"]


def dict_to_string(data):
    if isinstance(data, dict):
        return " ".join([f"{k} {v}" for k, v in data.items() if v])
    elif isinstance(data, list):
        return " | ".join([
            " ".join([f"{k} {v}" for k, v in d.items() if v])
            for d in data
        ])
    return str(data)


def simple_filter_products(query: str, products: list, top_k: int = 50):
    """
    Looser filtering:
    - Finds candidate products based on partial and fuzzy keyword matches.
    - Keeps top_k matches instead of excluding too many.
    """

    keywords = re.findall(r"\w+", query.lower())
    scored_products = []

    for product in products:
        text = f"{product.get('title', '')} {product.get('composition', '')} {product.get('highlights', '')}".lower()

        score = 0
        for kw in keywords:
            if kw in text:
                score += 2  # strong boost for exact keyword
            else:
                # fuzzy match (similarity > 0.7 considered relevant)
                for word in text.split():
                    if SequenceMatcher(None, kw, word).ratio() > 0.7:
                        score += 1

        if score > 0:
            scored_products.append((score, product))

    # Sort by score (best first) and keep top_k
    scored_products.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored_products[:top_k]]

@feed_route.get("/search_by_text")
async def searchByText(gender: str, searchQuery: str = Query(..., min_length=1)):
    # Convert incoming params to lowercase
    gender = gender.lower()
    searchQuery = searchQuery.lower()

    # Normalize plurals
    normalize_map = {
        "sneakers": "sneaker",
        "shoes": "shoe",
        "loafers": "loafer",
        "sandals": "sandal",
        "boots": "boot"
    }
    keywords = [normalize_map.get(kw, kw) for kw in searchQuery.split()]

    allowed_highlight_fields = [
        "category", "cloth_type", "colors",
        "material", "length", "fit", "style"
    ]

    # Build keyword conditions
    keyword_conditions = []
    for kw in keywords:
        regex = {"$regex": kw, "$options": "i"}

        keyword_conditions.append({
            "$or": [
                {"title": regex},
                {"brand": regex},
                {"description": regex},
                *[
                    {
                        "highlights": {
                            "$elemMatch": {
                                "$regex": f"^{field}:.*{kw}.*$",
                                "$options": "i"
                            }
                        }
                    }
                    for field in allowed_highlight_fields
                ]
            ]
        })

    # Build final Mongo query
    query = {
        "$and": [
            {"gender": {"$regex": f"^{gender}$", "$options": "i"}},
            *keyword_conditions
        ]
    }

    # Use aggregation to lowercase fields before comparison
    pipeline = [
        {
            "$addFields": {
                "title": {"$toLower": "$title"},
                "brand": {"$toLower": "$brand"},
                "description": {"$toLower": "$description"},
                "gender": {"$toLower": "$gender"},
                "highlights": {
                    "$map": {
                        "input": "$highlights",
                        "as": "h",
                        "in": {"$toLower": "$$h"}
                    }
                }
            }
        },
        {"$match": query}
    ]

    cursor = productsCollection.aggregate(pipeline)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)

    return results


@feed_route.post("/search_by_image")
async def searchByImage(body: Request):
    reqBody = await body.json()
    imageAnalysis = await getProductImageDetails(imageUrl=reqBody["imageUrl"])

    # Convert imageAnalysis to a string
    if isinstance(imageAnalysis, dict):
        analysis_str = dict_to_string(imageAnalysis)
    elif isinstance(imageAnalysis, list):
        analysis_str = " | ".join([dict_to_string(item) for item in imageAnalysis])
    else:
        analysis_str = str(imageAnalysis)

    # Fetch all products
    productsList = await productsCollection.find().to_list(length=None)
    serialProductsData = [serializeItem(item) for item in productsList]

    # Pre-filter products
    candidateProducts = simple_filter_products(analysis_str, serialProductsData, top_k=50)

    if not candidateProducts:
        return []

    open_ai_client = OpenAI(
        api_key="YOUR_API_KEY_HERE"
    )

    try:
        response = open_ai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a product search engine. "
                        "Your task is to match detected items from an image with product data. "
                        "Always return only valid JSON with product IDs."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
                    product_data: {json.dumps(candidateProducts)}

                    Task:
                    - Match products for each detected item in this description: {analysis_str}
                    - Use only title, composition, and highlights for matching
                    - gender, category, colors, style_notes, wear_type must strictly match
                    - If nothing matches for an item, skip it
                    - Final output must be a JSON array of unique product IDs only

                    Example output:
                    ["123", "456", "789"]
                    """,
                },
            ],
            max_tokens=300,
            temperature=0.2,
        )

        raw_output = response.choices[0].message.content.strip()
        print("🔎 GPT raw output:", raw_output)

        try:
            filteredResults = json.loads(raw_output)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\[.*\]", raw_output, re.DOTALL)
            if match:
                filteredResults = json.loads(match.group(0))
            else:
                print("❌ Could not parse GPT output")
                return []

        responseProducts = [p for p in candidateProducts if p["_id"] in filteredResults]

        return responseProducts

    except Exception as e:
        print("Couldn't generate results:", e)
        return []

@feed_route.get("/get_product_image_details")
async def getProductImageDetails(imageUrl: str):
    try:
        img_data = requests.get(imageUrl, timeout=20).content
        b64_image = base64.b64encode(img_data).decode("utf-8")
        client = OpenAI(
            api_key="YOUR_API_KEY_HERE"
        )
        response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fashion assistant. Analyze the clothing in the given image and respond ONLY in JSON."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this product image and return a JSON with keys: "
                                    "title, brand, gender, category (e.g. shirt, pants, dress, perfume, watch, etc), cloth_type (e.g. casual, formal), wear_type (e.g. top, bottom, outerwear, underwear)"
                                    "colors (list), patterns (if any), and extra_details."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                }
                                
                            }
                        ]
                    }
                ],
                temperature=0.2,
                response_format={ "type": "json_object" }
            )

            # Extract JSON string
        result = response.choices[0].message.content.strip().strip("\"[\\").strip("\"]\\").replace("\"", "").replace("\\", "").replace("\n", "").strip().replace("{", "").replace("}", "").split(",")
        result = [r.strip() for r in result]
        return {"analysis": result}

    except Exception as e:
        return {"error": str(e)}
    