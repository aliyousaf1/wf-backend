from email.message import EmailMessage
import random
import re
import secrets
import smtplib
import time
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
import requests
import urllib
from app.core.database import db
from app.models.user_model import UserModel
from jose import jwt
import os
from app.core.auth_utils import hash_password, verify_password
from app.models.auth_user_model import AuthUserModel
from app.models.notification_settings_model import NotificationSettingsModel
from fastapi.responses import RedirectResponse

onboarding_route = APIRouter(prefix="/onboarding", tags=["User Onboarding"])

brandsCollection = db["brands"]
sizeChartCollection = db["size_chart"]
usersCollection = db["users"]
authUserCollection = db["auth_user"]
notificationsCollection = db["notifications"]

def serializeItem(item):
    item["_id"] = str(item["_id"])
    return item

APPLE_PUBLIC_KEYS_URL = "https://appleid.apple.com/auth/keys"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = "https://app.whyfashion.com/onboarding/auth/google/callback"

@onboarding_route.post("/auth/apple")
async def auth_apple(request: Request):
    data = await request.json()
    token = data.get("token")

    if not token:
        raise HTTPException(status_code=400, detail="Token missing")

    # Fetch Apple public keys
    res = requests.get(APPLE_PUBLIC_KEYS_URL)
    keys = res.json()["keys"]

    # Decode header to select correct key
    header = jwt.get_unverified_header(token)
    key = next((k for k in keys if k["kid"] == header.get("kid")), None)

    if not key:
        raise HTTPException(status_code=400, detail="Invalid key")

    try:
        # Step 1 — decode WITHOUT audience verification
        decoded = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer="https://appleid.apple.com",
            options={"verify_aud": False}  # IMPORTANT
        )

        # Step 2 — manually verify audience
        allowed_auds = ["com.whyfashion.app", "com.whyfashion.auth"]
        aud = decoded.get("aud")

        if aud not in allowed_auds:
            raise HTTPException(status_code=401, detail="Invalid audience")

        # Extract user ID + email
        user_id = decoded["sub"]
        email = decoded.get("email")

        return {"user_id": user_id, "email": email}

    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@onboarding_route.api_route("/auth/apple/callback", methods=["GET", "POST"])
@onboarding_route.api_route("/auth/apple/callback/", methods=["GET", "POST"])
async def apple_callback(request: Request):
    """
    Handles Apple's redirect after authentication.
    Works for both GET and POST (with form data), for iOS & Android.
    """

    # 1️⃣ Get query params from URL
    params = dict(request.query_params)

    # 2️⃣ If POST, read form data (Apple sends POST form)
    if request.method == "POST":
        form = await request.form()  # <-- now works because python-multipart is installed
        params.update(form)

    # 3️⃣ Extract tokens
    id_token = params.get("id_token")
    code = params.get("code")
    state = params.get("state")

    if not id_token and not code:
        raise HTTPException(
            status_code=400,
            detail="Authorization failed: Missing id_token or code."
        )

    # 4️⃣ Build query string for deep linking
    query_params = urllib.parse.urlencode({
        "id_token": id_token or "",
        "code": code or "",
        "state": state or "",
    })

    deep_link_scheme = os.getenv("APP_DEEP_LINK_SCHEME")
    android_package = os.getenv("ANDROID_PACKAGE_NAME")

    # 5️⃣ iOS deep link
    deep_link_url = f"{deep_link_scheme}://callback?{query_params}"

    # 6️⃣ Android intent URL
    android_intent_url = (
        f"intent://callback?{query_params}"
        f"#Intent;"
        f"scheme={deep_link_scheme};"      # Your app’s scheme
        f"package={android_package};"      # Your app’s package
        f"end"
    )


    # 7️⃣ Detect Android via user-agent
    user_agent = request.headers.get("User-Agent", "").lower()
    if "android" in user_agent:
        return RedirectResponse(url=android_intent_url)

    # Default → iOS
    return RedirectResponse(url=deep_link_url)

@onboarding_route.post("/auth/google")
def google_login():
    google_auth_endpoint = "https://accounts.google.com/o/oauth2/v2/auth"
    scope = "openid email profile"
    response_type = "code"

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": response_type,
        "scope": scope,
        "access_type": "offline",
        "prompt": "consent"
    }

    url = f"{google_auth_endpoint}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url=url)
    
@onboarding_route.post("/auth/google/callback")
def google_callback(request: Request, code: str):
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    response = requests.post(token_url, data=data)
    tokens = response.json()

    # Optional: verify ID token and fetch user info
    user_info = requests.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    ).json()

    return JSONResponse(content={"user": user_info})

'''
    Creates a user document in the database
'''
@onboarding_route.post("/create_user")
async def createUser(user : UserModel):
    user_dict = user.model_dump()
    result = await usersCollection.insert_one(user_dict)

    if not result.inserted_id:
        raise HTTPException(500, "Couldn't create user")
    
    return {
       "status": "SUCCESS",
       "data": {
            "_id": str(result.inserted_id), 
            "data": user.model_dump()
        }
    }
'''
    Store manual sign up creds
'''
@onboarding_route.post("/manual_signup")
async def manualSignUp(authModel: AuthUserModel):
    # Password validation
    if not re.match(r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*(),.?":{}|<>]).{8,}$', authModel.password):
        return {
            "status": "FAILED",
            "message": "INVALID_PASSWORD"
        }
    # Check if user already exists
    existing_user = await authUserCollection.find_one({"email": authModel.email})
    anotherSignInMethod = await usersCollection.find_one({"email": authModel.email})
    if existing_user or anotherSignInMethod:
        return {
            "status": "FAILED",
            "message": "USER_ALREADY_EXISTS"
        }

    # Hash password
    hashed_pw = hash_password(authModel.password)
    
    # Generate token
    email_verification_token = generate_email_verification_token()
    
    # Insert user
    result = await authUserCollection.insert_one({
        "email": authModel.email,
        "password": hashed_pw,
        "email_verified": False,
        "verification_token": email_verification_token,
        "token_used": False,
        "token_expiry": int(time.time()) + 15 * 60  # expires in 15 minutes
    })

    if not result.inserted_id:
        raise HTTPException(status_code=500, detail="Couldn't create user")
    user = UserModel(email=authModel.email, login_method="MANUAL" )
    result = await usersCollection.insert_one(user.dict())

    # Generate verification link (local for now)
    verification_link = f"https://app.whyfashion.com/onboarding/verify-email?email={authModel.email}&token={email_verification_token}"

    # Send templated email
    send_email(
        authModel.email,
        "Welcome to Why Fashion — Please Confirm Your Email",
        generate_verification_email_html(verification_link)
    )

    return {
        "status": "SUCCESS",
        "message": "USER_CREATED",
        "detail": "Verification email sent successfully."
    }

@onboarding_route.get("/verify-email")
async def verifyEmail(request: Request):
    email = request.query_params.get('email')
    token = request.query_params.get('token')

    if not email or not token:
        raise HTTPException(status_code=400, detail="Email or token is missing.")

    result = await authUserCollection.find_one({
        "email": email,
        "verification_token": token
    })

    if not result:
        raise HTTPException(status_code=400, detail="Invalid verification link.")

    if result.get("token_used"):
        raise HTTPException(status_code=400, detail="This verification link has already been used.")

    expiry_ts = result.get("token_expiry")
    current_ts = int(time.time())

    if not expiry_ts or current_ts > expiry_ts:
        raise HTTPException(status_code=400, detail="Verification link has expired.")

    await authUserCollection.update_one(
        {"email": email},
        {"$set": {"email_verified": True, "token_used": True}}
    )

    redirect_url = f"whyfashionapp://login?email={email}&verified=true"
    return RedirectResponse(url=redirect_url)

@onboarding_route.post("/resend_verification_email")
async def resend_verification_email(email: str):
    # Check if user exists
    user = await authUserCollection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # If already verified
    if user.get("email_verified", False):
        return {
            "status": "FAILED",
            "message": "EMAIL_ALREADY_VERIFIED"
        }

    # Generate new verification token and expiry
    new_token = generate_email_verification_token()
    new_expiry = int(time.time()) + 15 * 60  # 15 minutes

    # Update in database
    await authUserCollection.update_one(
        {"email": email},
        {"$set": {
            "verification_token": new_token,
            "token_used": False,
            "token_expiry": new_expiry
        }}
    )

    # Generate verification link (local or production)
    verification_link = f"https://app.whyfashion.com/onboarding/verify-email?email={email}&token={new_token}"

    # Send email using your existing templated email function
    send_email(
        email,
        "Why Fashion — Verify Your Email",
        generate_verification_email_html(verification_link)
    )

    return {
        "status": "SUCCESS",
        "message": "VERIFICATION_EMAIL_RESENT",
        "detail": "A new verification email has been sent successfully."
    }


@onboarding_route.post("/login")
async def login(authModel: AuthUserModel):
    # Check auth credentials
    result = await authUserCollection.find_one({"email": authModel.email})

    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(authModel.password, result["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Add FCM token to user model
    fcm_token = authModel.fcm_token  # single token sent from client

    if fcm_token:
        await usersCollection.update_one(
            {"email": authModel.email},
            {"$addToSet": {"fcm_tokens": fcm_token}}  # $addToSet prevents duplicates
        )

    return {
        "status": "SUCCESS",
        "message": "Login successful",
        "email_verified": result.get("email_verified", False)
    }


def send_email(to_email: str, subject: str, html_content: str):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = "support@whyfashion.com"
    msg['To'] = to_email

    msg.set_content("Please view this email in an HTML-compatible email client.")
    msg.add_alternative(html_content, subtype='html')
    SMTP_HOST = os.getenv("SMTP_HOST")
    SMTP_PORT = int(os.getenv("SMTP_PORT"))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")
    
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
    print("Verification email sent successfully!")

@onboarding_route.post("/send-email")
async def email_route(to_email: str, subject: str, body: str):
    try:
        send_email(to_email, subject, body + "\n"+generate_email_verification_token())
        return {"status": "SUCCESS", "message": "Email sent!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
Updates user information
'''
@onboarding_route.patch("/update_user")
async def updateUser(email: str, data: dict):
    updateData = {k: v for k,v in data.items() if v is not None}
    result = await usersCollection.update_one({"email": email}, {"$set": updateData})

    if not result.modified_count:
        raise HTTPException(500, "Couldn't update user")
    
    return {
       "status": "SUCCESS"
    }

'''
Fetches and sends all the information of a user
'''

@onboarding_route.get("/user/{email}")
async def getUserDetails(email: str):
    userData = await usersCollection.find_one({"email": email})
    
    if not userData:
        raise HTTPException(
            status_code=404,
            detail="No matching user record was found."
        )

    # Normalize gender so client always receives an **array** in a single key `gender`
    # New users: may already have `genders` array field
    # Old users: only have single `gender` string → convert to array under `gender`
    serialized = serializeItem(userData)

    genders = serialized.get("genders")
    if isinstance(genders, list) and genders:
        normalized_gender_array = [str(g).strip() for g in genders if g is not None]
    else:
        single_gender = serialized.get("gender")
        if single_gender:
            normalized_gender_array = [str(single_gender).strip()]
        else:
            normalized_gender_array = []

    # Use single key `gender` for the array
    serialized["gender"] = normalized_gender_array
    # Remove legacy `genders` from response to avoid confusion
    serialized.pop("genders", None)

    return serialized

'''
Fetches and sends list of available brands in our app
'''
@onboarding_route.get("/brands")
async def getAllBrands():
    brandsList = await brandsCollection.find().to_list(length=None)
    brandsList = [serializeItem(brand) for brand in brandsList]
    print(brandsList)
    return brandsList

'''
Fetches and sends list of all available sizes in our app
'''
@onboarding_route.get("/size_chart")
async def getSizeChart():
    sizeList = await sizeChartCollection.find().to_list(length=None)
    sizeList[0].pop("_id")
    return sizeList[0]

@onboarding_route.post("/forgot_password")
async def forgot_password(request: dict):
    email = request.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    # Check if user exists
    user = await authUserCollection.find_one({"email": email})
    if not user:
        return {"status": "FAILED", "message": "USER_NOT_FOUND"}

    # Generate 6-digit OTP
    otp_code = str(random.randint(100000, 999999))

    # Save OTP with expiry (e.g., 10 minutes)
    await authUserCollection.update_one(
        {"email": email},
        {"$set": {
            "reset_code": otp_code,
            "reset_code_expiry": int(time.time()) + 10 * 60  # 10 min expiry
        }}
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    templatePath = os.path.join(base_dir, "../email_templates/forgot_password_template.html")
    # Read HTML email template
    with open(templatePath, "r") as f:
        html_template = f.read()

    # Replace placeholder
    html_content = html_template.replace("{{OTP_CODE}}", otp_code)

    # Send email
    send_email(
        email,
        "Password Reset Code — Why Fashion",
        html_content
    )

    return {
        "status": "SUCCESS",
        "message": "OTP_SENT",
        "detail": "Password reset code sent successfully to your email."
    }

@onboarding_route.post("/verify_reset_code")
async def verify_reset_code(request: dict):
    email = request.get("email")
    code = request.get("code")
    newPassword = request.get("newPassword")

    if not email or not code or not newPassword:
        raise HTTPException(status_code=400, detail="Email, code, and password are required")

    user = await authUserCollection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Validate OTP and expiry
    if user.get("reset_code") != code:
        return {"status": "FAILED", "message": "INVALID_CODE"}
    if int(time.time()) > user.get("reset_code_expiry", 0):
        return {"status": "FAILED", "message": "CODE_EXPIRED"}
        # Password validation
    if not re.match(r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*(),.?":{}|<>]).{8,}$', newPassword):
        return {
            "status": "FAILED",
            "message": "INVALID_PASSWORD"
        }
      # Hash new password
    hashed_pw = hash_password(newPassword)

    # Update password and remove OTP fields
    await authUserCollection.update_one(
        {"email": email},
        {"$set": {"password": hashed_pw},
         "$unset": {"reset_code": "", "reset_code_expiry": ""}}
    )
    return {"status": "SUCCESS", "message": "PASSWORD_UPDATED"}

@onboarding_route.delete("/delete_account")
async def deleteAccount(request: dict):
    email = request.get("email")
    await authUserCollection.delete_one({
        "email": email
    })
    await usersCollection.delete_one({
        "email": email
    })
    return {
        "status": "SUCCESS",
        "message": "ACCOUNT_DELETED"
    }

def generate_email_verification_token(token_bytes: int = 48):
    token = secrets.token_urlsafe(token_bytes)
    return token

def generate_verification_email_html(verify_link: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(base_dir, "../email_templates/verify_email_template.html")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    return template.replace("{{VERIFY_EMAIL_LINK}}", verify_link)

@onboarding_route.post("/set_notification_settings")
async def setNotificationSettings(notificationSettingsModel: NotificationSettingsModel):
    # Check auth credentials
    result = await usersCollection.find_one({"email": notificationSettingsModel.email})

    if not result:
        raise HTTPException(status_code=404, detail="User not found")

    
    prefs = notificationSettingsModel.model_dump()
    prefs.pop("email", None)

    await usersCollection.update_one(
        {"email": notificationSettingsModel.email},
        {"$set": {"notification_preferences": prefs}}
    )

    return {
        "status": "SUCCESS",
        "message": "Preferences saved",
    }

from datetime import datetime, timezone

@onboarding_route.get("/get_notifications")
async def getNotificationSettings(email: str):
    userDetails = await usersCollection.find_one({"email": email})

    if not userDetails:
        raise HTTPException(status_code=404, detail="User not found")

    creation_epoch = userDetails.get("created_at", 0)
    userCreatedDate = datetime.fromtimestamp(creation_epoch, timezone.utc)

    favorite_brands = userDetails.get("favorite_brands", [])
    prefs = userDetails.get("notification_preferences", {})

    if not prefs.get("appNotificationsMain"):
        return {
            "status": "SUCCESS",
            "notifications": [],
            "message": "TURNED_OFF_SETTINGS"
        }

    notifications_cursor = notificationsCollection.find({
        "createdAt": {"$gt": userCreatedDate},
        "$or": [
            {"brands": {"$size": 0}},
            {"brands": {"$in": favorite_brands}},
        ]
    })

    notifications = await notifications_cursor.to_list(length=None)

    # Convert ALL ObjectIds safely
    notifications = [convert_objectids_and_dates(n) for n in notifications]

    return {
        "status": "SUCCESS",
        "notifications": notifications,
        "message": "NOTIFICATIONS_FETCHED_SUCCESSFULLY",
    }

from datetime import datetime, timezone
from bson import ObjectId


def convert_objectids_and_dates(obj):
    """
    Recursively converts ObjectId to str and datetime to ISO-8601 string.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_obj[k] = convert_objectids_and_dates(v)
        return new_obj

    elif isinstance(obj, list):
        return [convert_objectids_and_dates(i) for i in obj]

    elif isinstance(obj, ObjectId):
        return str(obj)

    elif isinstance(obj, datetime):
        # Ensure UTC + ISO format
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.isoformat()

    else:
        return obj

@onboarding_route.get("/update_last_seen_announcements")
async def update_last_seen_announcements(email: str):
    userDetails = await usersCollection.find_one({"email": email})

    if not userDetails:
        raise HTTPException(status_code=404, detail="User not found")
    
    current_epoch = int(datetime.utcnow().timestamp())

    await usersCollection.update_one(
        {"email": email},
        {"$set": {"last_seen_announcements_at": current_epoch}}
    )

    return {
        "status": "SUCCESS",
        "message": "LAST_SEEN_UPDATED",
    }