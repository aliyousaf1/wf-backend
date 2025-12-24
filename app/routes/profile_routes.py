from fastapi import APIRouter, HTTPException
from app.core.database import db
from app.models.user_model import UserModel
profile_route = APIRouter(prefix="/profile", tags=["User Profile"])

usersCollection = db["users"]

def serializeItem(item):
    item["_id"] = str(item["_id"])
    return item

'''
    Creates a user document in the database
'''

'''
Updates user information
'''
@profile_route.patch("/update_user")
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
@profile_route.get("/user/{email}")
async def getUserDetails(email):
    userData = await usersCollection.find_one({"email": email})
    if not userData:
        return None
    userData = serializeItem(userData)
    return {
        "first_name": userData["first_name"],
        "last_name": userData["last_name"],
        "dob": userData["dob"]
    }

'''
Deletes a user document by email
'''
@profile_route.delete("/user/{email}")
async def deleteUser(email):
    deleted_user = await usersCollection.delete_one({"email": email})
    
    if deleted_user.deleted_count == 0:
        return {"message": "User not found"}
    
    return {"message": f"User with email {email} deleted successfully"}
