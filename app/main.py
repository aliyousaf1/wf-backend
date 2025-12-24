from fastapi import FastAPI
from app.routes.ping_route import pingRouter
from app.routes.onboading_routes import onboarding_route
from app.routes.feed_routes import feed_route
from app.routes.profile_routes import profile_route
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="WhyFashion : Dev-01")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (not safe for prod, better restrict)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pingRouter)
app.include_router(onboarding_route)
app.include_router(feed_route)
app.include_router(profile_route)