# app/main.py
from fastapi import FastAPI
from app.routers.match_job_description import router as match_job_description_router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables from the .env file
load_dotenv()


# List of origins that are allowed to access the resources
origins = [
    "http://localhost:4200",  # Angular app running locally
    "https://your-frontend-domain.com",  # Replace with your frontend domain
]

# Add CORSMiddleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI"}

# Include the routers
app.include_router(match_job_description_router, prefix="/api")
