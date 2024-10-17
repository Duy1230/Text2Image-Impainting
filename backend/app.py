from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import sam2_route

app = FastAPI()

app = FastAPI(
    title="Chatbot API",
    description="A simple chatbot API",
    version="1.0.0",
)

origins = [
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
)

# include the route
app.include_router(sam2_route, prefix="/sam2")
