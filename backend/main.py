from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.sam2_route import router as sam2_route

app = FastAPI(
    title="Chatbot API",
    description="A simple chatbot API",
    version="1.0.0"
)

origins = [
    # "http://localhost:3000",  # Add your frontend's origin here
    # If your frontend is served via ngrok
    # "https://8781-34-125-235-224.ngrok-free.app",
    # Add other origins if necessary
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# include the route
app.include_router(sam2_route, prefix="/sam2")
