import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from supabase import create_client, Client
from storage3.utils import StorageException
import mimetypes
from io import BytesIO
import uuid

import open_clip
import torch
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


# Get Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")


# Initialize Supabase client
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL and Key must be set in the .env file")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

# Configuration
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# Load the clip model
model, preprocess = open_clip.create_model_from_pretrained("ViT-B-32", pretrained="openai")
model.to(DEVICE)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Embedding dimension for CLIP ViT-B-32
EMBEDDING_DIMENSION = 512

def get_embedding(image: Image.Image) -> list[float]:
    """Generate normalized embedding for an image."""
    image = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].tolist()


async def store_embedding(filename: str, file_path: str, public_url: str, embedding: list[float]):
    """Store image metadata and embedding in Supabase."""
    response = supabase.table("image_embeddings").insert({
        "filename": filename,
        "file_path": file_path,
        "public_url": public_url,
        "embedding": embedding
    }).execute()
    return response


async def search_similar_images(query_embedding: list[float], match_threshold: float = 0.5, match_count: int = 5):
    """Search for similar images using pgvector similarity search."""
    response = supabase.rpc("match_images", {
        "query_embedding": query_embedding,
        "match_threshold": match_threshold,
        "match_count": match_count
    }).execute()
    return response.data

@app.get("/")
def home():
    return {"message": "FastAPI with reload is working!"}

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}

@app.post("/add-image/")
async def add_image(file: UploadFile = File(...)):
    try:
        # Determine the content type (MIME type)
        content_type = file.content_type or mimetypes.guess_type(file.filename)[0]

        # Validate content type
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Only PNG and JPEG images are allowed. Got: {content_type}"
            )

        # Read the file content asynchronously
        file_content = await file.read()

        # Generate a unique filename to avoid duplicates
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = f"uploads/{unique_filename}"

        # Upload the file to Supabase Storage
        try:
            supabase.storage.from_(BUCKET_NAME).upload(
                file=file_content,
                path=file_path,
                file_options={"content-type": content_type}
            )
        except StorageException as e:
            if "Duplicate" in str(e) or "409" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"File already exists: {file.filename}"
                )
            raise

        # Get the public URL for the uploaded file
        # Note: Bucket must be public, or use create_signed_url for private buckets
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)
        # For private buckets, use signed URL instead (expires in 1 hour):
        # signed_url = supabase.storage.from_(BUCKET_NAME).create_signed_url(file_path, 3600)

        # After successful Supabase upload, process the image for embedding
        img = Image.open(BytesIO(file_content)).convert("RGB")
        embedding = get_embedding(img)

        # Store embedding in Supabase pgvector
        await store_embedding(
            filename=file.filename,
            file_path=file_path,
            public_url=public_url,
            embedding=embedding
        )

        return {
            "status": "stored",
            "message": f"Successfully uploaded {file.filename} to Supabase storage.",
            "file_path": file_path,
            "public_url": public_url
        }

    except HTTPException:
        raise
    except Exception as e:
        # Catch other potential errors like network issues or file system errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during file upload: {str(e)}"
        )
    finally:
        # Ensure the uploaded file (spooled temporary file) is closed
        await file.close()

# Similarity threshold for cosine similarity (0-1, higher = more similar)
SIMILARITY_THRESHOLD = 0.5

@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    try:
        # Read the file content asynchronously
        file_content = await file.read()

        # Process the image for embedding search
        img = Image.open(BytesIO(file_content)).convert("RGB")
        query_embedding = get_embedding(img)

        # Search using pgvector
        results = await search_similar_images(
            query_embedding=query_embedding,
            match_threshold=SIMILARITY_THRESHOLD,
            match_count=5
        )

        if not results:
            return {
                "results": [],
                "message": "No similar images found. The uploaded image doesn't match any stored images."
            }

        # Format results with similarity score as percentage
        formatted_results = []
        for result in results:
            formatted_results.append({
                "filename": result["filename"],
                "file_path": result["file_path"],
                "public_url": result["public_url"],
                "similarity_score": round(result["similarity"] * 100, 2)
            })

        return {"results": formatted_results}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during search: {str(e)}"
        )
    finally:
        await file.close()

