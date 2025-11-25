from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
import tempfile
import os
import shutil

app = FastAPI(title="Face Recognition API")

@app.post("/api/face-recognition")
async def recognize_face(image: UploadFile = File(...)):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    
    try:
        shutil.copyfileobj(image.file, temp_file)
        temp_file.close()
        
        people = DeepFace.find(
            img_path=temp_file.name,
            db_path="./database/",
            model_name="Facenet512",
            detector_backend="retinaface",
            distance_metric="euclidean"
        )
        
        return {"result": people[0].to_dict('records') if len(people) > 0 and not people[0].empty else []}
        
    finally:
        os.unlink(temp_file.name)
        await image.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8882)