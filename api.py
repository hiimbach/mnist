import os 

from fastapi import FastAPI, File, UploadFile   

from utils.classifier import Classifier

# Model 
classifier = Classifier()

# API
app = FastAPI()

@app.post("/uploadfile/")
async def predict(file: UploadFile):
    # ipdb.set_trace()
    # return {"filename": file.filename}
    
    save_path = os.path.join('data/api', file.filename)
    content = await file.read()
    
    # Save file
    with open(save_path, "wb") as f:
        f.write(content)
        
    pred = classifier.predict(save_path)
    return {"number": pred}