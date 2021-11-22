from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/predict")
async def predict(
    format: str = Query(
        "json",
        regex=r"^txt|json$",
        description="The format in which the output will be returned",
    )
):
    return {"message": "Hello World", "format": format}

@app.get("/report")
async def report(
    format: str = Query(
        "json",
        regex=r"^html|txt|json$",
        description="The format in which the output will be returned",
    )
):
    return {"message": "Hello World", "format": format}
