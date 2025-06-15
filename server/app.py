import uvicorn
from fastapi import FastAPI
from utils.logger import logger

app = FastAPI()


@app.get("/")
async def asd():
    logger.debug("Hello World!")
    return {"messages": "Hello, World!"}


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")
