import os
import sys

import uvicorn

sys.path.append(os.path.join(os.getcwd(), "server"))
from api.endpoints import app

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8081, log_level="info")
