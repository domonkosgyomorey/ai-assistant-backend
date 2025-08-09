import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the global logging level
    format="%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d]\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# Create a logger instance
logger = logging.getLogger("ingestion")
logger.setLevel(logging.INFO)  # Ensure the logger level is set to INFO
