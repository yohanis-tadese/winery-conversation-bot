import logging

# Create a reusable logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)  # default level
formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Optional: file handler
# file_handler = logging.FileHandler("app.log")
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# Exported logger for other modules
__all__ = ["logger", "settings", "pg_pool", "init_milvus", "get_milvus_collection"]
