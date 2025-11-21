import logging
import os

class Logger:
    def __init__(self, log_file_path):
        # Ensure the log folder exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Setup logging (basicConfig)
        logging.basicConfig(
            filename=log_file_path,
            filemode='w',  # overwrite each time
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)
