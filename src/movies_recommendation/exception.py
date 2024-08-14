import sys
import traceback

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_error_message(error_message, error_detail)

    def get_error_message(self, error_message, error_detail: sys):
        exc_type, exc_obj, exc_tb = error_detail.exc_info()
        
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            formatted_message = f"Error in file: {file_name}, line: {line_number}, message: {error_message}"
        else:
            formatted_message = f"Error: {error_message}. Traceback information is unavailable."
        
        return formatted_message

    def __str__(self):
        return self.error_message
