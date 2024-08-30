import time
import requests

# Define the file name
file_name = "example.txt"

# Open the file in write mode ('w') which creates the file if it doesn't exist
with open(file_name, 'w') as file:
    # Write some text into the file
    file.write("Hello, this is a sample text!")

print(f"File '{file_name}' has been created and text has been written into it.")
