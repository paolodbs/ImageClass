import requests
from pathlib import Path
import os

#PATH NEEDS TO CHANGE
#SOLELY FOR TESTING PURPOSES
path_to_images = "/home/nicole/code/NicoleChant/images"
local_api_url = "http://127.0.0.1:8000/multipredict"

def get_extension(filename : str) -> str:
    return filename.split(".")[-1] if "." in filename else filename

def get_predictions(directory : Path) -> list[str]:
    images = []
    print("Preparing request...")
    admissible_file_extensions = ["jpg","png","jpeg"]
    for item in directory.iterdir():
        if item.is_file() and get_extension(item.name) in admissible_file_extensions:
            images.append( ( 'files', open(item , mode = "rb") ) )

    print("Sending Request...")
    print(images)
    response = requests.post(local_api_url , files = images)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error" : response.content }


if __name__ == "__main__":
    directory = Path(path_to_images)
    predictions = get_predictions(directory)
    print(predictions)
