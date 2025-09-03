import requests
from random import choice
from PIL import Image
from io import BytesIO

def get_random_image(query: str):
    results = requests.get(
        "http://localhost:8000/search",
        params={"query": query, "source": "atomic-image", "k": 10}
    ).json()
    random_document = choice(results)
    image_url = random_document["document"]["image_url"]
    print(image_url)
    response = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0 (platform; rv:gecko-version) Gecko/gecko-trail Firefox/firefox-version"})
    print(response.content)
    img = Image.open(BytesIO(response.content))
    return img
    
im = get_random_image("cat")
print(im)