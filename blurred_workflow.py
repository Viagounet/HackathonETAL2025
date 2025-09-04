import openai
import os
from openai import OpenAI
from generate_mask import generate
from random import choice
from PIL import Image
from flou import flou, enleve_masque, fusion_masque, pourcentage_reussite
import numpy as np
import cv2

print("~~~ Bienvenue dans le jeu défloutage! Votre objectif est de déflouter complètement l'image affichée ~~~")
currently_discovered = 0

client = OpenAI(
    api_key=os.getenv("LITELLM_API_KEY_SEMAFOR"), base_url="https://llmproxy.ai.orange/"
)


def llm_call(
    messages,
    model: str = "openai/gpt-4.1-mini",
) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return completion.choices[0].message.content


# image_to_blur = choice(["brittany-roscoff-hd.jpg", "cars.jpg", "groceries.jpg", "truck.jpg"])
image_to_blur = "notebooks/images/brittany-roscoff-hd.jpg"
initial_image = Image.open(image_to_blur).convert("RGB")
initial_mask = np.zeros(initial_image.size[::-1])
initial_mask = np.stack([initial_mask * 255] * 3, axis=-1).astype(np.uint8)

blurred_image = flou(initial_image, 161)

while currently_discovered < 95:
    user_input = input("> ")
    object_to_detect = llm_call([{"role": "system", "content": "The user is trying to unblur something in the image, please extract in 1 or 2 words (max, in english) what the user is looking for. Example => 'Je pense que l'image contient un bateau' ; you should answer: 'boat'"}, {"role": "user", "content": user_input}]) + "."
    print(f"OBJECT TO DETECT: {object_to_detect}")
    
    mask = generate(initial_image, object_to_detect)
    blurred_image, new_mask = enleve_masque(initial_image, blurred_image, initial_mask, mask)

    initial_mask = new_mask
    currently_discovered = pourcentage_reussite(initial_mask)
    print(currently_discovered)
    
    im = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    im.save("blurred.png")

    result = llm_call([{"role": "system", "content": "You are a helpful assistant for a game. The following is the game description given to the user: You will be shown a blurred image, and then guess what you think you are seeing. If you guess correctly, the real object(s) will become clear. Once you reveal most of the objects in the image, you win!\n\n____\n\nThe caption of that the user has to discover is the following: 'A beautiful city next to the sea, with a few boats visible'. If you see the user struggling, please give him a little hint, if he's not struggling, tell the player to keep playing."}, {"role": "user", "content": "Write a smart thing"}])
    print(result)