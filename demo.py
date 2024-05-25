import openai
import cv2
import os
import base64
import requests
from loguru import logger
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from loguru import logger


# Example usage
video_path = "2021-11-15-14-31-02.avi"
num_frames = 8
api_key = ""

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# Define the question
question = "Where are the car parking? And how many cars are there?"

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize variables to keep track of the best match
best_image_file = None
best_similarity_score = -float('inf')

# Process each image
for image_file in os.listdir('frames'):
    image_path = os.path.join('frames', image_file)
    image = Image.open(image_path)
    
    inputs = processor(text=[question], images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

    # Get the similarity score
    similarity_score = logits_per_image.item()  # since we have only one text input, we can take the single value

    # Update the best match if this score is higher
    if similarity_score > best_similarity_score:
        best_similarity_score = similarity_score
        best_image_file = image_file

# Output the most relevant image file
print(f"The most relevant image is: {best_image_file} with a similarity score of {best_similarity_score}")


# Load knowledge graph data
with open('kg/kg_output/graph_no_quotes.json', 'r') as kg_file:
    kg_data = json.load(kg_file)
    doc_id = best_image_file.split('.')[0].split('_')[1]
    
    images = []
    images.append({
        'frame_id': doc_id,
        'location': os.path.join('frames', 'frame_{}.png'.format(doc_id)),
        'caption': kg_data['graph']['doc_info'][doc_id]['image'],
    })
    
    for coref_doc in kg_data['graph']['coreferences'][doc_id]:
        images.append({
            'frame_id': coref_doc,
            'location': os.path.join('frames', 'frame_{}.png'.format(str(coref_doc))),
            'caption': kg_data['graph']['doc_info'][str(coref_doc)]['image'],
        })
        

# Prepare the system prompt
system_prompt = '''You are an AI assistant tasked with analyzing the video frames and captions that will be provided in order to answer the given question. Carefully examine each image and its associated caption. Based on the information given in the images and captions, provide a clear answer to the question. If the images and captions do not contain enough information to conclusively answer the question, indicate that the answer is unclear given the limited information available. Do not make assumptions or inferences beyond what is explicitly stated or shown. You should also elaborate where the evidence is found.'''



# Prepare messages for the API call
messages = [
    {
        'role': 'system',
        'content': [
            {
                'type': 'text',
                'text': system_prompt
            }
        ]
    },
    {
        'role': 'user',
        'content': [
            {
                'type': 'text',
                'text': question,
            }
        ]
    }
]

# Add all relevant images and their captions
for image in images:
    messages[1]['content'].append({
        "type": "text",
        "text": f"Image ID[{image['frame_id']}]: {image['caption']}"
    })

for image in images:
    messages[1]['content'].append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image['location'])}"
            }
        }
    )


# Call the OpenAI API
client = openai.Client(api_key=api_key)
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=messages,
    max_tokens=300
)

# Print the response
print(response.choices[0].message.content)