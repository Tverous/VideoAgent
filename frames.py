import openai
import cv2
import os
import base64
import requests
from loguru import logger
import json

# Example usage
video_path = "2021-11-15-14-31-02.avi"
num_frames = 8
api_key = ""


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def split_video_into_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames // num_frames

    frames = []
    for i in range(num_frames):
        frame_index = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if ret:
            frame_path = f"frames/frame_{i}.png"
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        else:
            print(f"Failed to extract frame {i}.")

    cap.release()
    return frames

def generate_captions(frames, api_key):
    openai.api_key = api_key

    contents = []
    contents.append({
        "type": "text",
        "text": "Generate a caption for the following image:"
    })

    captions = []
    for frame_path in frames:
        # Getting the base64 string
        base64_image = encode_image(frame_path)
        
        # contents.append({
        #     "type": "image_url",
        #     "image_url": {
        #         "url": f"data:image/jpeg;base64,{base64_image}"
        #     }
        # })
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "What’s in this image?"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
        }
        
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        logger.info(response.json()['choices'][0]['message']['content'])
        captions.append({
            'frame': frame_path,
            'caption': response.json()['choices'][0]['message']['content']
        })

    return captions

def save_captions_to_json(captions, json_file):
    with open(json_file, 'w') as file:
        json.dump(captions, file, indent=4)

def cleanup_frames(frames):
    for frame_path in frames:
        os.remove(frame_path)



frames = split_video_into_frames(video_path, num_frames)
# captions = generate_captions(frames, api_key)

# save_captions_to_json(captions, 'captions.json')
# cleanup_frames(frames)