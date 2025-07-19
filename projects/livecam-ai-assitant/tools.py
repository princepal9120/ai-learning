import cv2
from dotenv import load_dotenv
import os
import base64
from groq import Groq


load_dotenv()
def capture_image()-> str:
    """
    Captures one frame from the default webcam, resize it,
    encodes it as Base64 JPEG (raw string ) and returns it.
    """
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            for _ in range(10):
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            cv2.imwrite("sample.jpg", frame) # save frame as JPEG file
            ret, buf= cv2.imencode(".jpg", frame)
            if ret:
                return base64.b64encode(buf).decode('utf-8')
    raise RuntimeError("Could not open any webcam tried indices 0-3")
                

def anaylze_image_with_query(query: str)-> str:  
    """
    Expects a string withe 'query'      
    Captures the image and sends 
    the query and the image to 
    Groq's vision caht api and return the anaylsis.
    """

    img_base64= capture_image()
    model="meta-llama/llama-4-maverick-17b-128e-instruct"

    if not query or not img_base64:
        return "Error: query or image not found"
    
    client= Groq()
    messages=[
        {
            "role": "user",
            "content": [
               { "type": "text",
                "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                    }
                }
            ]
        }
    ]
    chat_completion= client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return chat_completion.choices[0].message.content 



 