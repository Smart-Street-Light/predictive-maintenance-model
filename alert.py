import requests
import json
import cv2
import base64
import numpy as np
import time

from datetime import datetime





def alert(imgUrl,message, location, JsonData):
    
# Get current date and time

    now = datetime.now()
    current_date = now.strftime('%Y-%m-%dT%H:%M:%S.%fZ') 
# #     _, buffer = cv2.imencode('.jpg', img)
    
# #     #cv2.imshow("Age-gender", img)
# #     print(buffer)
# #     current_time = now.time()
# #     #cv2.imwrite(current_date+'.jpg',img)
    
# #     # Encode the byte buffer to base64
# #     im1 = base64.b64encode(buffer).decode('utf-8')
# #     url1 = "https://api.cloudinary.com/v1_1/your_cloud_name/image/upload"
# #     str='data:image/jpeg;base64,'+im1
# #     payload1 = {'upload_preset': 'dnp0feqt','api_key':'388377638726978'}
# #     files=[
# #   ('file',('file',buffer,'application/octet-stream'))
# # ]
#     headers1 = {}

#     response = requests.request("POST", url1, headers=headers1, data=payload1,files=files)
#     rs1=response.json()

#     print(response.text)
#     print(type(im1))
    # Extract date and time individually
    

    url = "http://localhost:5015/incidents"

    payload = json.dumps({
    "type": message,
    "date": current_date,
    "time": now.strftime('%H:%M:%S'),
    "location": location,
    "image": imgUrl,
    "extraJson": JsonData
    })
    print(payload)
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
