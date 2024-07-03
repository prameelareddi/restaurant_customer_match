'''import requests

url = "https://www.fast2sms.com/dev/bulkV2"

message = "This is test Message sent from \
         Python Script using REST API."
numbers = "9666565307, 7032061497"
payload = f'sender_id=FTWSMS&message={message}&route=p&language=english&numbers={numbers}'

headers = {
    'authorization': 'euYhmXBbj1y7JAC4OLVrn2Eov5qH09kDzPKUxN3TIdcWMSsG8Q6naFZhxIPrVqyED3U8smSAMzc7NOL2',
    'Content-Type': "application/x-www-form-urlencoded",
}

response = requests.request("POST", url=url, data=payload, headers=headers)

print(response.text)'''


'''# check version of keras_vggface
import keras_vggface
# print version
print(keras_vggface.__version__)'''




'''# confirm mtcnn was installed correctly
import mtcnn
# print version
print(mtcnn.__version__)'''


model = VGGFace(model='resnet50')