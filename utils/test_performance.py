import requests
import time
url = 'http://192.168.1.36:5000/api/esim'

values ={ "query": ["花呗的安全没有验证成功", "花呗的安全没有验证成功"],
           "target": ["花呗安全验证没通过怎么回事用", "花呗的安全没有验证成功"]}

values['query'] = ["花呗的安全没有验证成功"] * 10
values["target"] = ["花呗的安全没有验证成功"] * 10
epoch = 100
start = time.time()
for x in range(100):
    r = requests.post(
        url=url, json=values, headers={
            'Content-Type': 'application/json'})
end = time.time()
print((end - start) / 100)
