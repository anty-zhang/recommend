import requests
import json

'''
两个获取ip的地理信息的接口:
http://freeapi.ipip.net/
http://ip-api.com/json/
'''

l = ['111.205.82.31', '221.216.60.66']
# {'China|GD|Guangdong|Shenzhen': 315, 'China|GD|Guangdong|Xiaolou': 10, 'China|GD|Guangdong|Guangzhou': 23, 'China|BJ|Beijing|Beijing': 38, 'China|JS|Jiangsu|Suzhou': 2, 'China|GD|Guangdong|Zhongshan': 1, 'China|SD|Shandong|Zibo': 10, 'China|HN|Hunan|Yueyang': 11, 'China|HN|Hunan|Changsha': 6, 'China|HB|Hubei|Wuhan': 3, 'China|GD|Guangdong|Zhaoqing': 1, 'China|ZJ|Zhejiang|Hangzhou': 1, 'China|GD|Guangdong|Huizhou': 1, 'China|BJ|Beijing|Jinrongjie': 6, "China|SN|Shaanxi|Xi'an": 6, 'China|BJ|Beijing|Babaoshan': 2, 'China|GD|Guangdong|Dongguan': 1, 'China|HE|Hebei|Shijiazhuang': 1, 'China|GD|Guangdong|Longgang District': 7, 'China|GD|Guangdong|Dalang': 1}
print(set(l))

# 获取本地ip
# ipAddress = requests.get('http://ip.42.pl/raw', ).text
# print(ipAddress)

res_dict = {}

for ipAddress in set(l):
    response = requests.get("http://ip-api.com/json/" + ipAddress).text
    # print(type(response))
    response = json.loads(response)  # 将网页的json格式的字符串数据转成字典
    # print(type(response))
    # print(response["countryCode"])
    # print(response)

    print(response)
    item = response['country'] + '|' + response['region'] + '|' + response['regionName'] + '|' + response['city']
    if item in res_dict:
        res_dict[item] += 1
    else:
        res_dict[item] = 1
print(res_dict)



