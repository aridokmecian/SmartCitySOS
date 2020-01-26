import requests
import json
import calendar
import time
import datetime

timestamp = str(int((time.time()) * 1000))
print (timestamp)
print("capturing video ...")
time.sleep(31)
print("video captured.") 
print ("waiting for video")
# time.sleep(240)
url = "https://hamilton.cityiq.io/api/v2/media/ondemand/assets/f6057765-ae16-4b8a-b0b8-c48de3b193c6/media?mediaType=VIDEO&timestamp=" + str(int(timestamp)) + "&duration=30"
payload = {}
headers = { 
  'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImxlZ2FjeS10b2tlbi1rZXkiLCJ0eXAiOiJKV1QifQ.eyJqdGkiOiIzNmE4OTgyOGQxOTc0NzBlYTRjMjk0MDAxOTk0YTEwYSIsInN1YiI6IkhhY2thdGhvbi5DSVRNLkhhbWlsdG9uIiwiYXV0aG9yaXRpZXMiOlsiaWUtY3VycmVudC5IQU1JTFRPTi1JRS1QRURFU1RSSUFOLklFLVBFREVTVFJJQU4uTElNSVRFRC5ERVZFTE9QIiwidWFhLnJlc291cmNlIiwiaWUtY3VycmVudC5IQU1JTFRPTi1JRS1QQVJLSU5HLklFLVBBUktJTkcuTElNSVRFRC5ERVZFTE9QIiwiaWUtY3VycmVudC5IQU1JTFRPTi1JRS1CSUNZQ0xFLklFLUJJQ1lDTEUuTElNSVRFRC5ERVZFTE9QIiwiaWUtY3VycmVudC5IQU1JTFRPTi1JRS1UUkFGRklDLklFLVRSQUZGSUMuTElNSVRFRC5ERVZFTE9QIiwiaWUtY3VycmVudC5IQU1JTFRPTi1JRS1FTlZJUk9OTUVOVEFMLklFLUVOVklST05NRU5UQUwuTElNSVRFRC5ERVZFTE9QIiwiaWUtY3VycmVudC5IQU1JTFRPTi1JRS1JTUFHRS5JRS1JTUFHRS5MSU1JVEVELkRFVkVMT1AiLCJpZS1jdXJyZW50LkhBTUlMVE9OLUlFLVZJREVPLklFLVZJREVPLkxJTUlURUQuREVWRUxPUCIsImllLWN1cnJlbnQuSEFNSUxUT04tSUUtQVVESU8uSUUtQVVESU8uTElNSVRFRC5ERVZFTE9QIl0sInNjb3BlIjpbImllLWN1cnJlbnQuSEFNSUxUT04tSUUtUEVERVNUUklBTi5JRS1QRURFU1RSSUFOLkxJTUlURUQuREVWRUxPUCIsInVhYS5yZXNvdXJjZSIsImllLWN1cnJlbnQuSEFNSUxUT04tSUUtUEFSS0lORy5JRS1QQVJLSU5HLkxJTUlURUQuREVWRUxPUCIsImllLWN1cnJlbnQuSEFNSUxUT04tSUUtQklDWUNMRS5JRS1CSUNZQ0xFLkxJTUlURUQuREVWRUxPUCIsImllLWN1cnJlbnQuSEFNSUxUT04tSUUtVFJBRkZJQy5JRS1UUkFGRklDLkxJTUlURUQuREVWRUxPUCIsImllLWN1cnJlbnQuSEFNSUxUT04tSUUtRU5WSVJPTk1FTlRBTC5JRS1FTlZJUk9OTUVOVEFMLkxJTUlURUQuREVWRUxPUCIsImllLWN1cnJlbnQuSEFNSUxUT04tSUUtSU1BR0UuSUUtSU1BR0UuTElNSVRFRC5ERVZFTE9QIiwiaWUtY3VycmVudC5IQU1JTFRPTi1JRS1WSURFTy5JRS1WSURFTy5MSU1JVEVELkRFVkVMT1AiLCJpZS1jdXJyZW50LkhBTUlMVE9OLUlFLUFVRElPLklFLUFVRElPLkxJTUlURUQuREVWRUxPUCJdLCJjbGllbnRfaWQiOiJIYWNrYXRob24uQ0lUTS5IYW1pbHRvbiIsImNpZCI6IkhhY2thdGhvbi5DSVRNLkhhbWlsdG9uIiwiYXpwIjoiSGFja2F0aG9uLkNJVE0uSGFtaWx0b24iLCJncmFudF90eXBlIjoiY2xpZW50X2NyZWRlbnRpYWxzIiwicmV2X3NpZyI6IjliM2ZmYjhhIiwiaWF0IjoxNTgwMDAxOTIxLCJleHAiOjE1ODA2MDY3MjEsImlzcyI6Imh0dHBzOi8vYXV0aC5hYS5jaXR5aXEuaW8vb2F1dGgvdG9rZW4iLCJ6aWQiOiJ1YWEiLCJhdWQiOlsiaWUtY3VycmVudC5IQU1JTFRPTi1JRS1JTUFHRS5JRS1JTUFHRS5MSU1JVEVEIiwiSGFja2F0aG9uLkNJVE0uSGFtaWx0b24iLCJpZS1jdXJyZW50LkhBTUlMVE9OLUlFLVBFREVTVFJJQU4uSUUtUEVERVNUUklBTi5MSU1JVEVEIiwiaWUtY3VycmVudC5IQU1JTFRPTi1JRS1CSUNZQ0xFLklFLUJJQ1lDTEUuTElNSVRFRCIsInVhYSIsImllLWN1cnJlbnQuSEFNSUxUT04tSUUtQVVESU8uSUUtQVVESU8uTElNSVRFRCIsImllLWN1cnJlbnQuSEFNSUxUT04tSUUtUEFSS0lORy5JRS1QQVJLSU5HLkxJTUlURUQiLCJpZS1jdXJyZW50LkhBTUlMVE9OLUlFLVRSQUZGSUMuSUUtVFJBRkZJQy5MSU1JVEVEIiwiaWUtY3VycmVudC5IQU1JTFRPTi1JRS1FTlZJUk9OTUVOVEFMLklFLUVOVklST05NRU5UQUwuTElNSVRFRCIsImllLWN1cnJlbnQuSEFNSUxUT04tSUUtVklERU8uSUUtVklERU8uTElNSVRFRCJdfQ.ifJM_dPgRWzRcMxbcIVa9DsLbMebpXTBfJMtorTXxbggmlC8JgitZ7sNRJz2dS-5CsJhgo_UMEasDnwhaP2PqbQD1W10BwUcAV4Q49sF_LhR1VehttWrwcni8bpKx2BSn0eeZNEMZp25fjJrqPJ3RoaMazDrFFLkfg7hosQllUIY9uc-vZCgseotENrfah6eTTR_RZ2g-fUuRbg5_w9egx02g306PJg-9TTNUsXnlcKqu2fRkGDmMciXbTz-FwTU4Ov5jpDSvN_8yIabA49kKMZb4gmjqSSXAmheeYwCJcClydh-DTSQFUzFh-wGpXTBZIGIz9OWZgbBqaLKNFh8ay1UCO62ZF7DIDMHQMkEO606g2P_lOxbam1WFwVTT6SFu12xeqjPLmPB-X8tT3xfr3YiPQkj_edizlIpTmIq-_saWWrXpDaEnunftDPl13e8K3cGze1RP5FxlAeGXz0UHOe3Vis5PrXlaG_yiHMKUpusRJ-9mfyuRqdLnXCU-zU5gNeS6ymJ1IkUupFOKzYn-WaTonvkoiS2Jq8Pxbt_rBf1O2Dfaorp0WwlOU9v_ZMYxpBhzhIVuCrGIYMFvrxgRi56htuRifVAMN70WAHmrNd-cruojmGTzdSDx09LpUzReCoMipgrV7zuLsRFaTF72lW-2a_dBhrQITENUmq6N0E',
  'Predix-Zone-Id': 'HAMILTON-IE-VIDEO'
}
print("downloading video ...")
response = requests.request("GET", url, headers=headers, data = payload)
url = json.loads(response.text)
round2 = url["pollUrl"]
print(round2)
response = requests.request("GET", round2, headers=headers, data = payload)
temp = json.loads(response.text)
while (temp['status'] != "SUCCESS"):
  time.sleep(1)
  response = requests.request("GET", round2, headers=headers, data = payload)
  print(temp)
  temp = json.loads(response.text)
# url = json.loads(response.text)
round3 = temp["listOfEntries"]["content"][0]["url"]
response = requests.request("GET", round3, headers=headers, data = payload)
print("downloaded video.")
f = open('binary.mp4', 'wb')
# print("writing video ...")
for b in response:
    f.write(b)
f.close()
print ("finished writing video.")
print("Done :D")