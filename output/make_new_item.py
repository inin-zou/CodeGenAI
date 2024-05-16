
import requests

def make_new_item():
 return requests.post('https://example.com/items', json={'name': 'New Item'}).json()
