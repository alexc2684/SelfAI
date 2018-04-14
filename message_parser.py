from bs4 import BeautifulSoup as bs
import os

MSG_PATH = '/Users/alexchan/Documents/college/personal/facebook-achan216/messages'

for msg in os.listdir(MSG_PATH):
    if msg.endswith('html'):
        path = MSG_PATH + '/' + msg
        with open(path, 'r') as f:
            print(path)
            html = bs(f, 'html.parser')
            print(html.get_text())
    # break
