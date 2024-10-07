from http import client
from sys import displayhook
from bs4 import BeautifulSoup
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import random
import time
import requests
import json
import webbrowser as wb

def uzgun():
    client_id = ""
    client_secret = ""
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))
    playlist_url = "https://open.spotify.com/playlist/6zgn9Tg0cFkTwlKFpflQPQ?si=c82009edb5a34606"
    

    playlist = spotify.playlist(playlist_url)
    # pd.json_normalize(playlist)
    playlist_items = spotify.playlist_items(playlist_url)
    deneme = str(playlist_items)
    df = pd.json_normalize(playlist_items['items'])
    filter_cols = [col for col in df]
    df = df[filter_cols]
    df.columns = [col.replace("track.","") for col in df]
    link = ""
    sarki_isimleri = []
    for i in range(50):
        link = df["name"].iloc[i]
        sarki_isimleri.append(link)
    link = "https://www.youtube.com/results?search_query=şanışer" + str(sarki_isimleri[random.randint(0,50)])
    wb.open(link)
    print(link)
    
    
def mutlu():
    client_id = ""
    client_secret = ""
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))
    playlist_url = "https://open.spotify.com/playlist/37i9dQZF1DX9ASuQophyb3?si=e9c67c46f5784e11"
    

    playlist = spotify.playlist(playlist_url)
    playlist_items = spotify.playlist_items(playlist_url)
    deneme = str(playlist_items)
    df = pd.json_normalize(playlist_items['items'])
    filter_cols = [col for col in df]
    df = df[filter_cols]
    df.columns = [col.replace("track.","") for col in df]
    link = ""
    sarki_isimleri = []
    for i in range(50):
        link = df["name"].iloc[i]
        sarki_isimleri.append(link)
    link = "https://www.youtube.com/results?search_query=" + str(sarki_isimleri[random.randint(0,50)])
    wb.open(link)
    print(link)