import requests

BOT_TOKEN = "8393591405:AAGs6mj4KDaxefBVxv7d6qKegy86WEWAOgk"
CHAT_ID = "1548646865"

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, data=data, timeout=5)
    except:
        pass
