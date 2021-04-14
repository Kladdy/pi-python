# Check the ip every 30 minutes and sends updates to Slack
# This uses the slack-cli library: https://github.com/rockymadden/slack-cli
from requests import get
import time
import subprocess
import json
import datetime

def get_date_time():
    return str(datetime.datetime.now())

def send_slack(ip):
    try:
        query = ["../../slack", "chat", "send", "-tx", f'ip update: {ip}', "-ch", "#pi"]
        res_json = subprocess.check_output(query) # Call slack process
        res = json.loads(res_json) # Parse json

        assert res["ok"] # Make sure we got ok as response

    except subprocess.CalledProcessError:
        print("Subprocess call failed!")
        print(f"Query: {' '.join(query)}")
        raise subprocess.CalledProcessError()

    except AssertionError as e: 
        print("Assertion 'ok == True' failed...")
        print(f"Query: {' '.join(query)}")
        print(res)
        raise AssertionError()


def check_ip(current_ip):
    print(f"{get_date_time()} ip check:")

    ip = get('https://api.ipify.org').text

    if ip != current_ip:
        print("ip has changed! Sending to Slack...")
        send_slack(ip)
        print("ip has been sent to Slack! Sleeping...")
    else:
        print("ip unchanged! Sleeping...")

    return ip

def main():
    # Initialize ip
    current_ip = "0.0.0.0"

    # Initial check of ip
    current_ip = check_ip(current_ip)

    while True:
        time.sleep(30*60) # Sleep for 30 minutes
        current_ip = check_ip(current_ip)
        
if __name__ == "__main__":
    main()