# Send a heartbeat every 5 minutes to Slack
# This uses the slack-cli library: https://github.com/rockymadden/slack-cli
from requests import get
import time
import subprocess
import json
import datetime

def get_date_time():
    return str(datetime.datetime.now())

def send_slack():
    try:
        query = ["../../slack", "chat", "send", "-tx", f'Heartbeat start: {get_date_time()}', "-ch", '#pi']
        res_json = subprocess.check_output(query) # Call slack process
        res = json.loads(res_json) # Parse json
        print(res)
        assert res["ok"] # Make sure we got ok as response

        timestamp = res["ts"]
        channel = res["channel"]
        
        return timestamp, channel

    except subprocess.CalledProcessError:
        print("Subprocess call failed!")
        print(f"Query: {' '.join(query)}")
        raise subprocess.CalledProcessError()

    except AssertionError: 
        print("Assertion 'ok == True' failed...")
        print(f"Query: {' '.join(query)}")
        print(res)
        raise AssertionError()

def update_slack(timestamp, channel):
    try:
        query = ["../../slack", "chat", "update", "-tx", f'Heartbeat update: {get_date_time()}', "-ch", f"{channel}", "-ts", f'{timestamp}']
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

def main():
    print("Sending first message to Slack...")
    timestamp, channel = send_slack()
    print("Done sending first message. Sleeping...")

    while True:
        time.sleep(10*60) # Sleep for 10 minutes
        print("Updating Slack...")
        update_slack(timestamp, channel)
        print("Done updating Slack! Sleeping...")
        
        
        
if __name__ == "__main__":
    main()