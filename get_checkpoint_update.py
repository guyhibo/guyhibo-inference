import boto3
import json
import os

def get_checkpoint_dir():
    print("Checking ckpt update")
    if len(os.listdir("./ckpt")) == 0:
        print("No ckpt in directory")
        f = open('project_env.json')
        env = json.load(f)
        
        s3_client = boto3.client(
        's3',
        aws_access_key_id=env["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=env["S3_ACCESS_SECRET_ACCESS_KEY"],
        region_name=env["S3_REGION"])
        
        s3_client.download_file(
            "guyhibo-pt-files",
            "custom_model8_3.pt",
            "./ckpt/custom_model8_3.pt")
    else:
        print("There is no update in ckpt")