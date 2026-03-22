import os
import urllib.request
import zipfile

EVAL_BUNDLE = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
LOCAL_PATH = os.path.expanduser("~/.cache/nanochat")

def main():

    os.makedirs(LOCAL_PATH, exist_ok=True)

    filepath_zip = os.path.join(LOCAL_PATH, 'eval_bundle.zip')
    if not os.path.exists(filepath_zip):
        print("Downloading eval bundle...")
        urllib.request.urlretrieve(EVAL_BUNDLE, filepath_zip)

    print("Unzipping eval bundle...")
    with zipfile.ZipFile(filepath_zip, 'r') as zip_ref:
        zip_ref.extractall(LOCAL_PATH)
    print("Eval bundle is ready.")


if __name__ == '__main__':
    main()
