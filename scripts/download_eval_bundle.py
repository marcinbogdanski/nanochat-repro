import os
import urllib.request
import zipfile
from nanorepro.common import get_base_path
BASE_DIR = get_base_path()

EVAL_BUNDLE = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def main():
    filepath_zip = os.path.join(BASE_DIR, 'eval_bundle.zip')
    if not os.path.exists(filepath_zip):
        print("Downloading eval bundle...")
        urllib.request.urlretrieve(EVAL_BUNDLE, filepath_zip)

    print("Unzipping eval bundle...")
    with zipfile.ZipFile(filepath_zip, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)
    print("Cleaning up zip file...")
    os.remove(filepath_zip)
    print("Eval bundle is ready.")


if __name__ == '__main__':
    main()
