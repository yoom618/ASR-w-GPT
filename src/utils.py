import os

def set_huggingface_cache_dir(cache_dir):
    os.system(f'export HF_HOME={cache_dir}')    # !export HF_HOME = cache_dir
    os.environ["HF_HOME"] = cache_dir
    # os.system('huggingface-cli whoami')         # !huggingface-cli whoami
    
    with open(os.path.join(cache_dir, 'token'), 'r') as f:
        token = f.read().strip()

    return token