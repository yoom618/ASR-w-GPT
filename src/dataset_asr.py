from torch.utils.data import Dataset
from datasets import load_dataset

DATASET_ARGS = dict(
    ami = dict(
        huggingface=dict(
            path="speech-seq2seq/ami",
            name="ihm",
        ),
        phase=dict(
            train='train',
            valid='validation',
            test='test',
        ),
        audio_column='audio',
        text_column='text',
    ),
)

def load_asr_dataset(name, phase, cache_dir=None, trust_remote_code=True, **kwargs):
    if name == 'ami':
        return AMIDataset(phase, cache_dir=cache_dir, trust_remote_code=trust_remote_code, **kwargs)
    else:
        raise ValueError(f"Unknown custom dataset name: {name}")


class AMIDataset(Dataset):
    def __init__(self, phase, cache_dir=None, trust_remote_code=True, **kwargs):
        self.dataset = load_dataset(**DATASET_ARGS['ami']['huggingface'], 
                                    split=phase,
                                    cache_dir=cache_dir,
                                    trust_remote_code=trust_remote_code,
                                    **kwargs)
        
        self.audio_column = DATASET_ARGS['ami']['audio_column']
        self.text_column = DATASET_ARGS['ami']['text_column']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][self.audio_column], self.dataset[index][self.text_column]



if __name__ == "__main__":
    from utils import set_huggingface_cache_dir

    dataset_name = "ami"
    cache_dir = "/data/yoom618/datasets/"

    token = set_huggingface_cache_dir("/data/yoom618/datasets/")

    train_dataset = load_asr_dataset(dataset_name, 
                                     DATASET_ARGS[dataset_name]['phase']['train'], 
                                     cache_dir=cache_dir)
    valid_dataset = load_asr_dataset(dataset_name, 
                                     DATASET_ARGS[dataset_name]['phase']['valid'], 
                                     cache_dir=cache_dir)
    test_dataset = load_asr_dataset(dataset_name, 
                                    DATASET_ARGS[dataset_name]['phase']['test'], 
                                    cache_dir=cache_dir)

    print(len(train_dataset), len(valid_dataset), len(test_dataset))

    sample_idx = 0
    print(train_dataset[sample_idx])
    print(valid_dataset[sample_idx])
    print(test_dataset[sample_idx])
