# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""LibriSpeech-PC dataset module refered from LibriSpeech dataset module."""


import os

import datasets

import json


_CITATION = {
    "librispeech":
    """\
    @inproceedings{panayotov2015librispeech,
    title={Librispeech: an ASR corpus based on public domain audio books},
    author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
    booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
    pages={5206--5210},
    year={2015},
    organization={IEEE}
    }""",
    "librispeech_pc":
    """\
    @article{meister2023librispeechpc,
        title={LibriSpeech-PC: Benchmark for Evaluation of Punctuation and Capitalization Capabilities of end-to-end ASR Models}, 
        author={A. Meister and M. Novikov and N. Karpov and E. Bakhturina and V. Lavrukhin and B. Ginsburg},
        journal={arXiv preprint arXiv:2310.02943},
        year={2023},
    }
    """
}

_DESCRIPTION = """\
Merge Librispeech audio files with punctuation and captalization restored transcripts from LibriSpeech-PC.
I refered to the original LibriSpeech dataset module script from HuggingFace Datasets (https://huggingface.co/datasets/openslr/librispeech_asr).
If you already have downloaded the LibriSpeech dataset via `load_dataset('openslr/librispeech_asr')`, the script will use the extracted audio files from the local directory and not download them twice. (only tested in my local environment though)
"""

_URL = "http://www.openslr.org/12"
_DL_URL = "http://www.openslr.org/resources/12/"

_URL_PC = "https://www.openslr.org/145"
_DL_URL_PC = "https://www.openslr.org/resources/145/"


_DL_URLS = {
    "clean": {
        "dev": _DL_URL + "dev-clean.tar.gz",
        "test": _DL_URL + "test-clean.tar.gz",
        "train.100": _DL_URL + "train-clean-100.tar.gz",
        "train.360": _DL_URL + "train-clean-360.tar.gz",
        "transcript_pc": _DL_URL_PC + "manifests.tar.gz",
    },
    "other": {
        "test": _DL_URL + "test-other.tar.gz",
        "dev": _DL_URL + "dev-other.tar.gz",
        "train.500": _DL_URL + "train-other-500.tar.gz",
        "transcript_pc": _DL_URL_PC + "manifests.tar.gz",
    },
    "all": {
        "dev.clean": _DL_URL + "dev-clean.tar.gz",
        "dev.other": _DL_URL + "dev-other.tar.gz",
        "test.clean": _DL_URL + "test-clean.tar.gz",
        "test.other": _DL_URL + "test-other.tar.gz",
        "train.clean.100": _DL_URL + "train-clean-100.tar.gz",
        "train.clean.360": _DL_URL + "train-clean-360.tar.gz",
        "train.other.500": _DL_URL + "train-other-500.tar.gz",
        "transcript_pc": _DL_URL_PC + "manifests.tar.gz",
    },
}


class LibrispeechASRConfig(datasets.BuilderConfig):
    """BuilderConfig for LibriSpeechASR."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(LibrispeechASRConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)


class LibrispeechASR(datasets.GeneratorBasedBuilder):
    """Librispeech dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 256
    DEFAULT_CONFIG_NAME = "all"
    BUILDER_CONFIGS = [
        LibrispeechASRConfig(name="clean", description="'Clean' speech."),
        LibrispeechASRConfig(name="other", description="'Other', more challenging, speech."),
        LibrispeechASRConfig(name="all", description="Combined clean and other dataset."),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "text_raw": datasets.Value("string"),
                    "text_normalized": datasets.Value("string"),
                    "speaker_id": datasets.Value("int64"),
                    "chapter_id": datasets.Value("int64"),
                    "id": datasets.Value("string"),
                    "duration": datasets.Value("float"),
                }
            ),
            supervised_keys=("file", "text"),
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download(_DL_URLS[self.config.name])
        # (Optional) In non-streaming mode, we can extract the archive locally to have actual local audio files:
        local_extracted_archive = dl_manager.extract(archive_path) if not dl_manager.is_streaming else {}
        
        # print(local_extracted_archive)
        # print(list(dl_manager.iter_archive(archive_path["transcript_pc"])))
        transcript_pc_dir = local_extracted_archive.get("transcript_pc")
        
        if self.config.name == "clean":
            train_splits = [
                datasets.SplitGenerator(
                    name="train.100",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("train.100"),
                        "files": dl_manager.iter_archive(archive_path["train.100"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "train-clean-100.json"),
                    },
                ),
                datasets.SplitGenerator(
                    name="train.360",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("train.360"),
                        "files": dl_manager.iter_archive(archive_path["train.360"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "train-clean-360.json"),
                    },
                ),
            ]
            dev_splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("dev"),
                        "files": dl_manager.iter_archive(archive_path["dev"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "dev-clean.json"),
                    },
                )
            ]
            test_splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("test"),
                        "files": dl_manager.iter_archive(archive_path["test"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "test-clean.json"),
                    },
                )
            ]
        elif self.config.name == "other":
            train_splits = [
                datasets.SplitGenerator(
                    name="train.500",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("train.500"),
                        "files": dl_manager.iter_archive(archive_path["train.500"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "train-other-500.json"),
                    },
                )
            ]
            dev_splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("dev"),
                        "files": dl_manager.iter_archive(archive_path["dev"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "dev-other.json"),
                    },
                )
            ]
            test_splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("test"),
                        "files": dl_manager.iter_archive(archive_path["test"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "test-other.json"),
                    },
                )
            ]
        elif self.config.name == "all":
            train_splits = [
                datasets.SplitGenerator(
                    name="train.clean.100",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("train.clean.100"),
                        "files": dl_manager.iter_archive(archive_path["train.clean.100"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "train-clean-100.json"),
                    },
                ),
                datasets.SplitGenerator(
                    name="train.clean.360",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("train.clean.360"),
                        "files": dl_manager.iter_archive(archive_path["train.clean.360"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "train-clean-360.json"),
                    },
                ),
                datasets.SplitGenerator(
                    name="train.other.500",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("train.other.500"),
                        "files": dl_manager.iter_archive(archive_path["train.other.500"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "train-other-500.json"),
                    },
                ),
            ]
            dev_splits = [
                datasets.SplitGenerator(
                    name="validation.clean",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("dev.clean"),
                        "files": dl_manager.iter_archive(archive_path["dev.clean"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "dev-clean.json"),
                    },
                ),
                datasets.SplitGenerator(
                    name="validation.other",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("dev.other"),
                        "files": dl_manager.iter_archive(archive_path["dev.other"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "dev-other.json"),
                    },
                ),
            ]
            test_splits = [
                datasets.SplitGenerator(
                    name="test.clean",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("test.clean"),
                        "files": dl_manager.iter_archive(archive_path["test.clean"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "test-clean.json"),
                    },
                ),
                datasets.SplitGenerator(
                    name="test.other",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("test.other"),
                        "files": dl_manager.iter_archive(archive_path["test.other"]),
                        "transcript_pc_fname": os.path.join(transcript_pc_dir, "test-other.json"),
                    },
                ),
            ]

        return train_splits + dev_splits + test_splits

    # def _generate_examples(self, files, local_extracted_archive):  # original
    #     """Generate examples from a LibriSpeech archive_path."""
    #     key = 0
    #     audio_data = {}
    #     transcripts = []
    #     for path, f in files:
    #         if path.endswith(".flac"):
    #             id_ = path.split("/")[-1][: -len(".flac")]
    #             audio_data[id_] = f.read()
    #         elif path.endswith(".trans.txt"):
    #             for line in f:
    #                 if line:
    #                     line = line.decode("utf-8").strip()
    #                     id_, transcript = line.split(" ", 1)
    #                     audio_file = f"{id_}.flac"
    #                     speaker_id, chapter_id = [int(el) for el in id_.split("-")[:2]]
    #                     audio_file = (
    #                         os.path.join(local_extracted_archive, audio_file)
    #                         if local_extracted_archive
    #                         else audio_file
    #                     )
    #                     transcripts.append(
    #                         {
    #                             "id": id_,
    #                             "speaker_id": speaker_id,
    #                             "chapter_id": chapter_id,
    #                             "file": audio_file,
    #                             "text": transcript,
    #                         }
    #                     )
    #         if audio_data and len(audio_data) == len(transcripts):
    #             for transcript in transcripts:
    #                 audio = {"path": transcript["file"], "bytes": audio_data[transcript["id"]]}
    #                 yield key, {"audio": audio, **transcript}
    #                 key += 1
    #             audio_data = {}
    #             transcripts = []



    def _generate_examples(self, files, local_extracted_archive, transcript_pc_fname):  # original
        """Generate examples from a LibriSpeech archive_path."""
        key, unseen = 0, 0
        audio_data = {}
        transcripts = []

        # Load transcripts from LibriSpeech-PC
        transcripts_pc = dict()
        with open(transcript_pc_fname, mode='r') as f:
            data = (f.read().splitlines())
            data = [json.loads(d) for d in data]
            for d in data:
                _id = d['audio_filepath'].split("/")[-1][: -len(".flac")]
                del d['audio_filepath']
                transcripts_pc.update(
                    {_id: d}  # keys in d : duration, text, text_raw
                )
        
        os.makedirs("./unexisting_transcripts_id", exist_ok=True)
        try:
            os.remove(f"./unexisting_transcripts_id/{os.path.basename(transcript_pc_fname)[:-5]}.txt")
        except FileNotFoundError:
            pass 
        
        for path, f in files:
            if path.endswith(".flac"):
                id_ = path.split("/")[-1][: -len(".flac")]
                audio_data[id_] = f.read()
            elif path.endswith(".trans.txt"):
                for line in f:
                    if line:
                        line = line.decode("utf-8").strip()
                        id_, transcript = line.split(" ", 1)
                        audio_file = f"{id_}.flac"
                        speaker_id, chapter_id = [int(el) for el in id_.split("-")[:2]]
                        audio_file = (
                            os.path.join(local_extracted_archive, audio_file)
                            if local_extracted_archive
                            else audio_file
                        )
                        transcripts.append(
                            {
                                "id": id_,
                                "speaker_id": speaker_id,
                                "chapter_id": chapter_id,
                                "file": audio_file,
                                "text_normalized": transcript,
                            }
                        )

            if audio_data and len(audio_data) == len(transcripts):
                for transcript in transcripts:
                    audio = {"path": transcript["file"], "bytes": audio_data[transcript["id"]]}
                    transcript_pc = transcripts_pc.pop(transcript["id"], {})
                    if transcript_pc:
                        yield key, {"audio": audio, **transcript, **transcript_pc}
                        key += 1
                    else:
                        with open(f"./unexisting_transcripts_id/{os.path.basename(transcript_pc_fname)[:-5]}.txt", mode='a') as log:
                            log.write(f"{transcript['id']}\n")
                        unseen += 1
                audio_data = {}
                transcripts = []

        print(f"{unseen} transcripts are dropped in LibriSpeech-PC dataset {os.path.basename(transcript_pc_fname)[:-5]} compared to LibriSpeech dataset.")



