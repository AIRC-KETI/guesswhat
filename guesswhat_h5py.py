from datasets import DownloadMode, DownloadManager, DatasetInfo
import datasets
import cv2
import h5py
from typing import AnyStr

class AutoEncoderDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def __init__(self, training_pickle: AnyStr, validation_pickle: AnyStr, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.training_pickle = training_pickle
        self.validation_pickle = validation_pickle

    def _info(self) -> DatasetInfo:
        features = datasets.Features({
            "image": datasets.Image()
        })

        return datasets.DatasetInfo(
            features=features
        )

    def _split_generators(self, dl_manager: DownloadManager):
        splits = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "h5_path": self.training_pickle
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "h5_path": self.validation_pickle
                }
            )
        ]

        return splits

    def _generate_examples(self, h5_path: AnyStr):
        with h5py.File(h5_path, "r") as infile:
            images = infile["images"]

            for _id in range(images.shape[0]):
                yield _id, {
                    "image": cv2.imdecode(images[_id][-1], cv2.IMREAD_COLOR)
                }

h5py_file = "./datasets/all/assets_2/{train_val}/images.h5"
dataset_builder = AutoEncoderDataset(h5py_file.format(train_val="train"), h5py_file.format(train_val="val"))
dataset_builder.download_and_prepare(download_mode=DownloadMode.FORCE_REDOWNLOAD)
huggingface_dataset = dataset_builder.as_dataset()

for i in huggingface_dataset["train"]:
    print(i)