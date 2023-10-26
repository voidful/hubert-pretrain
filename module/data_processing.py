from dataclasses import dataclass
from typing import Dict, List, Union

import torch
import torchaudio
from transformers import Wav2Vec2Processor


def encode_dataset(batch, processor):
    return batch


def create_label(code_generator, audio_feature):
    return [i['code'] for i in code_generator(input_values=torch.tensor(audio_feature))]


def prepare_dataset_hf(batch, processor, audio_feature_key, code_generator):
    audio = batch["audio"]
    batch[audio_feature_key] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).get(audio_feature_key)[0]
    batch["lengths"] = len(batch[audio_feature_key])
    batch["labels"] = torch.tensor(create_label(code_generator, batch[audio_feature_key])).squeeze()
    return batch


def prepare_dataset_custom(batch, audio_feature_key, code_generator):
    path = batch["path"]
    speech, sampling_rate = torchaudio.load(path)
    if sampling_rate != '16_000' or sampling_rate != '16000':
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        batch[audio_feature_key] = resampler.forward(speech.squeeze(0)).numpy()
    else:
        batch[audio_feature_key] = speech.squeeze(0).numpy()
    batch["lengths"] = len(batch[audio_feature_key])
    batch["labels"] = torch.tensor(create_label(code_generator, batch[audio_feature_key]))
    return batch


@dataclass
class DataCollatorWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    audio_feature_key: str = "input_values"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different length and need
        # different padding methods
        input_values = [{self.audio_feature_key: feature[self.audio_feature_key]} for feature in features]

        batch = self.processor.pad(
            input_values,
            padding=self.padding,
            return_tensors="pt",
        )

        # This might be what you intended, but double-check the logic to make sure.
        # Combining the features with the padded batch
        for idx, feature in enumerate(features):
            feature.update(batch)

        return features
