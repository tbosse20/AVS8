import base64
import collections
import os
import random
from typing import Dict, List, Union

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from TTS.tts.utils.data import prepare_data, prepare_stop_target, prepare_tensor
from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import compute_energy as calculate_energy
#NEW IMPORTS
import torchaudio
from librosa.core import resample
from librosa.util import fix_length
from transformers import AutoFeatureExtractor, Wav2Vec2ForXVector
#####

# to prevent too many open files error as suggested here
# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
torch.multiprocessing.set_sharing_strategy("file_system")


def _parse_sample(item):
    language_name = None
    attn_file = None
    if len(item) == 5:
        text, wav_file, speaker_name, language_name, attn_file = item
    elif len(item) == 4:
        text, wav_file, speaker_name, language_name = item
    elif len(item) == 3:
        text, wav_file, speaker_name = item
    else:
        raise ValueError(" [!] Dataset cannot parse the sample.")
    return text, wav_file, speaker_name, language_name, attn_file


def noise_augment_audio(wav):
    return wav + (1.0 / 32768.0) * np.random.rand(*wav.shape)


def string2filename(string):
    # generate a safe and reversible filename based on a string
    filename = base64.urlsafe_b64encode(string.encode("utf-8")).decode("utf-8", "ignore")
    return filename


class TTSDataset(Dataset):
    def __init__(
        self,
        outputs_per_step: int = 1,
        compute_linear_spec: bool = False,
        ap: AudioProcessor = None,
        samples: List[Dict] = None,
        tokenizer: "TTSTokenizer" = None,
        compute_f0: bool = False,
        compute_energy: bool = False,
        f0_cache_path: str = None,
        energy_cache_path: str = None,
        #NEW CHANGES FALSE TO TRUE
        return_wav: bool = True,
        #####
        batch_group_size: int = 0,
        min_text_len: int = 0,
        max_text_len: int = float("inf"),
        min_audio_len: int = 0,
        max_audio_len: int = float("inf"),
        phoneme_cache_path: str = None,
        precompute_num_workers: int = 0,
        speaker_id_mapping: Dict = None,
        d_vector_mapping: Dict = None,
        language_id_mapping: Dict = None,
        use_noise_augment: bool = False,
        start_by_longest: bool = False,
        verbose: bool = False,
    ):
        """Generic 📂 data loader for `tts` models. It is configurable for different outputs and needs.

        If you need something different, you can subclass and override.

        Args:
            outputs_per_step (int): Number of time frames predicted per step.

            compute_linear_spec (bool): compute linear spectrogram if True.

            ap (TTS.tts.utils.AudioProcessor): Audio processor object.

            samples (list): List of dataset samples.

            tokenizer (TTSTokenizer): tokenizer to convert text to sequence IDs. If None init internally else
                use the given. Defaults to None.

            compute_f0 (bool): compute f0 if True. Defaults to False.

            compute_energy (bool): compute energy if True. Defaults to False.

            f0_cache_path (str): Path to store f0 cache. Defaults to None.

            energy_cache_path (str): Path to store energy cache. Defaults to None.

            return_wav (bool): Return the waveform of the sample. Defaults to False.

            batch_group_size (int): Range of batch randomization after sorting
                sequences by length. It shuffles each batch with bucketing to gather similar lenght sequences in a
                batch. Set 0 to disable. Defaults to 0.

            min_text_len (int): Minimum length of input text to be used. All shorter samples will be ignored.
                Defaults to 0.

            max_text_len (int): Maximum length of input text to be used. All longer samples will be ignored.
                Defaults to float("inf").

            min_audio_len (int): Minimum length of input audio to be used. All shorter samples will be ignored.
                Defaults to 0.

            max_audio_len (int): Maximum length of input audio to be used. All longer samples will be ignored.
                The maximum length in the dataset defines the VRAM used in the training. Hence, pay attention to
                this value if you encounter an OOM error in training. Defaults to float("inf").

            phoneme_cache_path (str): Path to cache computed phonemes. It writes phonemes of each sample to a
                separate file. Defaults to None.

            precompute_num_workers (int): Number of workers to precompute features. Defaults to 0.

            speaker_id_mapping (dict): Mapping of speaker names to IDs used to compute embedding vectors by the
                embedding layer. Defaults to None.

            d_vector_mapping (dict): Mapping of wav files to computed d-vectors. Defaults to None.

            use_noise_augment (bool): Enable adding random noise to wav for augmentation. Defaults to False.

            start_by_longest (bool): Start by longest sequence. It is especially useful to check OOM. Defaults to False.

            verbose (bool): Print diagnostic information. Defaults to false.
        """
        super().__init__()
        self.batch_group_size = batch_group_size
        self._samples = samples
        self.outputs_per_step = outputs_per_step
        self.compute_linear_spec = compute_linear_spec
        self.return_wav = return_wav
        self.compute_f0 = compute_f0
        self.compute_energy = compute_energy
        self.f0_cache_path = f0_cache_path
        self.energy_cache_path = energy_cache_path
        self.min_audio_len = min_audio_len
        self.max_audio_len = max_audio_len
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.ap = ap
        self.phoneme_cache_path = phoneme_cache_path
        self.speaker_id_mapping = speaker_id_mapping
        self.d_vector_mapping = d_vector_mapping
        self.language_id_mapping = language_id_mapping
        self.use_noise_augment = use_noise_augment
        self.start_by_longest = start_by_longest

        self.verbose = verbose
        self.rescue_item_idx = 1
        self.pitch_computed = False
        self.tokenizer = tokenizer

        if self.tokenizer.use_phonemes:
            self.phoneme_dataset = PhonemeDataset(
                self.samples, self.tokenizer, phoneme_cache_path, precompute_num_workers=precompute_num_workers
            )

        if compute_f0:
            self.f0_dataset = F0Dataset(
                self.samples, self.ap, cache_path=f0_cache_path, precompute_num_workers=precompute_num_workers
            )
        if compute_energy:
            self.energy_dataset = EnergyDataset(
                self.samples, self.ap, cache_path=energy_cache_path, precompute_num_workers=precompute_num_workers
            )
        if self.verbose:
            self.print_logs()

    @property
    def lengths(self):
        lens = []
        for item in self.samples:
            _, wav_file, *_ = _parse_sample(item)
            audio_len = os.path.getsize(wav_file) / 16 * 8  # assuming 16bit audio
            lens.append(audio_len)
        return lens

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, new_samples):
        self._samples = new_samples
        if hasattr(self, "f0_dataset"):
            self.f0_dataset.samples = new_samples
        if hasattr(self, "energy_dataset"):
            self.energy_dataset.samples = new_samples
        if hasattr(self, "phoneme_dataset"):
            self.phoneme_dataset.samples = new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.load_data(idx)

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print("\n")
        print(f"{indent}> DataLoader initialization")
        print(f"{indent}| > Tokenizer:")
        self.tokenizer.print_logs(level + 1)
        print(f"{indent}| > Number of instances : {len(self.samples)}")

    def load_wav(self, filename):
        waveform = self.ap.load_wav(filename)
        assert waveform.size > 0
        return waveform

    def get_phonemes(self, idx, text):
        out_dict = self.phoneme_dataset[idx]
        assert text == out_dict["text"], f"{text} != {out_dict['text']}"
        assert len(out_dict["token_ids"]) > 0
        return out_dict

    def get_f0(self, idx):
        out_dict = self.f0_dataset[idx]
        item = self.samples[idx]
        assert item["audio_unique_name"] == out_dict["audio_unique_name"]
        return out_dict

    def get_energy(self, idx):
        out_dict = self.energy_dataset[idx]
        item = self.samples[idx]
        assert item["audio_unique_name"] == out_dict["audio_unique_name"]
        return out_dict

    @staticmethod
    def get_attn_mask(attn_file):
        return np.load(attn_file)

    def get_token_ids(self, idx, text):
        if self.tokenizer.use_phonemes:
            token_ids = self.get_phonemes(idx, text)["token_ids"]
        else:
            token_ids = self.tokenizer.text_to_ids(text)
        return np.array(token_ids, dtype=np.int32)

    def load_data(self, idx):
        item = self.samples[idx]

        raw_text = item["text"]

        wav = np.asarray(self.load_wav(item["audio_file"]), dtype=np.float32)

        # apply noise for augmentation
        if self.use_noise_augment:
            wav = noise_augment_audio(wav)

        # get token ids
        token_ids = self.get_token_ids(idx, item["text"])

        # get pre-computed attention maps
        attn = None
        if "alignment_file" in item:
            attn = self.get_attn_mask(item["alignment_file"])

        # after phonemization the text length may change
        # this is a shareful 🤭 hack to prevent longer phonemes
        # TODO: find a better fix
        if len(token_ids) > self.max_text_len or len(wav) < self.min_audio_len:
            self.rescue_item_idx += 1
            return self.load_data(self.rescue_item_idx)

        # get f0 values
        f0 = None
        if self.compute_f0:
            f0 = self.get_f0(idx)["f0"]
        energy = None
        if self.compute_energy:
            energy = self.get_energy(idx)["energy"]

        sample = {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "wav": wav,
            "pitch": f0,
            "energy": energy,
            "attn": attn,
            "item_idx": item["audio_file"],
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
            "wav_file_name": os.path.basename(item["audio_file"]),
            "audio_unique_name": item["audio_unique_name"],
        }
        return sample

    @staticmethod
    def _compute_lengths(samples):
        new_samples = []
        for item in samples:
            audio_length = os.path.getsize(item["audio_file"]) / 16 * 8  # assuming 16bit audio
            text_lenght = len(item["text"])
            item["audio_length"] = audio_length
            item["text_length"] = text_lenght
            new_samples += [item]
        return new_samples

    @staticmethod
    def filter_by_length(lengths: List[int], min_len: int, max_len: int):
        idxs = np.argsort(lengths)  # ascending order
        ignore_idx = []
        keep_idx = []
        for idx in idxs:
            length = lengths[idx]
            if length < min_len or length > max_len:
                ignore_idx.append(idx)
            else:
                keep_idx.append(idx)
        return ignore_idx, keep_idx

    @staticmethod
    def sort_by_length(samples: List[List]):
        audio_lengths = [s["audio_length"] for s in samples]
        idxs = np.argsort(audio_lengths)  # ascending order
        return idxs

    @staticmethod
    def create_buckets(samples, batch_group_size: int):
        assert batch_group_size > 0
        for i in range(len(samples) // batch_group_size):
            offset = i * batch_group_size
            end_offset = offset + batch_group_size
            temp_items = samples[offset:end_offset]
            random.shuffle(temp_items)
            samples[offset:end_offset] = temp_items
        return samples

    @staticmethod
    def _select_samples_by_idx(idxs, samples):
        samples_new = []
        for idx in idxs:
            samples_new.append(samples[idx])
        return samples_new

    def preprocess_samples(self):
        r"""Sort `items` based on text length or audio length in ascending order. Filter out samples out or the length
        range.
        """
        samples = self._compute_lengths(self.samples)

        # sort items based on the sequence length in ascending order
        text_lengths = [i["text_length"] for i in samples]
        audio_lengths = [i["audio_length"] for i in samples]
        text_ignore_idx, text_keep_idx = self.filter_by_length(text_lengths, self.min_text_len, self.max_text_len)
        audio_ignore_idx, audio_keep_idx = self.filter_by_length(audio_lengths, self.min_audio_len, self.max_audio_len)
        keep_idx = list(set(audio_keep_idx) & set(text_keep_idx))
        ignore_idx = list(set(audio_ignore_idx) | set(text_ignore_idx))

        samples = self._select_samples_by_idx(keep_idx, samples)

        sorted_idxs = self.sort_by_length(samples)

        if self.start_by_longest:
            longest_idxs = sorted_idxs[-1]
            sorted_idxs[-1] = sorted_idxs[0]
            sorted_idxs[0] = longest_idxs

        samples = self._select_samples_by_idx(sorted_idxs, samples)

        if len(samples) == 0:
            raise RuntimeError(" [!] No samples left")

        # shuffle batch groups
        # create batches with similar length items
        # the larger the `batch_group_size`, the higher the length variety in a batch.
        if self.batch_group_size > 0:
            samples = self.create_buckets(samples, self.batch_group_size)

        # update items to the new sorted items
        audio_lengths = [s["audio_length"] for s in samples]
        text_lengths = [s["text_length"] for s in samples]
        self.samples = samples

        if self.verbose:
            print(" | > Preprocessing samples")
            print(" | > Max text length: {}".format(np.max(text_lengths)))
            print(" | > Min text length: {}".format(np.min(text_lengths)))
            print(" | > Avg text length: {}".format(np.mean(text_lengths)))
            print(" | ")
            print(" | > Max audio length: {}".format(np.max(audio_lengths)))
            print(" | > Min audio length: {}".format(np.min(audio_lengths)))
            print(" | > Avg audio length: {}".format(np.mean(audio_lengths)))
            print(f" | > Num. instances discarded samples: {len(ignore_idx)}")
            print(" | > Batch group size: {}.".format(self.batch_group_size))

    @staticmethod
    def _sort_batch(batch, text_lengths):
        """Sort the batch by the input text length for RNN efficiency.

        Args:
            batch (Dict): Batch returned by `__getitem__`.
            text_lengths (List[int]): Lengths of the input character sequences.
        """
        text_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor(text_lengths), dim=0, descending=True)
        batch = [batch[idx] for idx in ids_sorted_decreasing]
        return batch, text_lengths, ids_sorted_decreasing

    def collate_fn(self, batch):
        r"""
        Perform preprocessing and create a final data batch:
        1. Sort batch instances by text-length
        2. Convert Audio signal to features.
        3. PAD sequences wrt r.
        4. Load to Torch.
        """
        #NEW COMPUTE EMBEDDINGS
        def spk_embedding(audio, sr:int = 16000) -> torch.Tensor:
            feat_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
            spk_emb_model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")
            audio = resample(np.array(audio), orig_sr=sr, target_sr=16000)
            inputs = feat_extractor(audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                embeddings = spk_emb_model(**inputs).embeddings
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        
            return embeddings
        #####

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.abc.Mapping):
            token_ids_lengths = np.array([len(d["token_ids"]) for d in batch])

            # sort items with text input length for RNN efficiency
            batch, token_ids_lengths, ids_sorted_decreasing = self._sort_batch(batch, token_ids_lengths)

            # convert list of dicts to dict of lists
            batch = {k: [dic[k] for dic in batch] for k in batch[0]}
            #NEW ADD SPK_EMBEDDING
            batch["wav"] = [fix_length(w, size=int(len(w)*1.01)) for w in batch["wav"]]
            wav_lengths = [w.shape[0] for w in batch["wav"]]
            
            max_wav_len = max(wav_lengths)
            spk_embeddings_list = []
            for w in batch["wav"]:
                w = fix_length(w, size=max_wav_len)
                embeddings = spk_embedding(w)
                spk_embeddings_list.append(embeddings)
            #####
            # get language ids from language names
            if self.language_id_mapping is not None:
                language_ids = [self.language_id_mapping[ln] for ln in batch["language_name"]]
            else:
                language_ids = None
            # get pre-computed d-vectors
            if self.d_vector_mapping is not None:
                embedding_keys = list(batch["audio_unique_name"])
                d_vectors = [self.d_vector_mapping[w]["embedding"] for w in embedding_keys]
            else:
                d_vectors = None

            # get numerical speaker ids from speaker names
            if self.speaker_id_mapping:
                speaker_ids = [self.speaker_id_mapping[sn] for sn in batch["speaker_name"]]
            else:
                speaker_ids = None
            # compute features
            mel = [self.ap.melspectrogram(w).astype("float32") for w in batch["wav"]]

            mel_lengths = [m.shape[1] for m in mel]

            # lengths adjusted by the reduction factor
            mel_lengths_adjusted = [
                m.shape[1] + (self.outputs_per_step - (m.shape[1] % self.outputs_per_step))
                if m.shape[1] % self.outputs_per_step
                else m.shape[1]
                for m in mel
            ]

            # compute 'stop token' targets
            stop_targets = [np.array([0.0] * (mel_len - 1) + [1.0]) for mel_len in mel_lengths]

            # PAD stop targets
            stop_targets = prepare_stop_target(stop_targets, self.outputs_per_step)

            # PAD sequences with longest instance in the batch
            token_ids = prepare_data(batch["token_ids"]).astype(np.int32)

            # PAD features with longest instance
            mel = prepare_tensor(mel, self.outputs_per_step)

            # B x D x T --> B x T x D
            mel = mel.transpose(0, 2, 1)

            # convert things to pytorch
            token_ids_lengths = torch.LongTensor(token_ids_lengths)
            token_ids = torch.LongTensor(token_ids)
            mel = torch.FloatTensor(mel).contiguous()
            mel_lengths = torch.LongTensor(mel_lengths)
            stop_targets = torch.FloatTensor(stop_targets)

            # speaker vectors
            if d_vectors is not None:
                d_vectors = torch.FloatTensor(d_vectors)

            if speaker_ids is not None:
                speaker_ids = torch.LongTensor(speaker_ids)

            if language_ids is not None:
                language_ids = torch.LongTensor(language_ids)

            # compute linear spectrogram
            linear = None
            if self.compute_linear_spec:
                linear = [self.ap.spectrogram(w).astype("float32") for w in batch["wav"]]
                linear = prepare_tensor(linear, self.outputs_per_step)
                linear = linear.transpose(0, 2, 1)
                assert mel.shape[1] == linear.shape[1]
                linear = torch.FloatTensor(linear).contiguous()

            # format waveforms
            wav_padded = None
            if self.return_wav:
                wav_lengths = [w.shape[0] for w in batch["wav"]]
                max_wav_len = max(mel_lengths_adjusted) * self.ap.hop_length
                wav_lengths = torch.LongTensor(wav_lengths)
                wav_padded = torch.zeros(len(batch["wav"]), 1, max_wav_len)
                for i, w in enumerate(batch["wav"]):
                    mel_length = mel_lengths_adjusted[i]
                    w = np.pad(w, (0, self.ap.hop_length * self.outputs_per_step), mode="edge")
                    w = w[: mel_length * self.ap.hop_length]
                    wav_padded[i, :, : w.shape[0]] = torch.from_numpy(w)
                wav_padded.transpose_(1, 2)

            ###
            # format F0
            if self.compute_f0:
                pitch = prepare_data(batch["pitch"])
                assert mel.shape[1] == pitch.shape[1], f"[!] {mel.shape} vs {pitch.shape}"
                pitch = torch.FloatTensor(pitch)[:, None, :].contiguous()  # B x 1 xT
            else:
                pitch = None
            # format energy
            if self.compute_energy:
                energy = prepare_data(batch["energy"])
                assert mel.shape[1] == energy.shape[1], f"[!] {mel.shape} vs {energy.shape}"
                energy = torch.FloatTensor(energy)[:, None, :].contiguous()  # B x 1 xT
            else:
                energy = None
            # format attention masks
            attns = None
            if batch["attn"][0] is not None:
                attns = [batch["attn"][idx].T for idx in ids_sorted_decreasing]
                for idx, attn in enumerate(attns):
                    pad2 = mel.shape[1] - attn.shape[1]
                    pad1 = token_ids.shape[1] - attn.shape[0]
                    assert pad1 >= 0 and pad2 >= 0, f"[!] Negative padding - {pad1} and {pad2}"
                    attn = np.pad(attn, [[0, pad1], [0, pad2]])
                    attns[idx] = attn
                attns = prepare_tensor(attns, self.outputs_per_step)
                attns = torch.FloatTensor(attns).unsqueeze(1)

            return {
                "token_id": token_ids,
                "token_id_lengths": token_ids_lengths,
                "speaker_names": batch["speaker_name"],
                "linear": linear,
                "mel": mel,
                "mel_lengths": mel_lengths,
                "stop_targets": stop_targets,
                "item_idxs": batch["item_idx"],
                "d_vectors": d_vectors,
                "speaker_ids": speaker_ids,
                "attns": attns,
                "waveform": wav_padded,
                "raw_text": batch["raw_text"],
                "pitch": pitch,
                "energy": energy,
                "language_ids": language_ids,
                "audio_unique_names": batch["audio_unique_name"],
                "spk_emb": spk_embeddings_list, #NEW here
            }

        raise TypeError(
            (
                "batch must contain tensors, numbers, dicts or lists;\
                         found {}".format(
                    type(batch[0])
                )
            )
        )


class PhonemeDataset(Dataset):
    """Phoneme Dataset for converting input text to phonemes and then token IDs

    At initialization, it pre-computes the phonemes under `cache_path` and loads them in training to reduce data
    loading latency. If `cache_path` is already present, it skips the pre-computation.

    Args:
        samples (Union[List[List], List[Dict]]):
            List of samples. Each sample is a list or a dict.

        tokenizer (TTSTokenizer):
            Tokenizer to convert input text to phonemes.

        cache_path (str):
            Path to cache phonemes. If `cache_path` is already present or None, it skips the pre-computation.

        precompute_num_workers (int):
            Number of workers used for pre-computing the phonemes. Defaults to 0.
    """

    def __init__(
        self,
        samples: Union[List[Dict], List[List]],
        tokenizer: "TTSTokenizer",
        cache_path: str,
        precompute_num_workers=0,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.cache_path = cache_path
        if cache_path is not None and not os.path.exists(cache_path):
            os.makedirs(cache_path)
            self.precompute(precompute_num_workers)

    def __getitem__(self, index):
        item = self.samples[index]
        ids = self.compute_or_load(string2filename(item["audio_unique_name"]), item["text"], item["language"])
        ph_hat = self.tokenizer.ids_to_text(ids)
        return {"text": item["text"], "ph_hat": ph_hat, "token_ids": ids, "token_ids_len": len(ids)}

    def __len__(self):
        return len(self.samples)

    def compute_or_load(self, file_name, text, language):
        """Compute phonemes for the given text.

        If the phonemes are already cached, load them from cache.
        """
        file_ext = "_phoneme.npy"
        cache_path = os.path.join(self.cache_path, file_name + file_ext)
        try:
            ids = np.load(cache_path)
        except FileNotFoundError:
            ids = self.tokenizer.text_to_ids(text, language=language)
            np.save(cache_path, ids)
        return ids

    def get_pad_id(self):
        """Get pad token ID for sequence padding"""
        return self.tokenizer.pad_id

    def precompute(self, num_workers=1):
        """Precompute phonemes for all samples.

        We use pytorch dataloader because we are lazy.
        """
        print("[*] Pre-computing phonemes...")
        with tqdm.tqdm(total=len(self)) as pbar:
            batch_size = num_workers if num_workers > 0 else 1
            dataloder = torch.utils.data.DataLoader(
                batch_size=batch_size, dataset=self, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn
            )
            for _ in dataloder:
                pbar.update(batch_size)

    def collate_fn(self, batch):
        ids = [item["token_ids"] for item in batch]
        ids_lens = [item["token_ids_len"] for item in batch]
        texts = [item["text"] for item in batch]
        texts_hat = [item["ph_hat"] for item in batch]
        ids_lens_max = max(ids_lens)
        ids_torch = torch.LongTensor(len(ids), ids_lens_max).fill_(self.get_pad_id())
        for i, ids_len in enumerate(ids_lens):
            ids_torch[i, :ids_len] = torch.LongTensor(ids[i])
        return {"text": texts, "ph_hat": texts_hat, "token_ids": ids_torch}

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print("\n")
        print(f"{indent}> PhonemeDataset ")
        print(f"{indent}| > Tokenizer:")
        self.tokenizer.print_logs(level + 1)
        print(f"{indent}| > Number of instances : {len(self.samples)}")


class F0Dataset:
    """F0 Dataset for computing F0 from wav files in CPU

    Pre-compute F0 values for all the samples at initialization if `cache_path` is not None or already present. It
    also computes the mean and std of F0 values if `normalize_f0` is True.

    Args:
        samples (Union[List[List], List[Dict]]):
            List of samples. Each sample is a list or a dict.

        ap (AudioProcessor):
            AudioProcessor to compute F0 from wav files.

        cache_path (str):
            Path to cache F0 values. If `cache_path` is already present or None, it skips the pre-computation.
            Defaults to None.

        precompute_num_workers (int):
            Number of workers used for pre-computing the F0 values. Defaults to 0.

        normalize_f0 (bool):
            Whether to normalize F0 values by mean and std. Defaults to True.
    """

    def __init__(
        self,
        samples: Union[List[List], List[Dict]],
        ap: "AudioProcessor",
        audio_config=None,  # pylint: disable=unused-argument
        verbose=False,
        cache_path: str = None,
        precompute_num_workers=0,
        normalize_f0=True,
    ):
        self.samples = samples
        self.ap = ap
        self.verbose = verbose
        self.cache_path = cache_path
        self.normalize_f0 = normalize_f0
        self.pad_id = 0.0
        self.mean = None
        self.std = None
        if cache_path is not None and not os.path.exists(cache_path):
            os.makedirs(cache_path)
            self.precompute(precompute_num_workers)
        if normalize_f0:
            self.load_stats(cache_path)

    def __getitem__(self, idx):
        item = self.samples[idx]
        f0 = self.compute_or_load(item["audio_file"], string2filename(item["audio_unique_name"]))
        if self.normalize_f0:
            assert self.mean is not None and self.std is not None, " [!] Mean and STD is not available"
            f0 = self.normalize(f0)
        return {"audio_unique_name": item["audio_unique_name"], "f0": f0}

    def __len__(self):
        return len(self.samples)

    def precompute(self, num_workers=0):
        print("[*] Pre-computing F0s...")
        with tqdm.tqdm(total=len(self)) as pbar:
            batch_size = num_workers if num_workers > 0 else 1
            # we do not normalize at preproessing
            normalize_f0 = self.normalize_f0
            self.normalize_f0 = False
            dataloder = torch.utils.data.DataLoader(
                batch_size=batch_size, dataset=self, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn
            )
            computed_data = []
            for batch in dataloder:
                f0 = batch["f0"]
                computed_data.append(f for f in f0)
                pbar.update(batch_size)
            self.normalize_f0 = normalize_f0

        if self.normalize_f0:
            computed_data = [tensor for batch in computed_data for tensor in batch]  # flatten
            pitch_mean, pitch_std = self.compute_pitch_stats(computed_data)
            pitch_stats = {"mean": pitch_mean, "std": pitch_std}
            np.save(os.path.join(self.cache_path, "pitch_stats"), pitch_stats, allow_pickle=True)

    def get_pad_id(self):
        return self.pad_id

    @staticmethod
    def create_pitch_file_path(file_name, cache_path):
        pitch_file = os.path.join(cache_path, file_name + "_pitch.npy")
        return pitch_file

    @staticmethod
    def _compute_and_save_pitch(ap, wav_file, pitch_file=None):
        wav = ap.load_wav(wav_file)
        pitch = ap.compute_f0(wav)
        if pitch_file:
            np.save(pitch_file, pitch)
        return pitch

    @staticmethod
    def compute_pitch_stats(pitch_vecs):
        nonzeros = np.concatenate([v[np.where(v != 0.0)[0]] for v in pitch_vecs])
        mean, std = np.mean(nonzeros), np.std(nonzeros)
        return mean, std

    def load_stats(self, cache_path):
        stats_path = os.path.join(cache_path, "pitch_stats.npy")
        stats = np.load(stats_path, allow_pickle=True).item()
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)

    def normalize(self, pitch):
        zero_idxs = np.where(pitch == 0.0)[0]
        pitch = pitch - self.mean
        pitch = pitch / self.std
        pitch[zero_idxs] = 0.0
        return pitch

    def denormalize(self, pitch):
        zero_idxs = np.where(pitch == 0.0)[0]
        pitch *= self.std
        pitch += self.mean
        pitch[zero_idxs] = 0.0
        return pitch

    def compute_or_load(self, wav_file, audio_unique_name):
        """
        compute pitch and return a numpy array of pitch values
        """
        pitch_file = self.create_pitch_file_path(audio_unique_name, self.cache_path)
        if not os.path.exists(pitch_file):
            pitch = self._compute_and_save_pitch(self.ap, wav_file, pitch_file)
        else:
            pitch = np.load(pitch_file)
        return pitch.astype(np.float32)

    def collate_fn(self, batch):
        audio_unique_name = [item["audio_unique_name"] for item in batch]
        f0s = [item["f0"] for item in batch]
        f0_lens = [len(item["f0"]) for item in batch]
        f0_lens_max = max(f0_lens)
        f0s_torch = torch.LongTensor(len(f0s), f0_lens_max).fill_(self.get_pad_id())
        for i, f0_len in enumerate(f0_lens):
            f0s_torch[i, :f0_len] = torch.LongTensor(f0s[i])
        return {"audio_unique_name": audio_unique_name, "f0": f0s_torch, "f0_lens": f0_lens}

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print("\n")
        print(f"{indent}> F0Dataset ")
        print(f"{indent}| > Number of instances : {len(self.samples)}")


class EnergyDataset:
    """Energy Dataset for computing Energy from wav files in CPU

    Pre-compute Energy values for all the samples at initialization if `cache_path` is not None or already present. It
    also computes the mean and std of Energy values if `normalize_Energy` is True.

    Args:
        samples (Union[List[List], List[Dict]]):
            List of samples. Each sample is a list or a dict.

        ap (AudioProcessor):
            AudioProcessor to compute Energy from wav files.

        cache_path (str):
            Path to cache Energy values. If `cache_path` is already present or None, it skips the pre-computation.
            Defaults to None.

        precompute_num_workers (int):
            Number of workers used for pre-computing the Energy values. Defaults to 0.

        normalize_Energy (bool):
            Whether to normalize Energy values by mean and std. Defaults to True.
    """

    def __init__(
        self,
        samples: Union[List[List], List[Dict]],
        ap: "AudioProcessor",
        verbose=False,
        cache_path: str = None,
        precompute_num_workers=0,
        normalize_energy=True,
    ):
        self.samples = samples
        self.ap = ap
        self.verbose = verbose
        self.cache_path = cache_path
        self.normalize_energy = normalize_energy
        self.pad_id = 0.0
        self.mean = None
        self.std = None
        if cache_path is not None and not os.path.exists(cache_path):
            os.makedirs(cache_path)
            self.precompute(precompute_num_workers)
        if normalize_energy:
            self.load_stats(cache_path)

    def __getitem__(self, idx):
        item = self.samples[idx]
        energy = self.compute_or_load(item["audio_file"], string2filename(item["audio_unique_name"]))
        if self.normalize_energy:
            assert self.mean is not None and self.std is not None, " [!] Mean and STD is not available"
            energy = self.normalize(energy)
        return {"audio_unique_name": item["audio_unique_name"], "energy": energy}

    def __len__(self):
        return len(self.samples)

    def precompute(self, num_workers=0):
        print("[*] Pre-computing energys...")
        with tqdm.tqdm(total=len(self)) as pbar:
            batch_size = num_workers if num_workers > 0 else 1
            # we do not normalize at preproessing
            normalize_energy = self.normalize_energy
            self.normalize_energy = False
            dataloder = torch.utils.data.DataLoader(
                batch_size=batch_size, dataset=self, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn
            )
            computed_data = []
            for batch in dataloder:
                energy = batch["energy"]
                computed_data.append(e for e in energy)
                pbar.update(batch_size)
            self.normalize_energy = normalize_energy

        if self.normalize_energy:
            computed_data = [tensor for batch in computed_data for tensor in batch]  # flatten
            energy_mean, energy_std = self.compute_energy_stats(computed_data)
            energy_stats = {"mean": energy_mean, "std": energy_std}
            np.save(os.path.join(self.cache_path, "energy_stats"), energy_stats, allow_pickle=True)

    def get_pad_id(self):
        return self.pad_id

    @staticmethod
    def create_energy_file_path(wav_file, cache_path):
        file_name = os.path.splitext(os.path.basename(wav_file))[0]
        energy_file = os.path.join(cache_path, file_name + "_energy.npy")
        return energy_file

    @staticmethod
    def _compute_and_save_energy(ap, wav_file, energy_file=None):
        wav = ap.load_wav(wav_file)
        energy = calculate_energy(wav, fft_size=ap.fft_size, hop_length=ap.hop_length, win_length=ap.win_length)
        if energy_file:
            np.save(energy_file, energy)
        return energy

    @staticmethod
    def compute_energy_stats(energy_vecs):
        nonzeros = np.concatenate([v[np.where(v != 0.0)[0]] for v in energy_vecs])
        mean, std = np.mean(nonzeros), np.std(nonzeros)
        return mean, std

    def load_stats(self, cache_path):
        stats_path = os.path.join(cache_path, "energy_stats.npy")
        stats = np.load(stats_path, allow_pickle=True).item()
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)

    def normalize(self, energy):
        zero_idxs = np.where(energy == 0.0)[0]
        energy = energy - self.mean
        energy = energy / self.std
        energy[zero_idxs] = 0.0
        return energy

    def denormalize(self, energy):
        zero_idxs = np.where(energy == 0.0)[0]# coding: utf-8

from typing import Dict, List, Union

import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.layers.tacotron.capacitron_layers import CapacitronVAE
from TTS.tts.layers.tacotron.gst_layers import GST
from TTS.tts.layers.tacotron.tacotron2 import Decoder, Encoder, Postnet
from TTS.tts.models.base_tacotron import BaseTacotron
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.capacitron_optimizer import CapacitronOptimizer
#NEW IMPORTS#
from TTS.vocoder.models.gan import GAN
from TTS.config import load_config
import numpy as np
import torchaudio
from librosa.core import resample
from transformers import AutoFeatureExtractor, Wav2Vec2ForXVector
from librosa.util import fix_length

#####

#NEW PATH#
VOCODER_CONFIG_PATH = "./vocoder/vocoder_models--universal--libri-tts--fullband-melgan/config.json"
VOCODER_MODEL = "./vocoder/vocoder_models--universal--libri-tts--fullband-melgan/model_file.pth"
#####

class Tacotron2(BaseTacotron):
    """Tacotron2 model implementation inherited from :class:`TTS.tts.models.base_tacotron.BaseTacotron`.

    Paper::
        https://arxiv.org/abs/1712.05884

    Paper abstract::
        This paper describes Tacotron 2, a neural network architecture for speech synthesis directly from text.
        The system is composed of a recurrent sequence-to-sequence feature prediction network that maps character
        embeddings to mel-scale spectrograms, followed by a modified WaveNet model acting as a vocoder to synthesize
        timedomain waveforms from those spectrograms. Our model achieves a mean opinion score (MOS) of 4.53 comparable
        to a MOS of 4.58 for professionally recorded speech. To validate our design choices, we present ablation
        studies of key components of our system and evaluate the impact of using mel spectrograms as the input to
        WaveNet instead of linguistic, duration, and F0 features. We further demonstrate that using a compact acoustic
        intermediate representation enables significant simplification of the WaveNet architecture.

    Check :class:`TTS.tts.configs.tacotron2_config.Tacotron2Config` for model arguments.

    Args:
        config (TacotronConfig):
            Configuration for the Tacotron2 model.
        speaker_manager (SpeakerManager):
            Speaker manager for multi-speaker training. Uuse only for multi-speaker training. Defaults to None.
    """

    def __init__(
        self,
        config: "Tacotron2Config",
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)

        self.decoder_output_dim = config.out_channels

        # pass all config fields to `self`
        # for fewer code change
        for key in config:
            setattr(self, key, config[key])

        # init multi-speaker layers
        if self.use_speaker_embedding or self.use_d_vector_file:
            self.init_multispeaker(config)
            self.decoder_in_features += self.embedded_speaker_dim  # add speaker embedding dim

        if self.use_gst:
            self.decoder_in_features += self.gst.gst_embedding_dim

        if self.use_capacitron_vae:
            self.decoder_in_features += self.capacitron_vae.capacitron_VAE_embedding_dim

        # embedding layer
        self.embedding = nn.Embedding(self.num_chars, 512, padding_idx=0)

        # base model layers
        self.encoder = Encoder(self.encoder_in_features)

        self.decoder = Decoder(
            self.decoder_in_features,
            self.decoder_output_dim,
            self.r,
            self.attention_type,
            self.attention_win,
            self.attention_norm,
            self.prenet_type,
            self.prenet_dropout,
            self.use_forward_attn,
            self.transition_agent,
            self.forward_attn_mask,
            self.location_attn,
            self.attention_heads,
            self.separate_stopnet,
            self.max_decoder_steps,
        )
        self.postnet = Postnet(self.out_channels)

        # setup prenet dropout
        self.decoder.prenet.dropout_at_inference = self.prenet_dropout_at_inference

        #NEW VOCODER#
        self.vocoder = GAN(load_config(VOCODER_CONFIG_PATH))
        self.vocoder.load_checkpoint(config=load_config(VOCODER_CONFIG_PATH), checkpoint_path=VOCODER_MODEL, eval=True)

        #NEW SPK EMBEDDING#
        self.spk_emb_model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
        #####

        # global style token layers
        if self.gst and self.use_gst:
            self.gst_layer = GST(
                num_mel=self.decoder_output_dim,
                num_heads=self.gst.gst_num_heads,
                num_style_tokens=self.gst.gst_num_style_tokens,
                gst_embedding_dim=self.gst.gst_embedding_dim,
            )

        # Capacitron VAE Layers
        if self.capacitron_vae and self.use_capacitron_vae:
            self.capacitron_vae_layer = CapacitronVAE(
                num_mel=self.decoder_output_dim,
                encoder_output_dim=self.encoder_in_features,
                capacitron_VAE_embedding_dim=self.capacitron_vae.capacitron_VAE_embedding_dim,
                speaker_embedding_dim=self.embedded_speaker_dim
                if self.capacitron_vae.capacitron_use_speaker_embedding
                else None,
                text_summary_embedding_dim=self.capacitron_vae.capacitron_text_summary_embedding_dim
                if self.capacitron_vae.capacitron_use_text_summary_embeddings
                else None,
            )

        # backward pass decoder
        if self.bidirectional_decoder:
            self._init_backward_decoder()
        # setup DDC
        if self.double_decoder_consistency:
            self.coarse_decoder = Decoder(
                self.decoder_in_features,
                self.decoder_output_dim,
                self.ddc_r,
                self.attention_type,
                self.attention_win,
                self.attention_norm,
                self.prenet_type,
                self.prenet_dropout,
                self.use_forward_attn,
                self.transition_agent,
                self.forward_attn_mask,
                self.location_attn,
                self.attention_heads,
                self.separate_stopnet,
                self.max_decoder_steps,
            )

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        """Final reshape of the model output tensors."""
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments
    
    def spk_embedding(self, audio_batch, sr:int = 24000) -> torch.Tensor:
        wav_lengths = [w.shape[1] for w in audio_batch]
        max_wav_len = max(wav_lengths)
        embeddings = []
        for audio in audio_batch:
            audio = resample(np.array(audio), orig_sr=sr, target_sr=16000)
            audio = fix_length(audio, size=int(max_wav_len*1.5))
            inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                embedding = self.spk_emb_model(**inputs).embeddings
            embedding = torch.nn.functional.normalize(embedding, dim=-1).cpu()
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
    
        return embeddings

    def forward(  # pylint: disable=dangerous-default-value=None
        self, text, text_lengths, mel_specs=None, mel_lengths=None, aux_input={"speaker_ids": None, "d_vectors": None}, raw_audio=None
    ):
        """Forward pass for training with Teacher Forcing.

        Shapes:
            text: :math:`[B, T_in]`
            text_lengths: :math:`[B]`
            mel_specs: :math:`[B, T_out, C]`
            mel_lengths: :math:`[B]`
            aux_input: 'speaker_ids': :math:`[B, 1]` and  'd_vectors': :math:`[B, C]`
        """
        #NEW AUDIO FROM MEL SPECTOGRAM
        aux_input = self._format_aux_input(aux_input)
        outputs = {"alignments_backward": None, "decoder_outputs_backward": None}
        # compute mask for padding
        # B x T_in_max (boolean)
        input_mask, output_mask = self.compute_masks(text_lengths, mel_lengths)
        # B x D_embed x T_in_max
        embedded_inputs = self.embedding(text).transpose(1, 2)
        # B x T_in_max x D_en
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        if self.gst and self.use_gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, mel_specs)

        if self.use_speaker_embedding or self.use_d_vector_file:
            if not self.use_d_vector_file:
                # B x 1 x speaker_embed_dim
                embedded_speakers = self.speaker_embedding(aux_input["speaker_ids"])[:, None]
            else:
                # B x 1 x speaker_embed_dim
                embedded_speakers = torch.unsqueeze(aux_input["d_vectors"], 1)
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, embedded_speakers)

        # capacitron
        if self.capacitron_vae and self.use_capacitron_vae:
            # B x capacitron_VAE_embedding_dim
            encoder_outputs, *capacitron_vae_outputs = self.compute_capacitron_VAE_embedding(
                encoder_outputs,
                reference_mel_info=[mel_specs, mel_lengths],
                text_info=[embedded_inputs.transpose(1, 2), text_lengths]
                if self.capacitron_vae.capacitron_use_text_summary_embeddings
                else None,
                speaker_embedding=embedded_speakers if self.capacitron_vae.capacitron_use_speaker_embedding else None,
            )
        else:
            capacitron_vae_outputs = None

        encoder_outputs = encoder_outputs * input_mask.unsqueeze(2).expand_as(encoder_outputs)

        # B x mel_dim x T_out -- B x T_out//r x T_in -- B x T_out//r
        decoder_outputs, alignments, stop_tokens = self.decoder(encoder_outputs, mel_specs, input_mask)
        # sequence masking
        if mel_lengths is not None:
            decoder_outputs = decoder_outputs * output_mask.unsqueeze(1).expand_as(decoder_outputs)
        # B x mel_dim x T_out
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        # sequence masking
        if output_mask is not None:
            postnet_outputs = postnet_outputs * output_mask.unsqueeze(1).expand_as(postnet_outputs)
        # B x T_out x mel_dim -- B x T_out x mel_dim -- B x T_out//r x T_in
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(decoder_outputs, postnet_outputs, alignments)
        #NEW INFERENCE USING VOCODER#
        vocoder_input = postnet_outputs.permute(0, 2, 1)
        # print("POSTENET OUTPUTS: ", postnet_outputs.shape)
        vocoder_output = self.vocoder.inference(vocoder_input)
        print(f"{vocoder_output.shape=}")
        spk_embedding2_output = self.spk_embedding(vocoder_output)

        outputs["spk_emb2"] = spk_embedding2_output
        #####
        if self.bidirectional_decoder:
            decoder_outputs_backward, alignments_backward = self._backward_pass(mel_specs, encoder_outputs, input_mask)
            outputs["alignments_backward"] = alignments_backward
            outputs["decoder_outputs_backward"] = decoder_outputs_backward
        if self.double_decoder_consistency:
            decoder_outputs_backward, alignments_backward = self._coarse_decoder_pass(
                mel_specs, encoder_outputs, alignments, input_mask
            )
            outputs["alignments_backward"] = alignments_backward
            outputs["decoder_outputs_backward"] = decoder_outputs_backward
        outputs.update(
            {
                "model_outputs": postnet_outputs,
                "decoder_outputs": decoder_outputs,
                "alignments": alignments,
                "stop_tokens": stop_tokens,
                "capacitron_vae_outputs": capacitron_vae_outputs,
                "spk_emb2": spk_embedding2_output,
            }
        )
        ########
        # print(f"{outputs['model_outputs'].shape=}, {outputs['decoder_outputs'].shape=}, {outputs['alignments'].shape=}, {outputs['stop_tokens'].shape=}")
        ########
        return outputs

    @torch.no_grad()
    def inference(self, text, aux_input=None):
        """Forward pass for inference with no Teacher-Forcing.

        Shapes:
           text: :math:`[B, T_in]`
           text_lengths: :math:`[B]`
        """
        aux_input = self._format_aux_input(aux_input)
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        if self.gst and self.use_gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, aux_input["style_mel"], aux_input["d_vectors"])

        if self.capacitron_vae and self.use_capacitron_vae:
            if aux_input["style_text"] is not None:
                style_text_embedding = self.embedding(aux_input["style_text"])
                style_text_length = torch.tensor([style_text_embedding.size(1)], dtype=torch.int64).to(
                    encoder_outputs.device
                )  # pylint: disable=not-callable
            reference_mel_length = (
                torch.tensor([aux_input["style_mel"].size(1)], dtype=torch.int64).to(encoder_outputs.device)
                if aux_input["style_mel"] is not None
                else None
            )  # pylint: disable=not-callable
            # B x capacitron_VAE_embedding_dim
            encoder_outputs, *_ = self.compute_capacitron_VAE_embedding(
                encoder_outputs,
                reference_mel_info=[aux_input["style_mel"], reference_mel_length]
                if aux_input["style_mel"] is not None
                else None,
                text_info=[style_text_embedding, style_text_length] if aux_input["style_text"] is not None else None,
                speaker_embedding=aux_input["d_vectors"]
                if self.capacitron_vae.capacitron_use_speaker_embedding
                else None,
            )

        if self.num_speakers > 1:
            if not self.use_d_vector_file:
                embedded_speakers = self.speaker_embedding(aux_input["speaker_ids"])[None]
                # reshape embedded_speakers
                if embedded_speakers.ndim == 1:
                    embedded_speakers = embedded_speakers[None, None, :]
                elif embedded_speakers.ndim == 2:
                    embedded_speakers = embedded_speakers[None, :]
            else:
                embedded_speakers = aux_input["d_vectors"]

            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, embedded_speakers)

        decoder_outputs, alignments, stop_tokens = self.decoder.inference(encoder_outputs)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(decoder_outputs, postnet_outputs, alignments)
        #NEW INFERENCE USING VOCODER#
        postnet_outputs = postnet_outputs.permute(0, 2, 1)
        # print("POSTENET OUTPUTS: ", postnet_outputs.shape)
        postnet_outputs = self.vocoder.inference(postnet_outputs)
        torchaudio.save("output.wav", postnet_outputs, 22050)
        #####
        outputs = {
            "model_outputs": postnet_outputs,
            "decoder_outputs": decoder_outputs,
            "alignments": alignments,
            "stop_tokens": stop_tokens,
        }
        return outputs

    def before_backward_pass(self, loss_dict, optimizer) -> None:
        # Extracting custom training specific operations for capacitron
        # from the trainer
        if self.use_capacitron_vae:
            loss_dict["capacitron_vae_beta_loss"].backward()
            optimizer.first_step()

    def train_step(self, batch: Dict, criterion: torch.nn.Module):
        """A single training step. Forward pass and loss computation.

        Args:
            batch ([Dict]): A dictionary of input tensors.
            criterion ([type]): Callable criterion to compute model loss.
        """
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        stop_targets = batch["stop_targets"]
        stop_target_lengths = batch["stop_target_lengths"]
        speaker_ids = batch["speaker_ids"]
        d_vectors = batch["d_vectors"]
        #THIS IS NEW#
        raw_audio = batch["spk_emb"]
        #####
        aux_input = {"speaker_ids": speaker_ids, "d_vectors": d_vectors}
        outputs = self.forward(text_input, text_lengths, mel_input, mel_lengths, aux_input, raw_audio)

        # set the [alignment] lengths wrt reduction factor for guided attention
        if mel_lengths.max() % self.decoder.r != 0:
            alignment_lengths = (
                mel_lengths + (self.decoder.r - (mel_lengths.max() % self.decoder.r))
            ) // self.decoder.r
        else:
            alignment_lengths = mel_lengths // self.decoder.r

        # compute loss
        with autocast(enabled=False):  # use float32 for the criterion
            loss_dict = criterion(
                outputs["model_outputs"].float(),
                outputs["decoder_outputs"].float(),
                mel_input.float(),
                None,
                outputs["stop_tokens"].float(),
                stop_targets.float(),
                stop_target_lengths,
                outputs["capacitron_vae_outputs"] if self.capacitron_vae else None,
                mel_lengths,
                None if outputs["decoder_outputs_backward"] is None else outputs["decoder_outputs_backward"].float(),
                outputs["alignments"].float(),
                alignment_lengths,
                None if outputs["alignments_backward"] is None else outputs["alignments_backward"].float(),
                text_lengths,
            )

        # compute alignment error (the lower the better )
        align_error = 1 - alignment_diagonal_score(outputs["alignments"])
        loss_dict["align_error"] = align_error
        return outputs, loss_dict

    def get_optimizer(self) -> List:
        if self.use_capacitron_vae:
            return CapacitronOptimizer(self.config, self.named_parameters())
        return get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr, self)

    def get_scheduler(self, optimizer: object):
        opt = optimizer.primary_optimizer if self.use_capacitron_vae else optimizer
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, opt)

    def before_gradient_clipping(self):
        if self.use_capacitron_vae:
            # Capacitron model specific gradient clipping
            model_params_to_clip = []
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if name != "capacitron_vae_layer.beta":
                        model_params_to_clip.append(param)
            torch.nn.utils.clip_grad_norm_(model_params_to_clip, self.capacitron_vae.capacitron_grad_clip)

    def _create_logs(self, batch, outputs, ap):
        """Create dashboard log information."""
        postnet_outputs = outputs["model_outputs"]
        alignments = outputs["alignments"]
        alignments_backward = outputs["alignments_backward"]
        mel_input = batch["mel_input"]

        pred_spec = postnet_outputs[0].data.cpu().numpy()
        gt_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()

        figures = {
            "prediction": plot_spectrogram(pred_spec, ap, output_fig=False),
            "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
        }

        if self.bidirectional_decoder or self.double_decoder_consistency:
            figures["alignment_backward"] = plot_alignment(alignments_backward[0].data.cpu().numpy(), output_fig=False)

        # Sample audio
        audio = ap.inv_melspectrogram(pred_spec.T)
        return figures, {"audio": audio}

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ) -> None:  # pylint: disable=no-self-use
        """Log training progress."""
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    def eval_step(self, batch: dict, criterion: nn.Module):
        return self.train_step(batch, criterion)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    @staticmethod
    def init_from_config(config: "Tacotron2Config", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (Tacotron2Config): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        ap = AudioProcessor.init_from_config(config)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(new_config, samples)
        return Tacotron2(new_config, ap, tokenizer, speaker_manager)

        energy *= self.std
        energy += self.mean
        energy[zero_idxs] = 0.0
        return energy

    def compute_or_load(self, wav_file, audio_unique_name):
        """
        compute energy and return a numpy array of energy values
        """
        energy_file = self.create_energy_file_path(audio_unique_name, self.cache_path)
        if not os.path.exists(energy_file):
            energy = self._compute_and_save_energy(self.ap, wav_file, energy_file)
        else:
            energy = np.load(energy_file)
        return energy.astype(np.float32)

    def collate_fn(self, batch):
        audio_unique_name = [item["audio_unique_name"] for item in batch]
        energys = [item["energy"] for item in batch]
        energy_lens = [len(item["energy"]) for item in batch]
        energy_lens_max = max(energy_lens)
        energys_torch = torch.LongTensor(len(energys), energy_lens_max).fill_(self.get_pad_id())
        for i, energy_len in enumerate(energy_lens):
            energys_torch[i, :energy_len] = torch.LongTensor(energys[i])
        return {"audio_unique_name": audio_unique_name, "energy": energys_torch, "energy_lens": energy_lens}

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print("\n")
        print(f"{indent}> energyDataset ")
        print(f"{indent}| > Number of instances : {len(self.samples)}")
