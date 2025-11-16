# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron tokenizers."""

from abc import ABC
from abc import abstractmethod

from tokenizers import Tokenizer
from rwkv_tokenizer import RWKV_TOKENIZER, TRIE_TOKENIZER
import pyrwkv_tokenizer  # The fast, Rust-based pip package
import os                # For path manipulation

from typing import List, Union


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print("> building {} tokenizer ...".format(args.tokenizer_type), flush=True)

    # Select and instantiate the tokenizer.

    if args.tokenizer_type.lower() == "HFTokenizer".lower():
        assert args.vocab_file is not None
        tokenizer = HFTokenizer(args.vocab_file)
    elif args.tokenizer_type.lower() == "PyRWKVTokenizer".lower():
        tokenizer = PyRWKVTokenizer(args.vocab_file)
    elif args.tokenizer_type.lower() == "RWKVTokenizer".lower():
        assert args.vocab_file is not None
        tokenizer = RWKVTokenizer(args.vocab_file)
    else:
        raise NotImplementedError(
            "{} tokenizer is not " "implemented.".format(args.tokenizer_type)
        )

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(
            " > padded vocab (size: {}) with {} dummy tokens "
            "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after),
            flush=True,
        )
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError(
            "detokenizer is not implemented for {} " "tokenizer".format(self.name)
        )

    @property
    def cls(self):
        raise NotImplementedError(
            "CLS is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def sep(self):
        raise NotImplementedError(
            "SEP is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def pad(self):
        raise NotImplementedError(
            "PAD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def eod(self):
        raise NotImplementedError(
            "EOD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def mask(self):
        raise NotImplementedError(
            "MASK is not provided for {} " "tokenizer".format(self.name)
        )


class HFTokenizer(AbstractTokenizer):
    """Designed to Integrate HF's Tokenizer library."""

    def __init__(self, vocab_file):
        name = "HFTokenizer"
        super().__init__(name)

        self.tokenizer = Tokenizer.from_file(vocab_file)
        self.eod_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_id = self.tokenizer.token_to_id("<|padding|>")

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text: str):
        return self.tokenizer.encode(text).ids

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        return self.tokenizer.encode_batch(text_batch)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id


class RWKVTokenizer(AbstractTokenizer):
    """RWKV Worlds Tokenizer."""

    def __init__(self, vocab_file='rwkv_vocab_v20230424.txt'):
        name = "RWKVTokenizer"
        super().__init__(name)

        self.tokenizer = TRIE_TOKENIZER(vocab_file)
        self.eod_id = 0  # self.tokenizer.token_to_id("<|endoftext|>")
        # self.pad_id = self.tokenizer.token_to_id("<|padding|>")

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        return self.tokenizer.decode

    def tokenize(self, text: str):
        return self.tokenizer.encode(text)

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        return self.tokenizer.encode_batch(text_batch)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

class PyRWKVTokenizer(AbstractTokenizer):
    """
    RWKV Worlds Tokenizer.
    This class is now a wrapper for the fast 'pyrwkv-tokenizer' (Rust) package.
    """

    def __init__(self, vocab_file=None):
        name = "WorldTokenizer"
        super().__init__(name)

        # print(f"--- [RWKVTokenizer] Loading fast pyrwkv-tokenizer from: {vocab_dir} ---")

        # Instantiate the fast tokenizer from the pip package
        # This requires 'rwkv_vocab_v20230424.txt' to be in that directory
        try:
            self.tokenizer = pyrwkv_tokenizer.RWKVTokenizer(name=name, vocab_filepath=vocab_file)
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Failed to load pyrwkv_tokenizer from file: {vocab_file}")
            print("And ensure you have installed the package: pip install pyrwkv-tokenizer")
            print(f"Error: {e}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            raise e

        # Manually set the EOD token ID
        self.eod_id = 0

    @property
    def vocab_size(self):
        # pyrwkv-tokenizer doesn't expose .get_vocab_size()
        # The World tokenizer vocab size is fixed at 65536
        return 65536

    @property
    def vocab(self):
        # This is not easily exposed by pyrwkv-tokenizer and not
        # needed by preprocess_data.py.
        raise NotImplementedError("pyrwkv-tokenizer does not expose .vocab")

    @property
    def inv_vocab(self):
        # Pass the .decode method
        return self.tokenizer.decode

    def tokenize(self, text: str):
        # Map the script's .tokenize() call to the library's .encode() method
        return self.tokenizer.encode(text)

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        return self.tokenizer.encode_batch(text_batch)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        # preprocess_data.py needs this to append EOD tokens
        return self.eod_id
