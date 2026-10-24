"""The Wan UMT5 encoder loader's tokenizer must follow the folder's tokenizer.json."""

import json
from pathlib import Path
from types import SimpleNamespace

from tokenizers import Tokenizer, models, pre_tokenizers

from invokeai.backend.model_manager.load.model_loaders.wan import WanT5EncoderLoader
from invokeai.backend.model_manager.taxonomy import SubModelType


def test_tokenizer_follows_folder_tokenizer_json(tmp_path: Path) -> None:
    # Same special-token layout as the Wan tokenizer: id 2 is <s> and <unk> is id 3.
    vocab = [("<pad>", 0.0), ("</s>", 0.0), ("<s>", 0.0), ("<unk>", 0.0), ("▁music", -1.0), ("▁a", -2.0), ("▁", -3.0)]
    expected = Tokenizer(models.Unigram(vocab, unk_id=3))
    expected.pre_tokenizer = pre_tokenizers.Metaspace()
    folder = tmp_path / "tokenizer"
    folder.mkdir()
    expected.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "T5Tokenizer", "pad_token": "<pad>", "eos_token": "</s>", "unk_token": "<unk>"})
    )

    loader = object.__new__(WanT5EncoderLoader)
    tokenizer = loader._load_model(SimpleNamespace(path=str(tmp_path)), SubModelType.Tokenizer)  # type: ignore[arg-type]

    text = "a \U0001d11e music"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert ids == expected.encode(text, add_special_tokens=False).ids
    assert 3 in ids and 2 not in ids
    assert (tokenizer.pad_token, tokenizer.eos_token, tokenizer.unk_token) == ("<pad>", "</s>", "<unk>")
