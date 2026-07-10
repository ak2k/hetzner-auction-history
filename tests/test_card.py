"""Tests for dataset-card generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import yaml  # transitive dep of huggingface_hub

from hetzner_auction_history import card

if TYPE_CHECKING:
    from pathlib import Path


def _frontmatter(text: str) -> dict[str, Any]:
    assert text.startswith("---\n")
    _, fm, _body = text.split("---\n", 2)
    return cast("dict[str, Any]", yaml.safe_load(fm))


def test_frontmatter_is_valid_yaml_with_expected_keys():
    text = card.render_card("me/ds")
    fm = _frontmatter(text)

    assert fm["license"] == "cc-by-4.0"
    assert fm["pretty_name"]
    assert "hetzner" in fm["tags"]

    configs = fm["configs"]
    assert len(configs) == 1
    assert configs[0]["config_name"] == "default"
    assert configs[0]["data_files"][0]["path"] == "data/*.parquet"
    assert configs[0]["data_files"][0]["split"] == "train"


def test_body_has_vat_note_and_radar_attribution():
    text = card.render_card("me/ds")
    assert "1.19" in text  # VAT divisor
    assert "19% German VAT" in text
    # Prominent Server Radar attribution + links.
    assert "Server Radar" in text
    assert "https://radar.iodev.org/" in text
    assert "https://github.com/elsbrock/hetzner-radar" in text
    assert "Not affiliated" in text


def test_repo_id_is_substituted_everywhere():
    text = card.render_card("trader298/hetzner-auction-history")
    assert "hf://datasets/trader298/hetzner-auction-history/data/" in text
    assert 'load_dataset("trader298/hetzner-auction-history"' in text
    assert "{repo_id}" not in text  # no unsubstituted placeholders


def test_write_card_writes_readme(tmp_path: Path):
    out = card.write_card("me/ds", tmp_path)
    assert out == tmp_path / "README.md"
    assert out.read_text(encoding="utf-8").startswith("---")
