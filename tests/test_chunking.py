from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_core.rag.chunking import ChunkingConfig, TextChunker


def test_config_defaults() -> None:
    config = ChunkingConfig()
    assert config.chunk_size == 600
    assert config.overlap == 80


def test_config_rejects_overlap_not_smaller_than_chunk() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=10, overlap=10)
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=5, overlap=6)


def test_chunk_empty_text() -> None:
    assert TextChunker().chunk("") == []
    assert TextChunker().chunk("   ") == []


def test_chunk_single_window() -> None:
    chunker = TextChunker(ChunkingConfig(chunk_size=10, overlap=2))
    assert chunker.chunk("um dois três") == ["um dois três"]


def test_chunk_multiple_windows_with_overlap() -> None:
    chunker = TextChunker(ChunkingConfig(chunk_size=3, overlap=1))
    text = "a b c d e f"
    assert chunker.chunk(text) == ["a b c", "c d e", "e f"]


def test_chunk_without_overlap() -> None:
    chunker = TextChunker(ChunkingConfig(chunk_size=2, overlap=0))
    assert chunker.chunk("a b c d") == ["a b", "c d"]


def test_default_chunker_uses_default_config() -> None:
    chunker = TextChunker()
    assert chunker.config == ChunkingConfig()


def test_chunk_does_not_emit_redundant_tail() -> None:
    chunker = TextChunker(ChunkingConfig(chunk_size=10, overlap=2))
    text = " ".join(str(n) for n in range(10))
    assert chunker.chunk(text) == [text]


def test_chunk_last_window_ends_exactly_at_text_end() -> None:
    chunker = TextChunker(ChunkingConfig(chunk_size=3, overlap=1))
    assert chunker.chunk("a b c d e") == ["a b c", "c d e"]


def test_chunk_preserves_line_breaks_and_indentation() -> None:
    chunker = TextChunker(ChunkingConfig(chunk_size=4, overlap=0))
    code = "def f():\n    return 1\n\nprint(f())"
    assert chunker.chunk(code) == ["def f():\n    return 1", "print(f())"]
