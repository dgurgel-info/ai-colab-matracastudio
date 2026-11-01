"""Divisão do roteiro em partes endereçáveis (sentenças inteiras, tamanho controlado)."""

from __future__ import annotations

import re
from typing import List

# Termina uma sentença: pontuação final (+ aspas/parêntese de fechamento) seguida de espaço,
# ou terminadores CJK, que não usam espaço.
_SENT_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+|(?<=[。！？])\s*|\n+")
_ABBREV = {
    "sr", "sra", "srta", "dr", "dra", "prof", "profa", "eng", "ex", "etc", "obs", "av", "tel",
    "mr", "mrs", "ms", "st", "vs", "jr", "no", "fig", "cap", "pág", "pag", "art", "inc", "ltd",
    "mt", "gen", "col", "sgt", "u.s", "e.g", "i.e",
}
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:—–])\s+")


def _ends_with_abbreviation(piece: str) -> bool:
    m = re.search(r"([\w.]+)\.$", piece.strip())
    if not m:
        return False
    token = m.group(1).lower()
    return token in _ABBREV or bool(re.fullmatch(r"[a-zà-ú]", token)) or bool(re.fullmatch(r"\d+", token))


def split_sentences(text: str) -> List[str]:
    """Sentenças em ordem, sem separar abreviações comuns, iniciais ou numeração (``3.``)."""
    text = re.sub(r"[ \t\r\f\v]+", " ", (text or "").strip())
    raw = [s.strip() for s in _SENT_END.split(text) if s and s.strip()]
    merged: List[str] = []
    for piece in raw:
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _split_long(sentence: str, max_chars: int) -> List[str]:
    """Quebra uma sentença maior que ``max_chars`` em orações e, por fim, em palavras."""
    if len(sentence) <= max_chars:
        return [sentence]
    pieces: List[str] = []
    current = ""
    for clause in _CLAUSE_SPLIT.split(sentence):
        if len(clause) > max_chars:  # oração ainda grande: quebra por palavras
            if current:
                pieces.append(current)
                current = ""
            line = ""
            for word in clause.split():
                if line and len(line) + 1 + len(word) > max_chars:
                    pieces.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            current = line
        elif current and len(current) + 1 + len(clause) > max_chars:
            pieces.append(current)
            current = clause
        else:
            current = f"{current} {clause}".strip()
    if current:
        pieces.append(current)
    return pieces


def split_into_parts(text: str, target_chars: int = 600, max_chars: int = 1500) -> List[str]:
    """Agrupa sentenças inteiras em partes de ~``target_chars`` (nunca acima de ``max_chars``).

    Cada parte começa e termina em fronteira de sentença sempre que possível, para que
    trocar uma parte não corte o ritmo da frase vizinha.
    """
    target_chars = max(40, int(target_chars))
    max_chars = max(target_chars, int(max_chars))
    sentences: List[str] = []
    for s in split_sentences(text):
        sentences.extend(_split_long(s, max_chars))

    parts: List[str] = []
    current = ""
    for s in sentences:
        if not current:
            current = s
        elif len(current) + 1 + len(s) <= target_chars:
            current = f"{current} {s}"
        else:
            parts.append(current)
            current = s
    if current:
        parts.append(current)
    # Evita uma última parte minúscula (soaria como remendo): funde com a anterior se couber.
    if len(parts) >= 2 and len(parts[-1]) < target_chars * 0.25 and len(parts[-2]) + 1 + len(parts[-1]) <= max_chars:
        parts[-2] = f"{parts[-2]} {parts[-1]}"
        parts.pop()
    return parts
