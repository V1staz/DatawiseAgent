"""Heuristic question-contract parsing for InfiAgentBench-style records."""

from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any

from .schemas import DataCard, QuestionContract, TargetAnswer

_FORMAT_RE = re.compile(r"@(\w+)\[([^\]]*)\]")
_QUOTED_RE = re.compile(r"['\"]([^'\"]{1,120})['\"]")

_METHOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "mean": ("mean", "average"),
    "median": ("median",),
    "mode": ("mode",),
    "std": ("standard deviation", " std", "std "),
    "population_std": ("population standard deviation", "population std"),
    "variance": ("variance",),
    "sum": ("sum", "total"),
    "count": ("count", "number of", "how many"),
    "min": ("minimum", "min"),
    "max": ("maximum", "max"),
    "pearson": ("pearson",),
    "spearman": ("spearman",),
    "correlation": ("correlation",),
    "iqr": ("iqr", "interquartile"),
    "z_score": ("z-score", "z score"),
    "shapiro_wilk": ("shapiro",),
    "t_test": ("t-test", " t test"),
    "chi_square": ("chi-square", "chi square"),
    "train_test_split": ("train/test", "train test", "random_state"),
}

_PREPROCESSING_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"drop\s+(?:rows\s+)?(?:with\s+)?missing", "drop_missing"),
    (r"remove\s+(?:rows\s+)?(?:with\s+)?missing", "drop_missing"),
    (r"fill\s+(?:missing|nan)", "impute_missing"),
    (r"imput", "impute_missing"),
    (r"standardi[sz]", "standardize"),
    (r"normali[sz]", "normalize"),
    (r"one[- ]hot", "one_hot_encode"),
    (r"label encod", "label_encode"),
    (r"convert .*datetime|datetime", "convert_datetime"),
    (r"filter", "filter_rows"),
)


def extract_format_targets(
    format_text: str | None,
    include_conditional: bool = True,
) -> list[TargetAnswer]:
    if not format_text:
        return []
    targets: list[TargetAnswer] = []
    for match in _FORMAT_RE.finditer(format_text):
        name, placeholder = match.groups()
        if not include_conditional and _is_conditional_alternative_target(
            format_text, match.start()
        ):
            continue
        local_text = _target_context(format_text, name, placeholder)
        rounding = _extract_rounding_for_target(
            format_text, name, placeholder, local_text
        )
        type_hint = _infer_answer_type(name, placeholder, local_text or format_text.lower())
        if type_hint in {"integer", "string", "boolean", "list", "dict"}:
            rounding = None
        targets.append(
            TargetAnswer(
                name=name,
                type=type_hint,
                rounding=rounding,
                format=f"@{name}[{placeholder}]",
                description=(local_text or None),
            )
        )
    if not targets and not include_conditional:
        return extract_format_targets(format_text, include_conditional=True)
    return targets


def _is_conditional_alternative_target(format_text: str, target_start: int) -> bool:
    """Heuristically detect @targets that belong to an alternative `if ... output` branch."""

    prefix = format_text[max(0, target_start - 180) : target_start].lower()
    return bool(
        re.search(
            r"\bif\b[\s\S]{0,150}\b(?:output|report|return|print)\s*$",
            prefix,
        )
        or re.search(
            r"\bif\b[\s\S]{0,150}\b(?:simply|please)\s+(?:output|report|return|print)\s*$",
            prefix,
        )
    )


def _target_context(text: str, name: str, placeholder: str, window: int = 220) -> str:
    """Return text most likely to describe one answer target.

    InfiAgentBench formats often list all @answers first and then list per-answer
    "where ..." clauses. A naive window after @answer can therefore steal the
    rounding rule of the next answer. This context keeps the local target line
    and augments it with clauses that explicitly mention the answer name or
    placeholder.
    """

    pieces: list[str] = []
    target_piece = _target_format_piece(text, name, window=window)
    if target_piece:
        pieces.append(target_piece)

    pieces.extend(_identifier_description_snippets(text, [placeholder, name]))

    return "\n".join(piece.strip() for piece in pieces if piece.strip())


def _split_clauses(text: str) -> list[str]:
    # Split target description clauses, but do not split before generic
    # follow-up sentences such as `Rounding off ...` because those often belong
    # to the previous target.
    coarse = [part.strip() for part in re.split(r"[\n;]+", text) if part.strip()]
    clauses: list[str] = []
    sentence_boundary = re.compile(
        r"(?<=[.!?])\s+(?=(?:where\s+)?['\"]|(?:where\s+)?\w+\s+(?:is|are)\b|@)",
        flags=re.IGNORECASE,
    )
    for part in coarse:
        clauses.extend(piece.strip() for piece in sentence_boundary.split(part) if piece.strip())
    return clauses


def _extract_rounding(text: str) -> int | None:
    lowered = text.lower()
    patterns = (
        r"round(?:ed|ing)?(?:\s+off)?(?:\s+the\s+answer)?\s+to\s+(\d+)\s+decimal",
        r"rounded\s+to\s+(\d+)\s+decimal",
        r"to\s+(\d+)\s+decimal\s+places",
        r"with\s+(\d+)\s+decimals?",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    word_to_int = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    match = re.search(r"(?:to|with)\s+(one|two|three|four|five|six)\s+decimals?", lowered)
    if match:
        return word_to_int[match.group(1)]
    return None


def _extract_all_roundings(text: str) -> list[int]:
    values: list[int] = []
    for clause in _split_clauses(text):
        value = _extract_rounding(clause)
        if value is not None:
            values.append(value)
    return values


def _extract_rounding_for_target(
    format_text: str,
    name: str,
    placeholder: str,
    local_text: str,
) -> int | None:
    """Infer rounding for one output target without leaking adjacent targets."""

    # The placeholder itself may contain the rounding instruction, e.g.
    # `@log_MEANJZH[First entry ..., rounded to three decimal places]`.
    value = _extract_rounding(f"@{name}[{placeholder}]")
    if value is not None:
        return value

    for snippet in _identifier_description_snippets(format_text, [placeholder, name]):
        value = _extract_rounding(snippet)
        if value is not None:
            return value

    local_roundings = _extract_all_roundings(local_text)
    if len(set(local_roundings)) == 1 and local_roundings:
        return local_roundings[0]

    all_roundings = _extract_all_roundings(format_text)
    unique_roundings = set(all_roundings)
    if len(unique_roundings) == 1:
        return all_roundings[0]
    return None


def _infer_answer_type(name: str, placeholder: str, text: str) -> str:
    name_and_placeholder = " ".join([name, placeholder]).lower()
    specific_text = _target_specific_type_text(name, placeholder, text)
    if specific_text:
        type_hint = _classify_answer_type(
            " ".join([name_and_placeholder, specific_text]).lower()
        )
        if type_hint != "unknown":
            return type_hint

    type_hint = _classify_answer_type(name_and_placeholder, use_name_hints=True)
    if type_hint != "unknown":
        return type_hint
    if re.search(r"['\"][^'\"]+['\"]", placeholder):
        return "string"
    return _classify_answer_type(" ".join([name, placeholder, text]).lower())


def _target_specific_type_text(name: str, placeholder: str, text: str) -> str:
    """Return the phrase that explicitly describes one target's value type."""

    return " ".join(_identifier_description_snippets(text, [placeholder, name]))


def _classify_answer_type(lowered: str, use_name_hints: bool = False) -> str:
    if re.search(r"\b(list|array)\b", lowered):
        return "list"
    if re.search(r"\b(dict|mapping)\b", lowered):
        return "dict"
    if (
        re.search(r"\b(bool|boolean)\b", lowered)
        or re.search(r"\btrue\b[\s\S]{0,40}\bfalse\b", lowered)
        or re.search(r"\bfalse\b[\s\S]{0,40}\btrue\b", lowered)
        or (use_name_hints and re.search(r"(?:^|_)(is|has|was|were|can)_", lowered))
    ):
        return "boolean"
    # Prefer explicit string/category/name descriptions before generic numeric
    # words that may appear in neighbouring examples or significance rules.
    if re.search(r"\b(string|category|name)\b", lowered):
        return "string"
    if re.search(r"\b(integer|int)\b", lowered) or "number of" in lowered:
        return "integer"
    if use_name_hints and re.search(r"(?:^|_)num(?:_|$)", lowered):
        return "integer"
    if (
        re.search(
            r"\b(floats?|floating[- ]?point|decimal|numeric|coefficient|statistic|mean|average|avg|median|std|stddev|standard deviation|standard_deviation|rate|accuracy|score|mse|rmse|skewness|kurtosis)\b",
            lowered,
        )
        or re.search(r"\bp[_ -]?value\b|\bpvalue\b", lowered)
        or (
            use_name_hints
            and re.search(
                r"(?:^|_)(mean|average|avg|median|std|stddev|standard|deviation|rate|accuracy|score|coefficient|statistic|mse|rmse|p[_]?value|pvalue)(?:_|$)",
                lowered,
            )
        )
        or ("number" in lowered and "number of" not in lowered)
    ):
        return "float"
    if re.search(r"\bcount\b", lowered):
        return "integer"
    if use_name_hints and re.search(r"(?:^|_)count(?:_|$)", lowered):
        return "integer"
    if re.search(r"\b(label)\b", lowered):
        return "string"
    return "unknown"


def _target_format_piece(text: str, name: str, window: int = 220) -> str | None:
    idx = text.find("@" + name)
    if idx < 0:
        return None
    next_match = _FORMAT_RE.search(text, idx + 1)
    end = next_match.start() if next_match else min(len(text), idx + window)
    return text[idx:end].strip()


def _clean_identifier(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().strip("\"'")


def _identifier_description_snippets(text: str, identifiers: list[str]) -> list[str]:
    """Return description snippets that explicitly belong to identifiers.

    This deliberately ignores the leading `@a[x] @b[y]` target list and only
    captures clauses like `"x" is ...`, `x is ...`, or grouped descriptions such
    as `"x", "y" are floats ...`. That avoids leaking the first target's
    rounding/type into later targets.
    """

    wanted = {
        _clean_identifier(identifier).lower()
        for identifier in identifiers
        if _clean_identifier(identifier)
    }
    if not wanted:
        return []

    snippets: list[str] = []
    seen: set[str] = set()
    for clause in _split_clauses(text):
        clause = clause.strip()
        if not clause:
            continue

        quoted_group = re.finditer(
            r"(?P<ids>(?:['\"][^'\"]+['\"]\s*(?:,\s*and\s*|,\s*|\s+and\s+)?\s*){1,8})\s+"
            r"(?P<verb>is|are)\s+(?:an?|the)?\s*(?P<desc>[^;\n]{0,260})",
            clause,
            flags=re.IGNORECASE,
        )
        for match in quoted_group:
            group_ids = {_clean_identifier(item).lower() for item in _QUOTED_RE.findall(match.group("ids"))}
            if wanted & group_ids:
                desc = _trim_adjacent_target_phrase(match.group("desc"))
                snippet = f"{match.group('ids').strip()} {match.group('verb')} {desc.strip()}"
                if snippet and snippet not in seen:
                    seen.add(snippet)
                    snippets.append(snippet)

        for identifier in wanted:
            escaped = re.escape(identifier)
            direct_patterns = (
                rf"['\"]{escaped}['\"]\s+(?:is|are)\s+(?:an?|the)?\s*([^;\n]{{0,260}})",
                rf"\b{escaped}\b\s+(?:is|are)\s+(?:an?|the)?\s*([^;\n]{{0,260}})",
                rf"['\"]{escaped}['\"]\s+should\s+be\s+(?:an?|the)?\s*([^;\n]{{0,260}})",
                rf"\b{escaped}\b\s+should\s+be\s+(?:an?|the)?\s*([^;\n]{{0,260}})",
            )
            for pattern in direct_patterns:
                match = re.search(pattern, clause, flags=re.IGNORECASE)
                if match:
                    full = match.group(0)
                    desc = _trim_adjacent_target_phrase(match.group(1))
                    snippet = full[: full.find(match.group(1))] + desc
                    snippet = snippet.strip()
                    if snippet and snippet not in seen:
                        seen.add(snippet)
                        snippets.append(snippet)
    return snippets


def _trim_adjacent_target_phrase(text: str) -> str:
    """Trim a matched `x is ...` phrase before the next target description."""

    return re.split(
        r"(?:\s+(?:and|while)\s+|\s*,\s*)(?:['\"][^'\"]+['\"]|\w+)\s+(?:is|are)\s+",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _extract_required_columns(text: str, data_card: DataCard | None) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    normalized_text = _normalize(text)

    if data_card is not None:
        for col in data_card.columns:
            norm = _normalize(col.name)
            if norm and norm in normalized_text:
                seen.add(col.name)
                columns.append(col.name)

    quoted = _QUOTED_RE.findall(text)
    if data_card is not None:
        known = {_normalize(col.name): col.name for col in data_card.columns}
        for item in quoted:
            mapped = known.get(_normalize(item))
            if mapped and mapped not in seen:
                seen.add(mapped)
                columns.append(mapped)
    else:
        for item in quoted:
            if item not in seen:
                seen.add(item)
                columns.append(item)
    return columns


def _extract_methods(text: str) -> list[str]:
    lowered = text.lower()
    methods: list[str] = []
    for method, keywords in _METHOD_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            methods.append(method)
    if "standard deviation" in lowered and "population" in lowered:
        methods = [m for m in methods if m != "std"] + ["population_std"]
    return sorted(set(methods))


def _extract_preprocessing(text: str) -> list[str]:
    lowered = text.lower()
    steps: list[str] = []
    for pattern, step in _PREPROCESSING_PATTERNS:
        if re.search(pattern, lowered):
            steps.append(step)
    if "do not perform" in lowered and ("clean" in lowered or "preprocess" in lowered):
        steps.append("no_extra_preprocessing")
    return sorted(set(steps))


def _extract_groupby(text: str, columns: list[str]) -> list[str]:
    lowered = text.lower()
    group_cols: list[str] = []
    for col in columns:
        col_lower = col.lower()
        if (
            f"for each {col_lower}" in lowered
            or f"by {col_lower}" in lowered
            or f"grouped by {col_lower}" in lowered
            or f"group by {col_lower}" in lowered
        ):
            group_cols.append(col)
    return group_cols


def _extract_filters(text: str) -> list[str]:
    filters: list[str] = []
    for line in re.split(r"[\n.;]", text):
        lowered = line.lower().strip()
        if any(token in lowered for token in ["only ", "filter", "where ", "select rows", "exclude"]):
            filters.append(line.strip())
    return filters[:8]


class ContractParser:
    """Parse a benchmark question into an auditable contract."""

    def parse(self, question: dict[str, Any], data_card: DataCard | None = None) -> QuestionContract:
        text_parts = [
            str(question.get("question", "")),
            str(question.get("constraints", "")),
            str(question.get("format", "")),
        ]
        text = "\n".join(part for part in text_parts if part)
        columns = _extract_required_columns(text, data_card)
        return QuestionContract(
            question_id=question.get("id"),
            question=str(question.get("question", "")),
            level=question.get("level"),
            concepts=list(question.get("concepts", []) or []),
            target_answers=extract_format_targets(
                question.get("format"),
                include_conditional=False,
            ),
            required_columns=columns,
            filters=_extract_filters(text),
            groupby=_extract_groupby(text, columns),
            methods=_extract_methods(text),
            preprocessing_steps=_extract_preprocessing(text),
            final_format=question.get("format"),
            raw_constraints=question.get("constraints"),
        )

    def to_prompt(self, contract: QuestionContract) -> str:
        payload = asdict(contract)
        return (
            "# Question Contract\n"
            "Use this contract as the source of truth. If the contract conflicts "
            "with the dataset, resolve column names explicitly before computing.\n"
            f"{payload}\n"
        )
