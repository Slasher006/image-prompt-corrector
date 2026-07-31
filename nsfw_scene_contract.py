"""Structured fidelity contracts for explicitly requested adult image prompts.

This module contains no GUI or model dependencies.  It turns adult prompt text
and preset entries into compact machine-readable facts that can be shared by
correction, validation, preset compatibility checks, and generated-image review.
"""

from __future__ import annotations

import re
from typing import Iterable


SEXUAL_SIGNAL_PATTERN = re.compile(
    r"\b(?:nsfw|nude|naked|erotic|sexual|sex|intercourse|masturbat\w*|"
    r"oral\s+(?:sex|stimulation)|blowjobs?|handjobs?|anal\s+sex|"
    r"manual(?:ly)?\s+stimulat(?:e|es|ed|ing|ion)|precum|"
    r"(?:grip(?:s|ped|ping)?|hold(?:s|ing)?|held|strok(?:e|es|ed|ing)|"
    r"rub(?:s|bed|bing)?)\b[^.!?;]{0,70}\b(?:penis|cock|"
    r"(?:his|her|their)(?:\s+erect)?\s+shaft|erect\s+shaft)\b|"
    r"penetrat\w*|orgasm\w*|climax\w*|foreplay|"
    r"seduc\w*|intimate|kiss(?:es|ed|ing)?|dildos?|vibrators?|strap[- ]ons?|"
    r"sex\s+toys?|adult\s+toys?|bondage)\b",
    re.IGNORECASE,
)
ROLE_PATTERN_TEXT = (
    r"(?:(?:adult\s+)?(?:woman(?:s|['’]s)?|women|man(?:s|['’]s)?|men|"
    r"female|females|male|males)"
    r"\s+partners?|"
    r"(?:adult\s+)?(?:woman(?:s|['’]s)?|women|man(?:s|['’]s)?|men|"
    r"female|females|male|males|"
    r"nonbinary\s+(?:person|people)|non-binary\s+(?:person|people)|"
    r"partner|partners|lover|lovers|subject|subjects|performer|performers|"
    r"dominant|dominants|submissive|submissives))"
)
ROLE_PATTERN = re.compile(rf"\b{ROLE_PATTERN_TEXT}\b", re.IGNORECASE)

NEGATIVE_CLAUSE_PATTERN = re.compile(
    r"(?i)\b(?:no|without|avoid|exclude|never|do\s+not|don't)\b[^,.!?;\n]*"
)
TOY_TERM_PATTERN_TEXT = (
    r"(?:dildos?|vibrators?|strap[- ]ons?|sex\s+toys?|adult\s+toys?|"
    r"anal\s+toys?|wand\s+massagers?)"
)
TOY_USE_ACTION_PATTERN_TEXT = (
    r"(?:use\w*|insert\w*|penetrat\w*|thrust\w*|masturbat\w*|"
    r"stimulat\w*|guid\w*|press\w*|fuck\w*)"
)
TOY_USE_PATTERN = re.compile(
    rf"(?:\b{TOY_USE_ACTION_PATTERN_TEXT}\b[^.!?;]{{0,100}}\b{TOY_TERM_PATTERN_TEXT}\b|"
    rf"\b{TOY_TERM_PATTERN_TEXT}\b[^.!?;]{{0,100}}\b"
    r"(?:vaginally|anally|inside|into|penetrat\w*|insert\w*|contact)\b|"
    rf"\b{TOY_TERM_PATTERN_TEXT}\b[^.!?;]{{0,35}}\b(?:in|inside)\s+"
    r"(?:(?:her|his|their|the|a)\s+)?"
    r"(?:vagina|vaginal\s+opening|vulva|pussy|anus|anal\s+opening|rectum)\b)",
    re.IGNORECASE,
)
DILDO_USE_PATTERN = re.compile(
    rf"(?:\b{TOY_USE_ACTION_PATTERN_TEXT}\b[^.!?;]{{0,100}}\bdildos?\b|"
    r"\bdildos?\b[^.!?;]{0,100}\b"
    r"(?:vaginally|anally|inside|into|penetrat\w*|insert\w*|contact)\b|"
    r"\bdildos?\b[^.!?;]{0,35}\b(?:in|inside)\s+"
    r"(?:(?:her|his|their|the|a)\s+)?"
    r"(?:vagina|vaginal\s+opening|vulva|pussy|anus|anal\s+opening|rectum)\b)",
    re.IGNORECASE,
)

PENIS_VENTRAL_ORIENTATION_PROMPT = (
    "The penis is oriented with its ventral underside facing the camera, the "
    "frenulum visibly centered on the ventral midline directly beneath the "
    "glans, and the dorsal surface facing away from the camera."
)
PENIS_ANATOMY_PATTERN = re.compile(
    r"\b(?:penis|penile|cock|dick|phallus|frenulum|frenular|"
    r"penisb(?:a|ä)ndchen)\b",
    re.IGNORECASE,
)
PENIS_VENTRAL_VIEWPOINT_PATTERN = re.compile(
    r"\b(?:from\s+(?:directly\s+)?(?:below|beneath)|viewed\s+from\s+beneath|"
    r"underneath\s+view|worm['’]?s[- ]eye(?:\s+view)?|"
    r"(?:camera|viewpoint)\s+(?:is\s+)?(?:placed|positioned|located)?\s*"
    r"(?:directly\s+)?(?:below|beneath|under)\s+(?:the\s+)?"
    r"(?:penis|genitals?|pelvis|groin)|"
    r"(?:camera|viewpoint)[^.!?]{0,80}\blooking\s+upward\b|"
    r"(?:camera|viewpoint)[^.!?]{0,80}\bat\s+(?:the\s+)?(?:feet|foot\s+end)"
    r"[^.!?]{0,80}\b(?:toward|towards)\s+(?:the\s+)?torso\b)\b",
    re.IGNORECASE,
)
PENIS_ORIENTATION_ANALYSIS_PATTERN = re.compile(
    r"\b(?:low[- ]angle|high[- ]angle|eye[- ]level|overhead|top[- ]down|"
    r"front(?:al)?\s+view|rear\s+view|side(?:\s+view|[- ]profile)?|"
    r"three-quarter\s+view|camera|viewpoint|lens|from\s+(?:above|below|beneath)|"
    r"supine|prone|reclining|lying|standing|kneeling|seated|crouching|"
    r"rotated|rotation|pointing|angled|erect|flaccid)\b",
    re.IGNORECASE,
)
PENIS_VENTRAL_REVERSED_PATTERNS = (
    re.compile(
        r"\bfrenulum\b[^.!?]{0,45}\b(?:on|along|at)\b[^.!?]{0,20}"
        r"\b(?:dorsal|upper|top)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:dorsal|upper|top)\b[^.!?]{0,35}"
        r"\b(?:contains?|showing|with|bearing)\b[^.!?]{0,25}\bfrenulum\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdorsal(?:\s+(?:side|surface))?\b[^.!?]{0,35}"
        r"\b(?:faces?|facing|turned|oriented)\s+"
        r"(?:(?:directly|toward|towards|to)\s+)?(?:the\s+)?"
        r"(?:camera|viewer)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bventral(?:\s+(?:side|surface|underside))?\b[^.!?]{0,35}"
        r"\b(?:faces?|facing|turned|oriented)\s+away\b",
        re.IGNORECASE,
    ),
)

CROSS_SUBJECT_MANUAL_ACTION_PATTERN = re.compile(
    r"\b(?:grip(?:s|ped|ping)?|hold(?:s|ing)?|held|stroke(?:s|d|ing)?|"
    r"rub(?:s|bed|bing)?|manual(?:ly)?\s+stimulat(?:e|es|ed|ing|ion)|"
    r"hand[- ]?to[- ]?penis\s+contact|handjobs?)\b",
    re.IGNORECASE,
)
CROSS_SUBJECT_MALE_GENITAL_OWNER_PATTERN = re.compile(
    r"(?:\b(?:his|(?:the\s+)?(?:[a-z-]+\s+){0,4}man['’]s)\s+"
    r"(?:[a-z-]+\s+){0,3}(?:penis|shaft|cock|dick|phallus)\b|"
    r"\b(?:penis|shaft|cock|dick|phallus)\s+of\s+(?:the\s+)?"
    r"(?:[a-z-]+\s+){0,4}man\b)",
    re.IGNORECASE,
)
CROSS_SUBJECT_EXPLICIT_MALE_GENITAL_OWNER_PATTERN = re.compile(
    r"(?:\b(?:the\s+)?(?:[a-z-]+\s+){0,4}man['’]s\s+"
    r"(?:[a-z-]+\s+){0,3}(?:penis|shaft|cock|dick|phallus)\b|"
    r"\b(?:penis|shaft|cock|dick|phallus)\s+of\s+(?:the\s+)?"
    r"(?:[a-z-]+\s+){0,4}man\b)",
    re.IGNORECASE,
)


def requires_explicit_cross_subject_genital_binding(text: str) -> bool:
    """Detect a woman manually stimulating a distinctly male-owned penis."""

    cleaned = NEGATIVE_CLAUSE_PATTERN.sub("", str(text or ""))
    has_woman = bool(re.search(r"\b(?:woman|female|lady|wife|bride)\b", cleaned, re.I))
    has_man = bool(re.search(r"\b(?:man|male|husband|groom)\b", cleaned, re.I))
    female_actor = bool(
        re.search(
            r"\b(?:she|her\s+(?:own\s+)?hands?|(?:the\s+)?(?:adult\s+)?"
            r"(?:woman|female|lady|wife|bride)(?:['’]s\s+hands?)?)\b",
            cleaned,
            re.IGNORECASE,
        )
    )
    sexual_context = bool(
        SEXUAL_SIGNAL_PATTERN.search(cleaned)
        or re.search(r"\b(?:erect\s+shaft|precum|semen)\b", cleaned, re.IGNORECASE)
    )
    return bool(
        has_woman
        and has_man
        and female_actor
        and sexual_context
        and CROSS_SUBJECT_MANUAL_ACTION_PATTERN.search(cleaned)
        and CROSS_SUBJECT_MALE_GENITAL_OWNER_PATTERN.search(cleaned)
    )


def cross_subject_genital_binding_instruction(text: str) -> str:
    """Return private FLUX wording guidance for overlapping intimate contact."""

    if not requires_explicit_cross_subject_genital_binding(text):
        return ""
    return (
        "Private cross-subject body-binding contract: in one natural action "
        "lead at the beginning of the final prompt, state that exactly two distinct "
        "adults are visible, and name both adults instead of relying only on her/his "
        "pronouns. State that the adult woman performs the manual contact on the "
        "adult man's penis and that the penis is visibly attached to the man's "
        "pelvis with a continuous base-to-tip direction extending outward from his "
        "groin. Do not quote or label this private contract in the final prompt."
    )


def cross_subject_genital_binding_lead(text: str) -> str:
    """Build a concise positive FLUX lead from the supplied spatial relationship."""

    if not requires_explicit_cross_subject_genital_binding(text):
        return ""
    cleaned = NEGATIVE_CLAUSE_PATTERN.sub("", str(text or ""))
    seated_man = bool(
        re.search(
            r"\b(?:seated\s+(?:adult\s+)?man|(?:adult\s+)?man\s+(?:is\s+)?seated)\b",
            cleaned,
            re.IGNORECASE,
        )
    )
    woman_behind_man = bool(
        re.search(
            r"\b(?:woman|female|lady|wife|bride)\b[^.!?;]{0,100}\bbehind\b"
            r"[^.!?;]{0,100}\b(?:man|male|husband|groom)\b",
            cleaned,
            re.IGNORECASE,
        )
    )
    wide = bool(re.search(r"\bwide(?:\s+full[- ]scene)?\s+shot\b|\bshot\s+wide\b", cleaned, re.I))
    erect = bool(re.search(r"\berect\b", cleaned, re.IGNORECASE))
    uses_fingers = bool(
        re.search(
            r"\b(?:her|(?:the\s+)?(?:adult\s+)?woman['’]s)\s+"
            r"(?:[a-z-]+\s+){0,2}fingers?\b",
            cleaned,
            re.IGNORECASE,
        )
    )
    frame = " in one continuous wide shot" if wide else " in one continuous scene"
    man_role = "the seated adult man" if seated_man else "the adult man"
    effector = "fingers" if uses_fingers else "hands"
    if seated_man and woman_behind_man:
        staging = ""
        action = (
            "The adult woman is positioned directly behind the seated adult man "
            f"and reaches forward with her {effector} to grip and manually stimulate "
        )
    else:
        staging = "One is an adult woman and the other is an adult man. "
        action = f"The adult woman uses her {effector} to grip and manually stimulate "
    anatomy = f"{man_role}'s {'erect ' if erect else ''}penis"
    return (
        f"Exactly two distinct adults are fully visible{frame}. "
        + staging
        + action
        + anatomy
        + ", whose base is visibly attached to the man's pelvis as the penis "
        + "extends outward from his groin in one continuous base-to-tip direction."
    )


def cross_subject_genital_binding_issues(
    final_prompt: str,
    original_prompt: str,
) -> list[str]:
    """Require role nouns beside the actor and anatomy owner for FLUX."""

    if not requires_explicit_cross_subject_genital_binding(original_prompt):
        return []
    candidate = str(final_prompt or "")
    opening = " ".join(candidate.split()[:90])
    if not re.search(r"\bexactly\s+two\s+distinct\s+adults\b", opening, re.IGNORECASE):
        return ["front-load exactly two distinct adults before secondary scene detail"]
    for sentence in re.split(r"(?<=[.!?;])\s+", opening):
        woman_before_action = re.search(
            r"\b(?:the\s+)?(?:[a-z-]+\s+){0,4}woman(?:['’]s\s+hands?)?\b"
            r"[^.!?;]{0,140}\b(?:grip(?:s|ped|ping)?|hold(?:s|ing)?|held|"
            r"stroke(?:s|d|ing)?|rub(?:s|bed|bing)?|manual(?:ly)?\s+"
            r"stimulat(?:e|es|ed|ing|ion))\b",
            sentence,
            re.IGNORECASE,
        )
        passive_woman_actor = re.search(
            r"\b(?:grip(?:ped|ping)?|held|strok(?:ed|ing)|rubbed|stimulated)\b"
            r"[^.!?;]{0,100}\bby\s+(?:the\s+)?(?:[a-z-]+\s+){0,4}woman\b",
            sentence,
            re.IGNORECASE,
        )
        if (
            (woman_before_action or passive_woman_actor)
            and CROSS_SUBJECT_EXPLICIT_MALE_GENITAL_OWNER_PATTERN.search(sentence)
            and re.search(
                r"\b(?:attached|connected|continuous)\b[^.!?;]{0,80}\b"
                r"(?:man's|male|pelvis)\b|\bman's\s+pelvis\b",
                opening,
                re.IGNORECASE,
            )
            and re.search(
                r"\b(?:base-to-tip|root-to-tip|extends?\s+outward)\b",
                opening,
                re.IGNORECASE,
            )
        ):
            return []
    return [
        "front-load the adult woman as the manual actor, the adult man as the "
        "penis owner, and a continuous outward base-to-tip direction from the "
        "man's pelvis"
    ]


def enforce_cross_subject_genital_binding(
    candidate: str,
    original_prompt: str,
) -> str:
    """Front-load one positive visual contract when FLUX role binding is weak."""

    cleaned = " ".join(str(candidate or "").split()).strip()
    lead = cross_subject_genital_binding_lead(original_prompt)
    if not lead:
        return cleaned
    cleaned = re.sub(
        r"(?is)^\s*Exactly\s+two\s+distinct\s+adults\b.*?"
        r"(?:continuous\s+(?:base|root)-to-tip\s+direction|"
        r"two\s+distinct\s+adult\s+bodies\s+in\s+the\s+overlapping\s+pose)\s*\.\s*",
        "",
        cleaned,
        count=1,
    )
    obsolete = re.compile(
        r"(?i)\bThe adult woman's hands visibly grip and manually stimulate the "
        r"adult man's penis, with her arms and his pelvis belonging to two distinct "
        r"adult bodies in the overlapping pose\.\s*"
    )
    cleaned = obsolete.sub("", cleaned).strip()
    cleaned = re.sub(
        r"(?i)^\s*(?:2|two)\s+people\s*\.\s*",
        "",
        cleaned,
        count=1,
    )
    cleaned = re.sub(
        r"(?i)\bA\s+woman\s+stands\s+behind\s+a\s+seated\s+man\s*,\s*"
        r"her\s+hands\s+gripping\s+his\s+(?:erect\s+)?penis\s+with\s+"
        r"deliberate\s+force\s+as\s+precum\s+glistens\s+under\s+warm\s*,\s*"
        r"intimate\s+lighting\s*\.\s*",
        "The woman's grip remains deliberately forceful while precum glistens "
        "at the tip of the man's penis under warm, intimate lighting. ",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\bThe\s+scene\s+visibly\s+includes\s+manual\s+stimulation\s+"
        r"of\s+(?:the\s+)?penis\s*\.?\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(re.escape(lead), "", cleaned, flags=re.IGNORECASE).strip(" .")
    compacted_sentences: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        redundant_action = bool(
            re.search(r"\b(?:woman|female)\b", sentence, re.IGNORECASE)
            and re.search(r"\b(?:man|male)\b", sentence, re.IGNORECASE)
            and CROSS_SUBJECT_MANUAL_ACTION_PATTERN.search(sentence)
            and CROSS_SUBJECT_MALE_GENITAL_OWNER_PATTERN.search(sentence)
        )
        if not redundant_action:
            compacted_sentences.append(sentence)
            continue
        details: list[str] = []
        if re.search(r"\bphotoreal(?:istic)?\b", sentence, re.IGNORECASE):
            details.append("Photorealistic rendering.")
        lens = re.search(r"\b(\d+)\s*mm\s+lens\b", sentence, re.IGNORECASE)
        if lens:
            details.append(f"The wide composition uses a {lens.group(1)}mm lens.")
        lighting = re.search(
            r"\b((?:soft|warm|natural|intimate|dramatic)(?:\s+[a-z-]+){0,3}\s+"
            r"light(?:ing)?)\b",
            sentence,
            re.IGNORECASE,
        )
        if lighting:
            details.append(lighting.group(1).capitalize() + " illuminates the scene.")
        if re.search(r"\b(?:deliberate\s+force|forceful)\b", sentence, re.IGNORECASE):
            details.append("The woman's grip remains deliberately forceful.")
        if re.search(r"\bprecum\b", sentence, re.IGNORECASE):
            details.append("Precum glistens at the tip of the man's penis.")
        compacted_sentences.extend(details)
    cleaned = " ".join(value.strip() for value in compacted_sentences if value.strip())
    combined = lead + ((" " + cleaned) if cleaned else "")
    return " ".join(combined.split()).strip()


def requests_visible_penis_ventral_orientation(text: str) -> bool:
    """Return whether the request makes the penile underside orientation visible."""

    cleaned = NEGATIVE_CLAUSE_PATTERN.sub("", str(text or ""))
    if re.search(
        r"\b(?:frenulum|frenular|penisb(?:a|ä)ndchen)\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return True
    penis_term = r"(?:penis|penile|cock|dick|phallus)"
    orientation_cue = r"(?:ventral(?:ly)?|underside|under[- ]surface|unterseite)"
    explicit_pair = re.search(
        rf"\b(?:{penis_term}\b[^.!?]{{0,35}}\b{orientation_cue}|"
        rf"{orientation_cue}\b[^.!?]{{0,35}}\b{penis_term})\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if explicit_pair:
        return True
    explicit_dorsal_pair = re.search(
        rf"\b(?:{penis_term}\b[^.!?]{{0,35}}\bdorsal|"
        rf"dorsal\b[^.!?]{{0,35}}\b{penis_term})\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if explicit_dorsal_pair:
        return False
    return bool(
        PENIS_ANATOMY_PATTERN.search(cleaned)
        and PENIS_VENTRAL_VIEWPOINT_PATTERN.search(cleaned)
    )


def needs_penis_orientation_analysis(text: str) -> bool:
    """Return whether pose or camera cues require conditional surface reasoning."""

    cleaned = NEGATIVE_CLAUSE_PATTERN.sub("", str(text or ""))
    return bool(
        PENIS_ANATOMY_PATTERN.search(cleaned)
        and PENIS_ORIENTATION_ANALYSIS_PATTERN.search(cleaned)
    )


def penis_ventral_orientation_instruction(text: str) -> str:
    """Return private dorsal/ventral guidance only for a requested visible underside."""

    underside_required = requests_visible_penis_ventral_orientation(text)
    if not underside_required and not needs_penis_orientation_analysis(text):
        return ""
    shared = (
        "Private viewpoint-aware penile anatomy contract: treat this as a "
        "front-to-back dorsal/ventral relationship, never as image-left versus "
        "image-right. Infer the visible surface only from the combined body pose, "
        "penile rotation, and camera viewpoint. A side view, ordinary front view, "
        "close-up, or generic low angle alone is not enough. "
    )
    if underside_required:
        return shared + (
            "The requested view exposes the underside, so the ventral surface faces "
            "the camera, the frenulum lies on the ventral midline directly beneath "
            "the glans, and the dorsal surface faces away. State that relationship "
            "concretely in the final visual prompt without clinical explanation."
        )
    return shared + (
        "Determine whether the supplied pose and viewpoint expose the ventral underside "
        "or the dorsal surface. If the underside is exposed, place the frenulum on the "
        "ventral midline directly beneath the glans. If the dorsal or top surface is "
        "exposed, do not move or invent the frenulum there. Add only the orientation "
        "that is visually supported."
    )


def penis_ventral_orientation_issues(
    final_prompt: str,
    original_prompt: str,
) -> list[str]:
    """Validate a specifically requested view of the penile ventral underside."""

    if not requests_visible_penis_ventral_orientation(original_prompt):
        return []
    candidate = str(final_prompt or "")
    issues: list[str] = []
    if any(pattern.search(candidate) for pattern in PENIS_VENTRAL_REVERSED_PATTERNS):
        issues.append("frenulum or ventral underside is assigned to the dorsal or top surface")
    required_patterns = (
        (r"\bfrenulum\b", "frenulum is not explicitly visible"),
        (
            r"\b(?:ventral(?:\s+(?:surface|side|underside|midline))?|underside)\b",
            "ventral underside is not explicit",
        ),
        (
            r"\b(?:directly\s+)?(?:beneath|under|below)\s+(?:the\s+)?glans\b",
            "frenulum is not bound beneath the glans",
        ),
        (
            r"\b(?:ventral(?:\s+(?:surface|side|underside))?|underside)\b"
            r"[^.!?]{0,90}\b(?:faces?|facing|toward|towards)\b"
            r"[^.!?]{0,35}\b(?:camera|viewer)\b",
            "ventral underside is not oriented toward the camera",
        ),
        (
            r"\bdorsal(?:\s+(?:surface|side))?\b[^.!?]{0,90}"
            r"\b(?:faces?|facing|turned)\b[^.!?]{0,35}\baway\b",
            "dorsal surface is not oriented away from the camera",
        ),
    )
    for pattern, label in required_patterns:
        if not re.search(pattern, candidate, flags=re.IGNORECASE):
            issues.append(label)
    return issues


def enforce_penis_ventral_orientation_contract(
    candidate: str,
    original_prompt: str,
) -> str:
    """Add one concise, correct orientation sentence when the request requires it."""

    cleaned = str(candidate or "").strip()
    if not requests_visible_penis_ventral_orientation(original_prompt):
        return cleaned
    cleaned = re.sub(
        r"(?i)\b(?:the\s+)?frenulum(?:\s+is)?(?:\s+clearly)?"
        r"(?:\s+visible|\s+located)?\s+(?:on|along|at)\s+(?:the\s+)?"
        r"(?:dorsal|upper|top)\s+(?:side|surface|midline)\b",
        "the frenulum is visible on the ventral midline directly beneath the glans",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\b(?:the\s+)?dorsal\s+(?:side|surface)\s+"
        r"(?:faces?|facing|turned\s+toward|oriented\s+toward)\s+"
        r"(?:the\s+)?(?:camera|viewer)\b",
        "the ventral underside facing the camera",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\b(?:the\s+)?ventral\s+(?:side|surface|underside)\s+"
        r"(?:faces?|facing|turned)\s+away\s+from\s+(?:the\s+)?"
        r"(?:camera|viewer)\b",
        "the dorsal surface facing away from the camera",
        cleaned,
    )
    if not penis_ventral_orientation_issues(cleaned, original_prompt):
        return cleaned
    if cleaned:
        return cleaned.rstrip(" .") + ". " + PENIS_VENTRAL_ORIENTATION_PROMPT
    return PENIS_VENTRAL_ORIENTATION_PROMPT

ACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("toy use", TOY_USE_PATTERN),
    (
        "masturbation",
        re.compile(
            r"\b(?:masturbat\w*|solo\s+pleasure|self[- ](?:pleasur\w*|stimulat\w*|"
            r"penetrat\w*)|fuck\w*\s+(?:herself|himself|themself|themselves))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "oral sex",
        re.compile(
            r"\b(?:oral\s+(?:sex|pleasure|intimacy|stimulation)|"
            r"mouth-to-genital\s+contact|blowjobs?|cunnilingus)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "anal sex",
        re.compile(
            r"\b(?:anal\s+(?:sex|intercourse|penetration|intimacy)|"
            r"penetration\s+at\s+(?:the\s+)?(?:anus|anal\s+opening)|"
            r"rear-entry\s+anal)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "vaginal intercourse",
        re.compile(
            r"\b(?:vaginal\s+(?:sex|intercourse|penetration)|"
            r"penetration\s+at\s+(?:the\s+)?vaginal\s+opening|"
            r"(?:missionary|face-to-face|seated|standing|side-by-side|rear-entry)\s+intercourse)\b",
            re.IGNORECASE,
        ),
    ),
    ("intercourse", re.compile(r"\b(?:intercourse|making\s+love|has?\s+sex)\b", re.IGNORECASE)),
    (
        "manual stimulation",
        re.compile(
            r"\b(?:manual(?:ly)?\s+stimulat(?:e|es|ed|ing|ion)|"
            r"handjobs?|fingering|"
            r"(?:grip(?:s|ped|ping)?|hold(?:s|ing)?|held|"
            r"strok(?:e|es|ed|ing)|rub(?:s|bed|bing)?)\b"
            r"[^.!?;]{0,70}\b(?:penis|cock|(?:his|her|their)"
            r"(?:\s+erect)?\s+shaft|erect\s+shaft))\b",
            re.IGNORECASE,
        ),
    ),
    ("kissing", re.compile(r"\b(?:kiss(?:es|ed|ing)?|making\s+out)\b", re.IGNORECASE)),
    ("undressing", re.compile(r"\b(?:undress\w*|remov\w+\s+(?:clothing|lingerie))\b", re.IGNORECASE)),
    ("bondage", re.compile(r"\b(?:bondage|restraints?|restrain\w*|blindfold|cuffs?|rope)\b", re.IGNORECASE)),
)
BODY_TARGET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "vaginal",
        re.compile(
            r"\b(?:vagina|vaginal|vaginally|vulva|vulval|labia|pussy)\b",
            re.IGNORECASE,
        ),
    ),
    ("anal", re.compile(r"\b(?:anus|anal|anally|rectum|rectal)\b", re.IGNORECASE)),
    ("oral", re.compile(r"\b(?:mouth|oral|lips|tongue)\b", re.IGNORECASE)),
    ("chest", re.compile(r"\b(?:chest|breasts?|nipples?)\b", re.IGNORECASE)),
    ("genital", re.compile(r"\b(?:genitals?|penis|vagina|vulva|testicles?|scrotum)\b", re.IGNORECASE)),
)
OBJECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dildo", re.compile(r"\bdildos?\b", re.IGNORECASE)),
    ("vibrator", re.compile(r"\bvibrators?\b", re.IGNORECASE)),
    ("strap-on", re.compile(r"\bstrap[- ]ons?\b", re.IGNORECASE)),
    ("anal toy", re.compile(r"\b(?:anal\s+toy|butt\s+plug|anal\s+beads?)\b", re.IGNORECASE)),
    ("adult toy", re.compile(r"\b(?:adult|sex)\s+toys?\b", re.IGNORECASE)),
)
PHASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aftercare", re.compile(r"\b(?:aftercare|afterglow|cleaning\s+up|morning\s+after|post-climax)\b", re.IGNORECASE)),
    (
        "climax",
        re.compile(
            r"\b(?:climax\w*|orgasm\w*|ejaculat\w*|peak|release)\b",
            re.IGNORECASE,
        ),
    ),
    ("active", re.compile(r"\b(?:penetrat\w*|intercourse|masturbat\w*|thrust\w*|oral\s+(?:sex|pleasure)|manual\s+stimulation)\b", re.IGNORECASE)),
    ("foreplay", re.compile(r"\b(?:foreplay|undress\w*|caress\w*|intimate\s+touch|deep\s+kiss)\b", re.IGNORECASE)),
    ("anticipation", re.compile(r"\b(?:anticipat\w*|seduc\w*|teas\w*|almost-touching|inviting\s+closer)\b", re.IGNORECASE)),
)
FLUID_OUTCOME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("precum", re.compile(r"\b(?:precum|pre[- ]ejaculate)\b", re.IGNORECASE)),
    ("semen", re.compile(r"\b(?:semen|cum)\b", re.IGNORECASE)),
    ("ejaculation", re.compile(r"\bejaculat\w*\b", re.IGNORECASE)),
)
ADDITIONAL_PARTICIPANT_PATTERN = re.compile(
    rf"\b(?:another|additional|second|third|fourth)\s+{ROLE_PATTERN_TEXT}\b",
    re.IGNORECASE,
)
REACTION_PATTERN = re.compile(
    r"\b(?:gasp\w*|moan\w*|trembl\w*|shudder\w*|flush\w*|"
    r"breathless|quickened\s+breath|parted\s+lips|closed\s+eyes|"
    r"pleasure|ecstasy|arousal|climax\w*|orgasm\w*)\b",
    re.IGNORECASE,
)
ACTION_CAUSE_PATTERN = re.compile(
    r"\b(?:as|while|when|in\s+response\s+to|because\s+of|from|during|"
    r"touch\w*|kiss\w*|penetrat\w*|thrust\w*|stimulat\w*|uses?|"
    r"press\w*|grip\w*|movement|rhythm|contact|pressure)\b",
    re.IGNORECASE,
)
SEQUENCE_PATTERN = re.compile(
    r"\b(?:then|next|afterward|afterwards|before\s+continuing|"
    r"moves?\s+from.+?\s+to|shifts?\s+from.+?\s+into|"
    r"begins?.+?\s+then|progress(?:es|ing)?\s+from)\b",
    re.IGNORECASE,
)
COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split()).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def _positive_contract_text(text: str) -> str:
    """Remove clauses that describe absent or prohibited scene content."""

    return " ".join(
        NEGATIVE_CLAUSE_PATTERN.sub(" ", str(text or "")).split()
    )


def _dildo_use_context(text: str) -> str:
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        if DILDO_USE_PATTERN.search(sentence):
            return sentence
    return ""


def _literal_dildo_contact(text: str) -> str:
    """Return only a source-supplied compact dildo/body relation."""

    match = re.search(
        r"\bdildos?\b\s+(?:in|inside)\s+"
        r"(?:(?:her|his|their|the|a)\s+)?"
        r"(?:vagina|vaginal\s+opening|vulva|pussy|anus|anal\s+opening|rectum)\b",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(match.group(0).split()) if match else ""


def dildo_direction_instruction(contract: dict[str, object]) -> str:
    """Return one concrete direction only for ordinary active dildo use."""

    target = str(contract.get("dildo_use_target", "")).strip()
    if not target:
        return ""
    return (
        "Keep the dildo visibly separate from the body: its rounded insertion tip "
        f"points toward the {target}, while its base or handle stays outside on the "
        "operator side and points away."
    )


def _participant_count(text: str) -> int | None:
    lowered = text.lower()
    if re.search(
        r"\b(?:solo|single)\s+"
        r"(?:(?:clearly\s+adult|mature\s+adult|middle-aged|older|adult|mature)\s+)?"
        r"(?:woman|man|person|subject)\b",
        lowered,
    ):
        return 1
    matches = re.findall(
        rf"\b(one|two|three|four|five|six|\d+)\s+{ROLE_PATTERN_TEXT}\b",
        lowered,
    )
    if matches:
        value = matches[0]
        return COUNT_WORDS.get(value, int(value) if value.isdigit() else 1)
    if re.search(r"\b(?:couple|two\s+partners|both\s+adults)\b", lowered):
        return 2
    if re.search(r"\b(?:three\s+partners|multiple\s+adults|group\s+sex)\b", lowered):
        return 3
    distinct_roles = {
        _canonical_role(match.group(0))
        for match in ROLE_PATTERN.finditer(lowered)
        if _canonical_role(match.group(0))
        in {"woman", "man", "female", "male", "nonbinary person", "non-binary person"}
    }
    if len(distinct_roles) >= 2:
        return len(distinct_roles)
    return None


RELATION_ACTIONS: tuple[tuple[str, str], ...] = (
    ("kissing", r"kiss(?:es|ed|ing)?"),
    ("penetration", r"penetrat(?:e|es|ed|ing)"),
    ("manual stimulation", r"(?:touch(?:es|ed|ing)?|stimulat(?:e|es|ed|ing))"),
    ("oral sex", r"(?:gives?|performs?)\s+oral\s+(?:sex|pleasure)\s+(?:to|on)"),
)


def _canonical_role(value: str) -> str:
    role = re.sub(r"^adult\s+", "", value.strip().lower())
    role = re.sub(r"\s+partners?$", "", role)
    role = re.sub(r"['’]s$", "", role)
    singular = {
        "womans": "woman",
        "women": "woman",
        "mans": "man",
        "men": "man",
        "female": "woman",
        "females": "woman",
        "male": "man",
        "males": "man",
        "partners": "partner",
        "lovers": "lover",
        "subjects": "subject",
        "performers": "performer",
        "dominants": "dominant",
        "submissives": "submissive",
        "nonbinary people": "nonbinary person",
        "non-binary people": "non-binary person",
    }
    return singular.get(role, role)


def _relations(text: str) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
    for action, action_pattern in RELATION_ACTIONS:
        verb_pattern = re.compile(rf"\b(?:{action_pattern})\b", re.IGNORECASE)
        for sentence in re.split(r"(?<=[.!?;])\s+", text):
            for verb_match in verb_pattern.finditer(sentence):
                preceding_roles = [
                    match
                    for match in ROLE_PATTERN.finditer(sentence[: verb_match.start()])
                    if verb_match.start() - match.end() <= 90
                ]
                following_roles = [
                    match
                    for match in ROLE_PATTERN.finditer(sentence[verb_match.end():])
                    if match.start() <= 70
                ]
                if not preceding_roles or not following_roles:
                    continue
                actor = preceding_roles[-1].group(0)
                receiver = following_roles[0].group(0)
                relations.append(
                    {
                        "actor": _canonical_role(actor),
                        "action": action,
                        "receiver": _canonical_role(receiver),
                    }
                )
    return relations


def extract_nsfw_scene_contract(
    text: str,
    *,
    content_format: str = "Single Image",
) -> dict[str, object]:
    """Extract participant, act, contact, object, phase, and reaction facts."""

    normalized = _positive_contract_text(text)
    roles = _unique(
        _canonical_role(match.group(0))
        for match in ROLE_PATTERN.finditer(normalized)
    )
    participant_count = _participant_count(normalized)
    acts = [label for label, pattern in ACT_PATTERNS if pattern.search(normalized)]
    # A solo adult actively using an intimate toy is still masturbating even
    # when a rewrite replaces that label with concrete visual mechanics such
    # as "thrusts a dildo into her vagina." Keep the semantic act stable so the
    # hard fidelity gate does not demand one exact word from the local model.
    if participant_count == 1 and "toy use" in acts and "masturbation" not in acts:
        acts.append("masturbation")
    if "intercourse" in acts and any(
        act in acts for act in ("anal sex", "vaginal intercourse")
    ):
        acts.remove("intercourse")
    targets = (
        [label for label, pattern in BODY_TARGET_PATTERNS if pattern.search(normalized)]
        if acts
        else []
    )
    if "genital" in targets and any(
        value in targets for value in ("vaginal", "anal")
    ):
        targets.remove("genital")
    objects = [label for label, pattern in OBJECT_PATTERNS if pattern.search(normalized)]
    if "adult toy" in objects and any(
        value in objects for value in ("dildo", "vibrator", "strap-on", "anal toy")
    ):
        objects.remove("adult toy")
    phases = [label for label, pattern in PHASE_PATTERNS if pattern.search(normalized)]
    chosen_phase = next(
        (
            phase
            for phase in ("climax", "active", "foreplay", "anticipation", "aftercare")
            if phase in phases
        ),
        "active" if acts else "",
    )
    reaction_terms = _unique(match.group(0).lower() for match in REACTION_PATTERN.finditer(normalized))
    fluid_outcomes = [
        label for label, pattern in FLUID_OUTCOME_PATTERNS if pattern.search(normalized)
    ]
    dildo_context = _dildo_use_context(normalized)
    dildo_use_target = ""
    if dildo_context and not re.search(
        r"\b(?:double[- ]ended|product|boxed|unopened|packaged|display)\b",
        dildo_context,
        flags=re.IGNORECASE,
    ):
        dildo_use_target = (
            "vaginal opening"
            if BODY_TARGET_PATTERNS[0][1].search(dildo_context)
            else "anal opening"
            if BODY_TARGET_PATTERNS[1][1].search(dildo_context)
            else "intended body-contact point"
        )
    return {
        "sexual": bool(SEXUAL_SIGNAL_PATTERN.search(normalized)),
        "participant_count": participant_count,
        "participant_roles": roles,
        "acts": acts,
        "body_targets": targets,
        "objects": objects,
        "dildo_use_target": dildo_use_target,
        "literal_core": _literal_dildo_contact(normalized),
        "relations": _relations(normalized),
        "phases": phases,
        "visible_phase": chosen_phase,
        "reactions": reaction_terms,
        "fluid_outcomes": fluid_outcomes,
        "single_phase_required": content_format == "Single Image",
    }


def format_nsfw_scene_contract(
    contract: dict[str, object],
    *,
    risk_level: str = "Balanced improvement",
) -> str:
    """Render a small literal-core contract without teaching scene mechanics."""

    if not contract.get("sexual"):
        return ""
    facts: list[str] = []
    participant_count = contract.get("participant_count")
    if participant_count is not None:
        facts.append(f"adult count={participant_count}")
    acts = ", ".join(contract.get("acts", []))
    if acts:
        facts.append(f"action={acts}")
    targets = ", ".join(contract.get("body_targets", []))
    if targets:
        facts.append(f"contact={targets}")
    objects = ", ".join(contract.get("objects", []))
    if objects:
        facts.append(f"object={objects}")
    literal_core = str(contract.get("literal_core", "")).strip()
    if literal_core:
        facts.append(f"literal wording={literal_core}")
    lines = ["Private literal adult-scene core:"]
    if facts:
        lines.append("- Keep these source facts unchanged: " + "; ".join(facts) + ".")
    relations = contract.get("relations", [])
    if isinstance(relations, list):
        for relation in relations:
            if isinstance(relation, dict):
                lines.append(
                    "- Keep this role direction: "
                    f"{relation.get('actor', '')} -> {relation.get('action', '')} -> "
                    f"{relation.get('receiver', '')}."
                )
    lines.extend(
        (
            "- State the core action once in short, ordinary image-generator wording.",
            "- Do not explain anatomy, toy geometry, insertion mechanics, or chronological phases unless the user requested them.",
        )
    )
    if risk_level == "Strict cleanup":
        lines.append(
            "- Strict cleanup: invent no scene facts; a source containing only the literal core stays concise."
        )
    elif risk_level == "Creative enhancement":
        lines.append(
            "- Creative enhancement: keep the literal core first, then build one coherent visual direction through compatible setting, staging, camera, lighting, style, material, and visible reaction details."
        )
        lines.append(
            "- Creative additions must not introduce another participant, sexual act, fetish, body target, object, identity, power dynamic, fluid, or outcome."
        )
    else:
        lines.append(
            "- Balanced improvement: keep the literal core first, then add one compact compatible visual cluster such as setting and staging, camera and composition, or lighting and style."
        )
        lines.append(
            "- Do not invent nudity, fluids, exclusions, another participant, sexual act, fetish, body target, object, identity, power dynamic, or outcome."
        )
    lines.append("- Do not quote or label this private literal core in the final prompt.")
    return "\n".join(lines)


def reaction_binding_issues(text: str, *, participant_count: int | None) -> list[str]:
    """Find unowned reactions in scenes with more than one adult participant."""

    if not participant_count or participant_count < 2:
        return []
    issues: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        reactions = _unique(match.group(0).lower() for match in REACTION_PATTERN.finditer(sentence))
        if not reactions:
            continue
        if not ROLE_PATTERN.search(sentence):
            issues.append("reaction is not assigned to a named adult role: " + ", ".join(reactions[:3]))
        if not ACTION_CAUSE_PATTERN.search(sentence):
            issues.append("reaction is not tied to its causing action or contact: " + ", ".join(reactions[:3]))
    return _unique(issues)


def enforce_reaction_binding(text: str, original_prompt: str) -> str:
    """Bind clear gender pronouns and their reactions to the source action."""

    source = extract_nsfw_scene_contract(original_prompt)
    participant_count = source.get("participant_count")
    if not isinstance(participant_count, int) or participant_count < 2:
        return str(text or "").strip()
    cross_subject_manual = requires_explicit_cross_subject_genital_binding(
        original_prompt
    )
    sentences: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", str(text or "").strip()):
        if not REACTION_PATTERN.search(sentence):
            sentences.append(sentence)
            continue
        bound = re.sub(
            r"^\s*Her\b",
            "The adult woman's",
            sentence,
            count=1,
            flags=re.IGNORECASE,
        )
        bound = re.sub(
            r"^\s*She\b",
            "The adult woman",
            bound,
            count=1,
            flags=re.IGNORECASE,
        )
        bound = re.sub(
            r"^\s*His\b",
            "The adult man's",
            bound,
            count=1,
            flags=re.IGNORECASE,
        )
        bound = re.sub(
            r"^\s*He\b",
            "The adult man",
            bound,
            count=1,
            flags=re.IGNORECASE,
        )
        cause_searchable = re.sub(
            r"(?i)\bnot\s+from\s+[^,.!?;]{1,60}?\s+but\s+[^,.!?;]+",
            "",
            bound,
        )
        if (
            cross_subject_manual
            and ROLE_PATTERN.search(bound)
            and not ACTION_CAUSE_PATTERN.search(cause_searchable)
        ):
            punctuation = bound[-1] if bound[-1:] in ".!?" else "."
            bound = bound.rstrip(".!?") + (
                " during her manual stimulation of the adult "
                "man's penis"
            ) + punctuation
        sentences.append(bound)
    return " ".join(sentence for sentence in sentences if sentence).strip()


def single_phase_issues(text: str, *, content_format: str) -> list[str]:
    """Reject visible multi-step progression in a normal still-image prompt."""

    if content_format != "Single Image" or not SEQUENCE_PATTERN.search(text):
        return []
    phases = [label for label, pattern in PHASE_PATTERNS if pattern.search(text)]
    if len(phases) >= 2:
        return [
            "single-image prompt contains a visible multi-phase progression: "
            + ", ".join(phases)
        ]
    return []


def nsfw_scene_contract_issues(
    final_prompt: str,
    original_prompt: str,
    *,
    content_format: str = "Single Image",
) -> list[str]:
    """Compare the final adult scene with explicit source acts and relationships."""

    source = extract_nsfw_scene_contract(original_prompt, content_format=content_format)
    if not source.get("sexual"):
        return []
    candidate = extract_nsfw_scene_contract(final_prompt, content_format=content_format)
    issues: list[str] = []
    identity_roles = {
        "woman",
        "man",
        "nonbinary person",
        "non-binary person",
    }
    source_roles = set(source.get("participant_roles", [])) & identity_roles
    candidate_roles = set(candidate.get("participant_roles", [])) & identity_roles
    for role in sorted(source_roles - candidate_roles):
        issues.append(f"missing requested adult participant role: {role}")
    source_count = source.get("participant_count")
    candidate_count = candidate.get("participant_count")
    if (
        isinstance(source_count, int)
        and source_count > 1
        and (
            not isinstance(candidate_count, int)
            or candidate_count < source_count
        )
    ):
        issues.append(
            f"missing requested adult participant count: expected {source_count}"
        )
    if (
        isinstance(source_count, int)
        and isinstance(candidate_count, int)
        and candidate_count > source_count
    ):
        issues.append(
            "unrequested additional adult participant count: "
            f"expected {source_count}, found {candidate_count}"
        )
    if (
        ADDITIONAL_PARTICIPANT_PATTERN.search(final_prompt)
        and not ADDITIONAL_PARTICIPANT_PATTERN.search(original_prompt)
    ):
        issues.append("unrequested additional adult participant")
    candidate_acts = set(candidate.get("acts", []))
    source_acts = set(source.get("acts", []))
    for act in source_acts:
        if act == "intercourse" and candidate_acts.intersection(
            {"intercourse", "anal sex", "vaginal intercourse"}
        ):
            continue
        if act not in candidate_acts:
            issues.append(f"missing requested sexual act family: {act}")
    authorized_acts = set(source_acts)
    if "intercourse" in source_acts:
        authorized_acts.update(("anal sex", "vaginal intercourse"))
    for act in sorted(candidate_acts - authorized_acts):
        issues.append(f"unrequested sexual act family added: {act}")
    source_targets = set(source.get("body_targets", []))
    candidate_targets = set(candidate.get("body_targets", []))
    for target in source_targets - candidate_targets:
        issues.append(f"missing requested body/contact target: {target}")
    if "vaginal" in source_targets and "anal" in candidate_targets - source_targets:
        issues.append("unrequested anal contact added to a vaginal source")
    if "anal" in source_targets and "vaginal" in candidate_targets - source_targets:
        issues.append("unrequested vaginal contact added to an anal source")
    authorized_targets = set(source_targets)
    if "oral sex" in source_acts:
        authorized_targets.add("oral")
    if source_acts.intersection({"manual stimulation", "masturbation"}):
        authorized_targets.add("genital")
    for target in sorted(candidate_targets - authorized_targets):
        if target == "anal" and "vaginal" in source_targets:
            continue
        if target == "vaginal" and "anal" in source_targets:
            continue
        issues.append(f"unrequested body/contact target added: {target}")
    source_objects = set(source.get("objects", []))
    candidate_objects = set(candidate.get("objects", []))
    for requested_object in source.get("objects", []):
        generic_toy_satisfied = (
            requested_object == "adult toy"
            and bool(candidate_objects.intersection(
                {"adult toy", "dildo", "vibrator", "strap-on", "anal toy"}
            ))
        )
        if requested_object not in candidate_objects and not generic_toy_satisfied:
            issues.append(f"missing requested adult object: {requested_object}")
    for added_object in sorted(candidate_objects - source_objects):
        if "adult toy" in source_objects:
            continue
        issues.append(f"unrequested adult object added: {added_object}")

    source_phases = set(source.get("phases", []))
    if source_acts:
        source_phases.add("active")
    source_visible_phase = str(source.get("visible_phase", ""))
    if source_visible_phase:
        source_phases.add(source_visible_phase)
    candidate_phases = set(candidate.get("phases", []))
    for phase in sorted(candidate_phases - source_phases):
        issues.append(f"unrequested visible sexual phase or outcome added: {phase}")
    source_fluids = set(source.get("fluid_outcomes", []))
    candidate_fluids = set(candidate.get("fluid_outcomes", []))
    for fluid in sorted(candidate_fluids - source_fluids):
        issues.append(f"unrequested sexual fluid or outcome added: {fluid}")

    candidate_relations = {
        (
            str(relation.get("actor", "")),
            str(relation.get("action", "")),
            str(relation.get("receiver", "")),
        )
        for relation in candidate.get("relations", [])
        if isinstance(relation, dict)
    }
    for relation in source.get("relations", []):
        if not isinstance(relation, dict):
            continue
        key = (
            str(relation.get("actor", "")),
            str(relation.get("action", "")),
            str(relation.get("receiver", "")),
        )
        if key not in candidate_relations:
            issues.append(
                "missing or reversed sexual role binding: "
                f"{key[0]} -> {key[1]} -> {key[2]}"
            )
    issues.extend(
        reaction_binding_issues(
            final_prompt,
            participant_count=candidate_count,
        )
    )
    issues.extend(single_phase_issues(final_prompt, content_format=content_format))
    return _unique(issues)


def infer_nsfw_preset_metadata(
    kind: str,
    category: str,
    value: str,
) -> dict[str, object]:
    """Infer complete compatibility metadata for one adult catalog entry."""

    text = f"{category} {value}".lower()
    contract = extract_nsfw_scene_contract(value)
    participant_modes: list[str] = []
    if re.search(r"\b(?:solo|one adult|single adult|own expression)\b", text):
        participant_modes.append("solo")
    if re.search(r"\b(?:three|multiple|group|polyamorous)\b", text):
        participant_modes.append("group")
    if re.search(r"\b(?:partners?|lovers?|couple|two-person|both adults)\b", text):
        participant_modes.append("couple")
    if not participant_modes:
        participant_modes.append("any")
    return {
        "kind": kind,
        "category": category,
        "value": value,
        "participant_modes": _unique(participant_modes),
        "act_families": list(contract.get("acts", [])),
        "body_targets": list(contract.get("body_targets", [])),
        "objects": list(contract.get("objects", [])),
        "phase": str(contract.get("visible_phase", "")),
        "requires_separate_object": bool(contract.get("objects")),
        "reaction_cues": _unique(
            match.group(0).lower() for match in REACTION_PATTERN.finditer(value)
        ),
    }


def nsfw_preset_compatibility_issues(
    metadata: Iterable[dict[str, object]],
    *,
    content_format: str = "Single Image",
) -> list[str]:
    """Return conflicts across selected adult presets."""

    entries = list(metadata)
    modes = {
        str(mode)
        for entry in entries
        for mode in entry.get("participant_modes", [])
        if str(mode) != "any"
    }
    phases = {
        str(entry.get("phase", ""))
        for entry in entries
        if str(entry.get("phase", ""))
    }
    acts = {
        str(act)
        for entry in entries
        for act in entry.get("act_families", [])
    }
    issues: list[str] = []
    if "solo" in modes and modes.intersection({"couple", "group"}):
        issues.append("solo and multi-participant adult presets are selected together")
    if content_format == "Single Image" and len(phases) > 1:
        issues.append(
            "adult presets span multiple visible phases for one still: "
            + ", ".join(sorted(phases))
        )
    if "anal sex" in acts and "vaginal intercourse" in acts and content_format == "Single Image":
        issues.append("anal and vaginal intercourse presets compete for the same still-image beat")
    return issues


def format_nsfw_preset_contract(
    metadata: Iterable[dict[str, object]],
    *,
    content_format: str = "Single Image",
) -> str:
    """Summarize selected adult preset metadata as private compatibility guidance."""

    entries = list(metadata)
    if not entries:
        return ""
    acts = _unique(
        str(act)
        for entry in entries
        for act in entry.get("act_families", [])
    )
    phases = _unique(str(entry.get("phase", "")) for entry in entries)
    objects = _unique(
        str(obj)
        for entry in entries
        for obj in entry.get("objects", [])
    )
    issues = nsfw_preset_compatibility_issues(entries, content_format=content_format)
    lines = ["Private NSFW preset compatibility:"]
    if acts:
        lines.append("- Selected act families: " + ", ".join(acts))
    if phases:
        lines.append("- Selected phases: " + ", ".join(phases))
    if objects:
        lines.append("- Selected separate objects: " + ", ".join(objects))
    if issues:
        lines.append("- Resolve these conflicts in favor of the user's draft: " + "; ".join(issues))
    lines.append("- Do not quote or label this private preset metadata in the final prompt.")
    return "\n".join(lines)


def strip_nsfw_catalog_labels(text: str) -> str:
    """Remove internal adult-library category names while preserving chosen values."""

    cleaned = re.sub(
        r"(?i)\bNSFW\s*(?:[—–-]|,)\s*[^:.;\n]{1,100}\s*:\s*",
        "",
        str(text or ""),
    )
    cleaned = re.sub(
        r"(?i)\bNSFW\s*[—–-]\s*(?:adult\s+)?(?:erotic|sexual)\s+",
        "",
        cleaned,
    )
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def nsfw_image_audit_contract(
    original_prompt: str,
    corrected_prompt: str,
    *,
    content_format: str = "Single Image",
) -> str:
    """Return exact adult-scene checks for generated-image inspection."""

    source = extract_nsfw_scene_contract(original_prompt, content_format=content_format)
    corrected = extract_nsfw_scene_contract(corrected_prompt, content_format=content_format)
    acts = _unique([*source.get("acts", []), *corrected.get("acts", [])])
    targets = _unique([*source.get("body_targets", []), *corrected.get("body_targets", [])])
    objects = _unique([*source.get("objects", []), *corrected.get("objects", [])])
    direction = dildo_direction_instruction(source) or dildo_direction_instruction(corrected)
    penile_orientation = requests_visible_penis_ventral_orientation(
        f"{original_prompt}\n{corrected_prompt}"
    )
    cross_subject_binding = requires_explicit_cross_subject_genital_binding(
        f"{original_prompt}\n{corrected_prompt}"
    )
    lines = [
        "NSFW visual fidelity audit:",
        f"- Adult participant count and distinct roles: {source.get('participant_count') or 'as requested'}; "
        + (", ".join(source.get("participant_roles", [])) or "preserve the supplied adult roles"),
        "- Required sexual act families: " + (", ".join(acts) or "preserve the requested intimate action"),
        "- Required body/contact targets: " + (", ".join(targets) or "preserve the supplied contact"),
        "- Required separate objects: " + (", ".join(objects) or "none explicitly extracted"),
        *([f"- Required normal-use object direction: {direction}"] if direction else []),
        *(
            [
                "- Required cross-subject anatomy binding: exactly two distinct "
                "adults; the woman's hands perform the manual contact and the penis "
                "has its base anatomically attached to the man's pelvis and extends "
                "outward from his groin in a continuous base-to-tip direction. Mark "
                "participant_count, action_roles, body_ownership, anatomical_attachment, "
                "and anatomical_orientation as fail if "
                "the image shows one fused subject, assigns the penis to the woman, "
                "shows detached or floating anatomy, or does not show a continuous "
                "outward attachment and orientation from the man's pelvis."
            ]
            if cross_subject_binding
            else []
        ),
        *(
            [
                "- Required penile orientation: the visible frenulum is on the ventral "
                "midline directly beneath the glans, the ventral underside faces the "
                "camera, and the dorsal surface faces away."
            ]
            if penile_orientation
            else []
        ),
        f"- Required visible phase: {source.get('visible_phase') or corrected.get('visible_phase') or 'one decisive phase'}",
        "- Verify actor and receiver roles, limb ownership, object/body separation, contact direction, "
        "and participant-specific reactions. Report every mismatch in nsfw_fidelity.",
    ]
    return "\n".join(lines)
