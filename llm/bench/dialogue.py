"""Synthetic multi-session memory benchmark (LLM plan L0).

A scenario is a flat list of events:

    {"op": "remember", "fact": <fact id>, "text": ..., "session": s}
    {"op": "sleep", "n_blocks": n}
    {"op": "query", "cue": ..., "target": <fact id or None>, "kind": ...}

Facts are (person, attribute) -> value statements. Each session stores some
*target* facts (queried later), some *distractors* (never queried) and some
*near-duplicates* of earlier targets: the same attribute for another person,
or a sibling attribute for the same person ("dentist" vs "vet" appointment).
After each session comes a sleep; queries probe facts from the current and
earlier sessions, so lag (events since storage) and sleeps-since-storage vary.

Query kinds:
    direct      cue reuses the statement's wording
    paraphrase  cue shares few content words with the statement
    near_dup    direct cue for a target that has a near-duplicate stored
    negative    (person, attribute) never stored: any confident hit is a false recall

Generation is deterministic for a given seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

PEOPLE = ["Alice", "Bruno", "Chen", "Dana", "Elif", "Farid", "Greta", "Hiro",
          "Ines", "Jonas", "Kemal", "Lena", "Marta", "Nikolai", "Olga", "Pavel"]

# attribute -> (statement template, direct cue, paraphrase cue, value pools)
ATTRIBUTES: dict[str, tuple[str, str, str, dict[str, list[str]]]] = {
    "dentist": ("{p}'s dentist appointment is on {day} at {time}.",
                "When is {p}'s dentist appointment?",
                "What time does {p} see the tooth doctor?",
                {"day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                 "time": ["9am", "11am", "2pm", "4pm", "6pm"]}),
    "vet": ("{p}'s vet appointment is on {day} at {time}.",
            "When is {p}'s vet appointment?",
            "When does {p} take the animal to the doctor?",
            {"day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
             "time": ["8am", "10am", "1pm", "3pm", "5pm"]}),
    "car": ("{p} drives a {color} {model}.",
            "What car does {p} drive?",
            "Which vehicle does {p} own?",
            {"color": ["red", "blue", "silver", "black", "green"],
             "model": ["Volvo", "Skoda", "Toyota", "Fiat", "Tesla"]}),
    "pet": ("{p} has a {animal} named {name}.",
            "What is the name of {p}'s {animal}?",
            "What do they call the animal living with {p}?",
            {"animal": ["cat", "dog", "parrot", "rabbit"],
             "name": ["Pixel", "Biscuit", "Nemo", "Luna", "Otto"]}),
    "hometown": ("{p} grew up in {city}.",
                 "Where did {p} grow up?",
                 "What is {p}'s home city from childhood?",
                 {"city": ["Kazan", "Porto", "Lyon", "Gdansk", "Bilbao", "Tartu"]}),
    "allergy": ("{p} is allergic to {food}.",
                "What is {p} allergic to?",
                "Which food makes {p} sick?",
                {"food": ["peanuts", "shellfish", "gluten", "kiwi", "sesame"]}),
    "birthday": ("{p}'s birthday is on {month} {day}.",
                 "When is {p}'s birthday?",
                 "On what date was {p} born?",
                 {"month": ["March", "June", "August", "November"],
                  "day": ["3", "11", "19", "27"]}),
    "project": ("{p} is working on the {proj} project.",
                "Which project is {p} working on?",
                "What does {p} spend work hours on these days?",
                {"proj": ["Orion", "Basalt", "Heron", "Quartz", "Juniper"]}),
}

# sibling attributes: a near-duplicate for the same person
SIBLINGS = {"dentist": "vet", "vet": "dentist"}


@dataclass(frozen=True)
class Fact:
    fact_id: str
    person: str
    attribute: str
    values: tuple[tuple[str, str], ...]

    def _fmt(self, template: str) -> str:
        return template.format(p=self.person, **dict(self.values))

    @property
    def text(self) -> str:
        return self._fmt(ATTRIBUTES[self.attribute][0])

    def cue(self, kind: str) -> str:
        return self._fmt(ATTRIBUTES[self.attribute][2 if kind == "paraphrase" else 1])


def generate(seed: int = 0, n_sessions: int = 4, targets_per_session: int = 4,
             distractors_per_session: int = 4, near_dups_per_session: int = 2,
             negatives_per_session: int = 2, sleep_blocks: int = 1) -> list[dict]:
    rng = random.Random(seed)
    used: set[tuple[str, str]] = set()
    counter = 0

    def new_fact(person: str | None = None, attribute: str | None = None) -> Fact | None:
        nonlocal counter
        pairs = [(p, a) for p in PEOPLE for a in ATTRIBUTES
                 if (p, a) not in used
                 and (person is None or p == person)
                 and (attribute is None or a == attribute)]
        if not pairs:
            return None
        p, a = rng.choice(pairs)
        used.add((p, a))
        pools = ATTRIBUTES[a][3]
        counter += 1
        return Fact(f"f{counter:03d}", p, a,
                    tuple((k, rng.choice(v)) for k, v in sorted(pools.items())))

    events: list[dict] = []
    targets: list[Fact] = []
    has_near_dup: set[str] = set()

    def store(f: Fact, session: int, role: str) -> None:
        events.append({"op": "remember", "fact": f.fact_id, "text": f.text,
                       "session": session, "role": role,
                       "person": f.person, "attribute": f.attribute})

    for s in range(n_sessions):
        stored: list[tuple[Fact, str]] = []
        for _ in range(targets_per_session):
            if (f := new_fact()) is not None:
                targets.append(f)
                stored.append((f, "target"))
        for _ in range(distractors_per_session):
            if (f := new_fact()) is not None:
                stored.append((f, "distractor"))
        # near-duplicates of targets from this or earlier sessions
        for orig in rng.sample(targets, min(near_dups_per_session, len(targets))):
            sib = SIBLINGS.get(orig.attribute)
            f = (new_fact(person=orig.person, attribute=sib) if sib
                 else new_fact(attribute=orig.attribute))
            if f is not None:
                has_near_dup.add(orig.fact_id)
                stored.append((f, "near_dup"))
        rng.shuffle(stored)
        for f, role in stored:
            store(f, s, role)

        # queries: every target stored so far, alternately direct/paraphrase,
        # so each fact is probed at several lags
        for i, t in enumerate(targets):
            if t.fact_id in has_near_dup:
                kind = "near_dup"
            else:
                kind = "direct" if (i + s) % 2 == 0 else "paraphrase"
            events.append({"op": "query", "cue": t.cue(kind), "target": t.fact_id,
                           "kind": kind, "session": s})
        for _ in range(negatives_per_session):
            free = [(p, a) for p in PEOPLE for a in ATTRIBUTES if (p, a) not in used]
            if not free:
                break
            p, a = rng.choice(free)
            ghost = Fact("none", p, a, tuple((k, v[0]) for k, v in sorted(ATTRIBUTES[a][3].items())))
            events.append({"op": "query", "cue": ghost.cue("direct"), "target": None,
                           "kind": "negative", "session": s})
        events.append({"op": "sleep", "n_blocks": sleep_blocks})
    return events


def save(events: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(events, f, indent=1)


def load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)
