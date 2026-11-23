"""Run lexical retrieval against a frozen corpus and a deterministic fixture model."""

import json
import re

from pydantic import JsonValue

from tokenbench.trace import TraceSink

_RECORDS = (
    ("alder-01", "Alder Point", "AP-104", "ceramics", "Mira"),
    ("amber-02", "Amber Quay", "AQ-211", "textiles", "Ivo"),
    ("aspen-03", "Aspen Grove", "AG-307", "glassware", "Nora"),
    ("birch-04", "Birch Point", "BP-409", "prints", "Kai"),
    ("cedar-05", "Cedar Vale", "CV-502", "furniture", "Tess"),
    ("elm-06", "Elm Grove", "EG-618", "maps", "Paz"),
    ("fern-07", "Fern Bank", "FB-723", "metalwork", "Lena"),
    ("hazel-08", "Hazel Quay", "HQ-825", "photographs", "Omar"),
    ("ivy-09", "Ivy Bank", "IB-906", "sculpture", "Uma"),
    ("juniper-10", "Juniper Vale", "JV-031", "instruments", "Ravi"),
    ("larch-11", "Larch Point", "LP-115", "manuscripts", "Mei"),
    ("maple-12", "Maple Quay", "MQ-222", "posters", "Sol"),
)
_FIELDS = ("catalog code", "collection", "coordinator")
_CORPUS = [
    {
        "source_id": source_id,
        "title": title,
        **dict(zip(_FIELDS, (code, collection, coordinator), strict=True)),
    }
    for source_id, title, code, collection, coordinator in _RECORDS
]


def terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def run(task_input: dict[str, JsonValue], trace: TraceSink) -> JsonValue:
    """Retrieve documents, assemble a prompt, and extract a cited answer locally."""
    query = task_input.get("query")
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    query_terms = terms(query)
    documents = sorted(_CORPUS, key=lambda document: -len(terms(document["title"]) & query_terms))[
        :3
    ]
    prompt = json.dumps({"query": query, "documents": documents}, sort_keys=True)
    with trace.model_call() as call:
        answer = fixture_completion(prompt)
        call.estimate_usage(len(prompt.split()), len(json.dumps(answer).split()))
    return answer


def fixture_completion(prompt: str) -> JsonValue:
    payload = json.loads(prompt)
    query = payload["query"].casefold()
    for document in payload["documents"]:
        if document["title"].casefold() in query:
            for field in _FIELDS:
                if field in query:
                    return {"answer": document[field], "source_id": document["source_id"]}
    return None
