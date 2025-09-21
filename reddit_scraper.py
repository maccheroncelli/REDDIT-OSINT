"""Simple Reddit scraper for collecting a user's posts and comments.

This module provides a command line interface that retrieves the combined
overview feed for a Reddit user and separates submissions (posts) and
comments.  The script does not require API credentials and uses Reddit's
public JSON endpoints instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib import error, parse, request

BASE_URL = "https://www.reddit.com/user/{username}/.json"
DEFAULT_USER_AGENT = "REDDIT-OSINT-scraper/0.1 (by u/example)"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
DEFAULT_ANALYSIS_LIMIT = 30
MAX_SNIPPET_LENGTH = 280


@dataclass
class RedditEntry:
    """Representation of a Reddit post or comment."""

    id: str
    type: str
    permalink: str
    created_utc: float
    score: int
    subreddit: str
    url: Optional[str]
    title: Optional[str]
    body: Optional[str]

    @classmethod
    def from_listing(cls, kind: str, data: Dict) -> "RedditEntry":
        entry_type = "comment" if kind == "t1" else "post"
        permalink = data.get("permalink")
        if permalink and permalink.startswith("/"):
            permalink = f"https://www.reddit.com{permalink}"

        return cls(
            id=data.get("id", ""),
            type=entry_type,
            permalink=permalink or "",
            created_utc=float(data.get("created_utc", 0.0)),
            score=int(data.get("score", 0)),
            subreddit=data.get("subreddit", ""),
            url=data.get("url_overridden_by_dest")
            or data.get("url")
            if entry_type == "post"
            else None,
            title=data.get("title") if entry_type == "post" else None,
            body=data.get("selftext")
            if entry_type == "post"
            else data.get("body"),
        )


class OllamaError(RuntimeError):
    """Raised when interaction with the Ollama API fails."""


def _normalize_entries(
    username: str, posts: Sequence[RedditEntry], comments: Sequence[RedditEntry]
) -> List[Dict[str, Any]]:
    """Convert Reddit entries into normalized JSON-serializable dictionaries."""

    records: List[Dict[str, Any]] = []

    def _entry_to_record(entry: RedditEntry) -> Dict[str, Any]:
        created_iso = (
            datetime.fromtimestamp(entry.created_utc, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

        text_parts = [part.strip() for part in (entry.title, entry.body) if part]
        combined_text = "\n\n".join(filter(None, text_parts))

        return {
            "username": username,
            "id": entry.id,
            "type": entry.type,
            "subreddit": entry.subreddit,
            "created_utc": entry.created_utc,
            "created_iso": created_iso,
            "score": entry.score,
            "permalink": entry.permalink,
            "url": entry.url,
            "title": entry.title,
            "body": entry.body,
            "text": combined_text,
            "word_count": len(combined_text.split()) if combined_text else 0,
        }

    for collection in (posts, comments):
        for item in collection:
            records.append(_entry_to_record(item))

    records.sort(key=lambda item: item["created_utc"], reverse=True)
    return records


def _write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _shorten_text(text: str, limit: int = MAX_SNIPPET_LENGTH) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _build_analysis_prompt(
    username: str, records: Sequence[Dict[str, Any]], limit: int
) -> str:
    selected = list(records[:limit]) if limit > 0 else []
    if not selected:
        activity_section = "No activity was captured for this user. Respond with 'unknown' for all requested fields."
    else:
        lines = []
        for record in selected:
            snippet_source = record.get("text") or record.get("title") or ""
            snippet = _shorten_text(snippet_source) if snippet_source else "(no textual content provided)"
            lines.append(
                f"- [{record['type']}] r/{record['subreddit']} on {record['created_iso']} (score {record['score']}): {snippet}"
            )
        activity_section = "\n".join(lines)

    prompt = f"""
You are an OSINT analyst reviewing public Reddit activity for the user u/{username}.
Analyse the behavioural signals in order to provide:
- An overall summary of the user's activity and tone.
- The general sentiment the user expresses.
- A list of hobbies or interests suggested by the content.
- Whether the user appears left-leaning, right-leaning, centrist, or unclear on politics.
- Whether the user appears progressive, conservative, or unclear on social issues.
- Any notable recurring topics or communities.
- Any additional insights you can infer.

Base your conclusions only on the provided Reddit activity. If a field cannot be inferred, respond with "unknown" for that field.
Return **only** a JSON object with the following keys: "summary", "sentiment", "hobbies", "political_alignment", "social_issues_alignment", "notable_topics", and "additional_insights". Lists should contain strings.

Reddit activity:
{activity_section}
"""
    return prompt.strip()


def _call_ollama(prompt: str, model: str, endpoint: str) -> str:
    url = endpoint.rstrip("/") + "/api/generate"
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
    ).encode("utf-8")

    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            response_payload = json.load(resp)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise OllamaError(f"Failed to connect to Ollama at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OllamaError("Received non-JSON response from Ollama.") from exc

    text = response_payload.get("response", "")
    if not text:
        raise OllamaError("Ollama returned an empty response.")
    return text


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3]
        stripped = stripped.strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def _parse_analysis_response(text: str) -> Any:
    cleaned = _strip_json_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return text


def fetch_user_activity(
    username: str,
    limit: Optional[int] = None,
    pause: float = 2.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Dict[str, List[RedditEntry]]:
    """Fetch a Reddit user's posts and comments.

    Parameters
    ----------
    username:
        Reddit username without the ``u/`` prefix.
    limit:
        Maximum total number of items (posts + comments) to retrieve. ``None``
        means no explicit limit.
    pause:
        Seconds to wait between paginated requests to respect Reddit's API
        rate limits.
    user_agent:
        User agent string to present to Reddit. You should set this to
        something descriptive for your use case.
    """

    headers = {"User-Agent": user_agent}
    params: Dict[str, Optional[str]] = {"limit": "100", "after": None}
    remaining = limit
    posts: List[RedditEntry] = []
    comments: List[RedditEntry] = []

    while True:
        if remaining is not None:
            if remaining <= 0:
                break
            params["limit"] = str(min(100, remaining))

        payload = _request_with_retry(BASE_URL.format(username=username), headers, params)
        data = payload.get("data", {})
        children: Iterable[Dict] = data.get("children", [])

        if not children:
            break

        for child in children:
            kind = child.get("kind", "")
            if kind not in {"t1", "t3"}:
                continue

            entry = RedditEntry.from_listing(kind, child.get("data", {}))
            if entry.type == "post":
                posts.append(entry)
            else:
                comments.append(entry)

            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break
        else:
            # Only executed if the for-loop didn't break meaning limit not reached
            pass

        if remaining is not None and remaining <= 0:
            break

        after = data.get("after")
        if not after:
            break
        params["after"] = after

        time.sleep(max(0.0, pause))

    return {"posts": posts, "comments": comments}


def _request_with_retry(
    url: str,
    headers: Dict[str, str],
    params: Dict[str, Optional[str]],
    retries: int = 3,
) -> Dict:
    """Perform a GET request with simple retry logic for 429/5xx responses."""

    backoff = 1.0
    for _ in range(retries):
        query = parse.urlencode({k: v for k, v in params.items() if v is not None})
        full_url = f"{url}?{query}" if query else url
        req = request.Request(full_url, headers=headers)

        try:
            with request.urlopen(req, timeout=15) as resp:
                return json.load(resp)
        except error.HTTPError as exc:
            status = exc.code
            if status == 404:
                username = url.split("/user/")[1].split("/")[0]
                raise ValueError(f"User '{username}' not found.") from None
            if status == 429:
                retry_after = exc.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after else backoff
                time.sleep(wait_time)
                backoff *= 2
                continue
            if 500 <= status < 600:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except error.URLError:
            time.sleep(backoff)
            backoff *= 2
            continue

    raise RuntimeError(f"Failed to retrieve data after {retries} attempts.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scrape Reddit posts/comments, normalise them, and optionally analyse the activity with a local LLM."
    )
    parser.add_argument("username", help="Reddit username (without the u/ prefix)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum total number of items (posts + comments) to fetch.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="Seconds to wait between requests to respect API limits (default: 2).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save the scraped data as JSON. Use '-' to stream to stdout.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="Custom user-agent string to use for requests.",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Path to store normalised activity as JSON Lines. Defaults to '<username>_activity.jsonl'.",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Disable writing the normalised JSONL file.",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip the local LLM analysis step.",
    )
    parser.add_argument(
        "--analysis-output",
        type=str,
        default=None,
        help="Optional file path to save the LLM analysis (JSON). If omitted, the analysis prints to stdout.",
    )
    parser.add_argument(
        "--analysis-limit",
        type=int,
        default=DEFAULT_ANALYSIS_LIMIT,
        help=f"Maximum number of recent items to send to the LLM (default: {DEFAULT_ANALYSIS_LIMIT}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model to use for summarisation (default: qwen2.5:7b-instruct).",
    )
    parser.add_argument(
        "--ollama-endpoint",
        type=str,
        default=DEFAULT_OLLAMA_ENDPOINT,
        help="Base URL for the Ollama API (default: http://localhost:11434).",
    )

    args = parser.parse_args(argv)

    if args.analysis_limit is not None and args.analysis_limit < 0:
        parser.error("--analysis-limit must be greater than or equal to 0.")

    try:
        results = fetch_user_activity(
            username=args.username,
            limit=args.limit,
            pause=args.pause,
            user_agent=args.user_agent,
        )
    except ValueError as exc:
        parser.error(str(exc))
    except error.HTTPError as exc:
        parser.error(f"HTTP error: {exc}")
    except RuntimeError as exc:
        parser.error(str(exc))

    payload = {
        "username": args.username,
        "posts": [asdict(entry) for entry in results["posts"]],
        "comments": [asdict(entry) for entry in results["comments"]],
    }

    normalized_records = _normalize_entries(
        args.username, results["posts"], results["comments"]
    )

    if not args.no_jsonl:
        jsonl_path = Path(args.jsonl).expanduser() if args.jsonl else Path(f"{args.username}_activity.jsonl")
        _write_jsonl(normalized_records, jsonl_path)
        print(f"Saved normalised activity to {jsonl_path}", file=sys.stderr)

    if args.output:
        if args.output == "-":
            json.dump(payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            output_path = Path(args.output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"Saved raw JSON export to {output_path}", file=sys.stderr)

    analysis_payload: Any = None
    if not args.no_analysis:
        prompt = _build_analysis_prompt(args.username, normalized_records, args.analysis_limit)
        try:
            analysis_text = _call_ollama(prompt, args.model, args.ollama_endpoint)
        except OllamaError as exc:
            parser.error(str(exc))

        analysis_payload = _parse_analysis_response(analysis_text)
        if not isinstance(analysis_payload, dict):
            print(
                "Warning: LLM response was not valid JSON; returning raw text instead.",
                file=sys.stderr,
            )

        if args.analysis_output:
            output_path = Path(args.analysis_output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fh:
                if isinstance(analysis_payload, dict):
                    json.dump(analysis_payload, fh, indent=2, ensure_ascii=False)
                    fh.write("\n")
                else:
                    fh.write(str(analysis_payload))
                    if not str(analysis_payload).endswith("\n"):
                        fh.write("\n")
            print(f"Saved analysis to {output_path}", file=sys.stderr)
        else:
            if isinstance(analysis_payload, dict):
                json.dump(analysis_payload, sys.stdout, indent=2, ensure_ascii=False)
                sys.stdout.write("\n")
            else:
                sys.stdout.write(str(analysis_payload) + "\n")
    elif not args.output:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
