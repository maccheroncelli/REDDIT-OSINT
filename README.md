# Reddit OSINT Scraper

This repository contains a lightweight Python script that collects the public
posts and comments for a Reddit user without requiring API credentials. The
tool normalises the activity into JSON Lines and can drive a local large
language model (via [Ollama](https://ollama.com/)) to summarise the user's
behaviour, sentiment, interests, and political leanings.

## Requirements

* Python 3.8+
* [Ollama](https://ollama.com/) running locally (default endpoint
  `http://localhost:11434`) with the models you intend to use. The tool defaults
  to `qwen2.5:7b-instruct` but any installed model can be selected.

## Usage

```bash
python reddit_scraper.py <username> [options]
```

Example:

```bash
python reddit_scraper.py spez --limit 100 --analysis-output spez_analysis.json
```

### Key options

* `--jsonl PATH` &ndash; location for the normalised JSONL export. Defaults to
  `<username>_activity.jsonl`. Use `--no-jsonl` to skip writing it.
* `--output FILE` &ndash; save the raw combined JSON payload (`-` streams to
  stdout). When `--no-analysis` is supplied and no output file is given, the raw
  JSON prints to stdout.
* `--limit N` &ndash; cap the total number of posts plus comments collected.
* `--pause SECONDS` &ndash; delay between paginated requests (default 2 seconds).
* `--model NAME` &ndash; Ollama model to drive analysis. Defaults to
  `qwen2.5:7b-instruct`.
* `--analysis-output FILE` &ndash; location to store the LLM analysis JSON. If
  omitted, the analysis prints to stdout.
* `--analysis-limit N` &ndash; how many of the most recent posts/comments to send
  to the LLM (default 30). Reduce this if you encounter context length issues.
* `--no-analysis` &ndash; skip the Ollama step entirely. This is useful when you
  just want the scraped/normalised data.

The normalised JSONL export contains one item per line with merged metadata and
text content to make downstream processing easier. When Ollama analysis is
enabled, the model is prompted to return a JSON object describing:

* Overall summary of the user's behaviour.
* Sentiment of their activity.
* Hobbies or interests.
* Political and social issue leanings.
* Notable topics or communities.
* Additional insights suggested by the content.

If the LLM responds with valid JSON, it is emitted as such; otherwise the raw
text is returned with a warning. You can install additional models with
`ollama pull <model>` and select them via `--model`.
