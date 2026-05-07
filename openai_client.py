from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from openai import OpenAI


def canonicalize_url(url):
    """Drop tracking params (utm_*, openai-specific) and the fragment so the
    same page surfaced in web_search and cited in the answer compares equal."""
    if not url:
        return url
    parts = urlsplit(url)
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith("utm_")]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), ""))


class OpenAIClient:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)

    def search(
        self,
        user_content,
        allowed_domains=None,
        model="gpt-5-mini",
        reasoning_effort="minimal",
    ):
        tool_config = {"type": "web_search"}
        if allowed_domains:
            tool_config["filters"] = {"allowed_domains": allowed_domains}

        kwargs = {
            "model": model,
            "tools": [tool_config],
            # Force web_search every call so behavior is comparable across queries
            # (otherwise transactional/conversational queries skip search entirely).
            "tool_choice": {"type": "web_search"},
            "include": ["web_search_call.action.sources"],
            "input": user_content,
        }
        # GPT-5 family are reasoning models: temperature/top_p/seed are rejected.
        # `minimal` is the lowest effort the API exposes.
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        response = self.client.responses.create(**kwargs)
        return self.extraction(response)

    def extraction(self, response):
        sources = []
        web_query = None
        citations = []
        output_text = getattr(response, "output_text", "") or ""

        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "web_search_call":
                action = getattr(item, "action", None)
                if action is None:
                    continue
                if web_query is None:
                    web_query = getattr(action, "query", None)
                for src in getattr(action, "sources", None) or []:
                    url = canonicalize_url(getattr(src, "url", None))
                    if url and url not in sources:
                        sources.append(url)
            elif item_type == "message":
                for content in getattr(item, "content", None) or []:
                    for ann in getattr(content, "annotations", None) or []:
                        url = canonicalize_url(getattr(ann, "url", None))
                        if url:
                            citations.append(url)

        for url in citations:
            if url not in sources:
                sources.append(url)

        labels = [sources.index(url) for url in citations]

        return {
            "query_used": web_query,
            "output": output_text,
            "sources": sources,
            "citations": citations,
            "labels": labels,
        }
