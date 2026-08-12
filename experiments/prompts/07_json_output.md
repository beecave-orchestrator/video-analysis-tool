# 07 JSON Output

## Goal

Try constrained machine-readable output to see whether structured captions improve tag extraction.

## Prompt

Return ONLY valid JSON, no markdown, no explanation. Use this exact schema: {"description": "brief description", "people": int, "clothing": "clothing state", "acts": ["..."], "positions": ["..."], "objects": ["..."], "setting": "..."}. Use explicit sexual vocabulary in the lists.
