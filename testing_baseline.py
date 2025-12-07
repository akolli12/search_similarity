import re
import json
import time
import os
from typing import Any, Dict, Tuple, Optional, List
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
import sglang
from datasets import load_dataset


CONFIG = {
    # Replace these with the real endpoints you mentioned (you said you created them previously)
    'SUBMIT_PARAGRAPHS_URL': os.environ.get('SUBMIT_PARAGRAPHS_URL', 'http:/localhost:8000/submit_paragraphs'),
    'SEARCH_URL': os.environ.get('SEARCH_URL', 'http:/localhost:8000/search'),
    'API_KEY': os.environ.get('API_KEY', None),  # optional header param

    # Musique dataset path (expected to be a JSONL or CSV with fields: id, question, answer)
    'MUSIQUE_PATH': os.environ.get('MUSIQUE_PATH', './musique_dev.jsonl'), #CHANGE THIS

    # SGLang / model options (adjust according to your sglang setup)
    'SGLANG_MODEL_NAME': os.environ.get('SGLANG_MODEL_NAME', 'qwen-4b-instruct'),
    'SGLANG_HOST': os.environ.get('SGLANG_HOST', None),

    # Retry / timeouts
    'HTTP_TIMEOUT': 30,
    'TOOL_CALL_TIMEOUT': 60,
}

HEADERS = {}
if CONFIG['API_KEY']:
    HEADERS['Authorization'] = f"Bearer {CONFIG['API_KEY']}"

def http_post_json(url: str, payload: dict, headers: Optional[dict] = None, timeout: int = 30) -> dict:
    headers = headers or {}
    headers.update({'Content-Type': 'application/json'})
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def normalize_answer(s: str) -> str:
    return " ".join(s.lower().strip().split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)



def submit_paragraphs_tool(paragraphs: List[dict], **kwargs) -> dict:
    """Submit paragraphs to your endpoint. paragraphs is a list of dicts formatted for your API.
    Returns the JSON response from the server.
    """
    payload = {'paragraphs': paragraphs}
    payload.update(kwargs)
    resp = http_post_json(CONFIG['SUBMIT_PARAGRAPHS_URL'], payload, headers=HEADERS, timeout=CONFIG['HTTP_TIMEOUT'])
    return resp


def search_tool(query: str, top_k: int = 5, **kwargs) -> dict:
    payload = {'query': query, 'top_k': top_k}
    payload.update(kwargs)
    resp = http_post_json(CONFIG['SEARCH_URL'], payload, headers=HEADERS, timeout=CONFIG['HTTP_TIMEOUT'])
    return resp


def register_tools_with_sglang(client: Any = None):
    if client is None:
        if sglang is None:
            raise RuntimeError('sglang not installed or client not provided')
        client = sglang.Client(host=CONFIG['SGLANG_HOST'], model=CONFIG['SGLANG_MODEL_NAME'])

    @client.tool(name='submit_paragraphs')
    def _submit_paragraphs_tool(paragraphs: list, **kwargs):
        return submit_paragraphs_tool(paragraphs, **kwargs)

    @client.tool(name='search')
    def _search_tool(query: str, top_k: int = 5, **kwargs):
        return search_tool(query, top_k=top_k, **kwargs)

    return client



TOOL_CALL_REGEX = re.compile(r"<tool-call\s+tool=\"(?P<tool>[^\"]+)\">\s*(?P<json>\{.+?\})\s*</tool-call>", re.DOTALL)


def parse_tool_call(text: str) -> Optional[Tuple[str, dict, str]]:
    """
    Return (tool_name, params_dict, remainder_text) if a tool call is found.
    remainder_text is the original text without the tool-call block.
    """
    m = TOOL_CALL_REGEX.search(text)
    if m:
        tool_name = m.group('tool')
        json_blob = m.group('json')
        try:
            params = json.loads(json_blob)
        except Exception as e:
            # try to fix common issues (single quotes)
            try:
                params = json.loads(json_blob.replace("'", '"'))
            except Exception:
                raise ValueError(f"Failed to parse tool JSON: {e}\n>>> {json_blob}")
        remainder = text[:m.start()] + text[m.end():]
        return tool_name, params, remainder

    # Try the JSON-in-text style
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and 'tool' in parsed and 'args' in parsed:
            return parsed['tool'], parsed['args'], ''
    except Exception:
        pass

    return None


class ReActAgent:
    def __init__(self, client: Any = None, max_steps: int = 3):
        if sglang is None and client is None:
            raise RuntimeError('sglang SDK not available; install or pass a client')
        self.client = client
        self.max_steps = max_steps

    def generate(self, msgs: List[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Wraps the sglang model generate call. Keep this small so you can swap clients easily."""
        if self.client is not None:
            resp = self.client.generate(messages=msgs, max_tokens=max_tokens, temperature=temperature)
            if isinstance(resp, dict):
                return resp.get('content') or resp.get('text') or json.dumps(resp)
            return str(resp)
        else:
            raise RuntimeError('No sglang client supplied')

    def run_once(self, question: str) -> Tuple[str, List[dict]]:
        """Run the 2-step ReAct flow you described for a single question and return final answer + trace.
        Trace is list of message dicts including tool responses.
        """
        # Step 0: initial message
        msgs = [{'role': 'user', 'content': question}]
        trace = []

        # Step 1: model produces first-round output (should include tool call)
        resp1 = self.generate(msgs)
        trace.append({'role': 'assistant', 'content': resp1})

        # Attempt to parse a tool call from resp1
        parsed = parse_tool_call(resp1)
        if not parsed:
            # If no tool call, we treat resp1 as final answer
            return resp1, trace

        tool_name, tool_args, remainder = parsed

        # Dispatch to our known tools
        tool_resp = None
        if tool_name == 'search':
            tool_resp = search_tool(**tool_args)
        elif tool_name == 'submit_paragraphs':
            tool_resp = submit_paragraphs_tool(**tool_args)
        else:
            raise ValueError(f'Unknown tool requested: {tool_name}')

        # Step 2: feed the tool response back into the model
        # Use role 'tool' if your SGLang client supports it; otherwise use 'user'
        msgs = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': resp1},
            {'role': 'tool', 'content': json.dumps(tool_resp)}
        ]
        trace.append({'role': 'tool', 'content': tool_resp})

        # Final generate
        resp2 = self.generate(msgs)
        trace.append({'role': 'assistant', 'content': resp2})
        return resp2, trace

def load_musique(path: str) -> List[dict]:
    if path.endswith('.jsonl') or path.endswith('.json'):
        data = []
        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data.append(json.loads(line))
        return data
    elif path.endswith('.csv') or path.endswith('.tsv'):
        df = pd.read_csv(path)
        return df.to_dict(orient='records')
    else:
        raise ValueError('Unsupported dataset path extension')


def run_experiments(agent: ReActAgent, dataset: List[dict], save_path: str = './results.csv') -> pd.DataFrame:
    records = []
    for ex in tqdm(dataset, desc='Examples'):
        qid = ex.get('id') or ex.get('question_id') or None
        question = ex['question']
        gold = ex.get('answer', '')
        try:
            pred, trace = agent.run_once(question)
        except Exception as e:
            pred = f'[[ERROR: {e}]]'
            trace = []
        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)
        records.append({'id': qid, 'question': question, 'prediction': pred, 'gold': gold, 'em': em, 'f1': f1, 'trace': trace})

        time.sleep(0.1)

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    return df



if __name__ == '__main__':
    # Create or connect to SGLang client
    client = None
    if sglang is not None and CONFIG['SGLANG_HOST']:
        client = sglang.Client(host=CONFIG['SGLANG_HOST'], model=CONFIG['SGLANG_MODEL_NAME'])
        try:
            register_tools_with_sglang(client)
        except Exception as e:
            print('Warning: failed to register tools with sglang client:', e)

    agent = ReActAgent(client=client)

    # Load dataset
    ds = load_dataset("dgslibisey/MuSiQue", split="train") #load_musique(CONFIG['MUSIQUE_PATH'])
    # Run experiments
    out_df = run_experiments(agent, ds, save_path='./musique_react_results.csv')
    print('Done. Results saved to ./musique_react_results.csv')



