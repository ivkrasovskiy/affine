import sys; sys.path.append('/workspace/affine/')
import re
import time
import random
import asyncio
import affine as af
from typing import Any, Dict, Optional, Tuple
import functools
from typing import Callable
import requests
import os
import json





class RetryNeeded(ValueError):
    pass

def retry(fn: Callable | int = 5, retries: int | None = None) -> Callable:
    if retries is None:
        if not isinstance(fn, int):
            raise ValueError("retry() has to be closed when used as a decorator")
        return functools.partial(retry, retries=fn)

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        for i in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                af.logger.trace(f'Error encountered: {e} - Retry {i}/{retries}')
                if i == retries - 1:
                    raise

    return _wrapped


@retry()
def fallback_models(min_completion_cost: float = 0.0, min_context: int = 65536,
                    owners: tuple = ('chutesai',), max_completion_cost: float = 1e9):
    models = requests.get('https://llm.chutes.ai/v1/models')
    models.raise_for_status()
    models = models.json()['data']

    model_ids = []
    for mod in models:
        if not max_completion_cost >= mod['pricing']['completion'] >= min_completion_cost:
            continue
        if mod.get('context_length', mod.get('max_model_len', 0)) < min_context:
            continue
        if 'text' not in mod.get('input_modalities', []):
            continue
        if 'text' not in mod.get('output_modalities', []):
            continue
        model_ids.append(mod['id'])
    if not model_ids:
        raise RetryNeeded
    return model_ids

N_SAMPLES = 1000
DATA_DIR = '/workspace/data/abd/'
os.makedirs(DATA_DIR, exist_ok=True)
MODELS = fallback_models(max_completion_cost=0.15)
PROMPT_TEMPLATE = """You are a programming expert. Given a Python program and its expected output, you need to determine the exact input that would produce this output.

Program:
```python
{program}
```

Expected Output:
```
{output}
```

Task: Analyze the program to understand what input format it expects from stdin, then provide the input data that would produce the expected output.

You can provide any explanations, analysis, or reasoning you want. However, you MUST include the input data within <INPUT> </INPUT> tags.

Format the input data like this:
<INPUT>
[input data here - each line on a separate line as the program expects]
</INPUT>

I will extract only the content between these tags.

Requirements for the input data within the tags:
1. Each line of input should be on a separate line
2. Use the exact format the program expects  
3. Provide the raw input values that should be fed to stdin
4. Do not include any prefixes or extra formatting within the INPUT tags

Please analyze the program and provide the required input:"""

INPUT_GENERATION_PROMPT = """Given this Python program and an example of how it works, generate a NEW valid input that would be accepted by the program:

Program:
```python
{program}
```

Example:
Input: {example_input}
Output: {example_output}

Now generate a NEW input that would work with this program. Make it different from the example but follow the same format and pattern.

Requirements:
1. Each line of input should be on a separate line
2. Use the exact format the program expects from stdin
3. Provide only raw input values
4. Do not include any prefixes or extra formatting
5. Make it different from the example input

Format your response with <INPUT> </INPUT> tags like this:
<INPUT>
[your new input data here]
</INPUT>

Please generate a valid input:"""

dataset = af.singleton('rl-python', lambda: af.utils.R2BufferedDataset(
        dataset_name="satpalsr/rl-python",
        buffer_size=5,
        max_batch=5,
))

def extract_input_from_response(response: str) -> str:
    """Pull out the last <INPUT>…</INPUT> block."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)
    matches = re.findall(r"<INPUT>(.*?)</INPUT>", response, re.IGNORECASE | re.DOTALL)
    if not matches:
        af.logger.trace("No <INPUT> tags found in response.")
        return ""
    lines = [ln.rstrip() for ln in matches[-1].strip().splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    extracted_input = "\n".join(lines)
    return extracted_input

def validate_input_for_program(program: str, inp: str) -> bool:
    """Heuristic: ensure at least as many lines as input() calls."""
    calls = program.count("input()")
    lines = inp.splitlines() if inp else []
    if "for _ in range(int(input()))" in program and lines and lines[0].isdigit():
        valid = len(lines) > int(lines[0])
        af.logger.trace(f"Validation result for loop-based input: {valid}")
        return valid
    valid = len(lines) >= calls
    return valid


executor = af.utils.ProgramExecutor()

async def amain():
    from tqdm.auto import tqdm, trange
    filepath = f"{DATA_DIR}/{time.time()}.jsonl"

    samples = []
    
    for _ in trange(N_SAMPLES):
        MODEL = random.choice(MODELS)
        
        sample = await dataset().get()
        program, example_in, example_out = [sample[key] for key in ['program', 'inputs', 'output']]
        
        prompt = INPUT_GENERATION_PROMPT.format(
            program=program,
            example_input=example_in,
            example_output=example_out
        )
        
        resp = await af.query(prompt, model=MODEL)
        llm_resp = resp.response
        if not llm_resp:
            continue
        
        gen_input = extract_input_from_response(llm_resp)
        if not validate_input_for_program(program, gen_input):
            continue
        
        # Ensure final newline for stdin-based programs
        if gen_input and not gen_input.endswith("\n"):
            gen_input += "\n"
        
        output, error = executor.execute(program, gen_input)
        # loop = asyncio.get_running_loop()
        # output, error = await loop.run_in_executor(None, executor.execute, program, gen_input)
    
        item = {**sample,
                        'synth_input': gen_input,
                        'synth_output': output,
                        'synth_error': error,
                        'synth_model': MODEL,
                        # 'synth_response': llm_resp
                       }
        samples.append(item)
        line = json.dumps(item, ensure_ascii=False)
        
        # normal blocking file I/O inside a small critical section
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main():    
    asyncio.run(amain())


if __name__ == '__main__':
    main()
