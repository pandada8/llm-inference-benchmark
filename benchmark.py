import asyncio
import httpx
import argparse
import time
import orjson
import statistics
import datetime
import os
import numpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['qwen', 'llama'])
parser.add_argument('--backend', choices=['vllm'])
parser.add_argument('--batch', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
parser.add_argument('--endpoint')
parser.add_argument('--duration', type=int, default=60)
args = parser.parse_args()

QWEN_PROMPT = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
output from 1 to 100<|im_end|>
<|im_start|>assistant
'''

def get_endpoint():
    if args.backend == 'vllm':
        return args.endpoint + '/worker_generate_stream'
    elif args.backend in ['tgi', 'lorax']:
        return args.endpoint + '/generate'

def get_body():
    prompt = ''
    stop = []
    if args.model == 'qwen':
        prompt = QWEN_PROMPT
        stop = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    if args.backend == 'vllm':
        return {
            "prompt": prompt,
            "max_new_tokens": 1024,
            "temperature": 0,
            "stop": stop,
            'echo': False,
        }

def count_token(chunk):
    if args.backend == 'vllm':
        # print(repr(chunk))
        return orjson.loads(chunk[:-1])['usage']['completion_tokens']

async def requests_worker():
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(5, read=None)) as client:
        endpoint = get_endpoint()             
        body = get_body()
        tokens = []
        ticks = []

        while True:
            ticks.append([time.perf_counter()])
            tokens.append(0)
            sys.stdout.write('.')
            sys.stdout.flush()

            async with client.stream('POST', endpoint, json=body) as r:
                async for chunk in r.aiter_text():
                    tt = time.perf_counter() 
                    ticks[-1].append(tt)
                    delta = tt - start
                    if delta > args.duration:
                        return ticks, tokens
                    if chunk == '':
                        continue
                    tokens[-1] = count_token(chunk)
                    
async def main():
    print(args)
    final_result = []
    for batch_no, batch in enumerate(args.batch):
        print(f'--- process with batch size {batch}')
        workers = []
        for i in range(batch):
            workers.append(requests_worker())
        total_tokens = 0
        first_token_latencies = []
        non_first_token_latency = []
        all_ticks = []
        for tick_group, tokens in await asyncio.gather(*workers):
            total_tokens += sum(tokens)
            all_ticks.append(tick_group)
            for ticks in tick_group:
                if len(ticks) == 1:
                    continue
                diff = numpy.diff(ticks).tolist()
                first_token_latencies.append(diff[0])
                non_first_token_latency.extend(diff[1:])
                
        final_result.append({
            'batch': batch,
            'total_token': total_tokens,
            'token_per_s': total_tokens / args.duration,
            'avg_first_token_latency': statistics.mean(first_token_latencies),
            'min_token_latency': min(non_first_token_latency),
            'max_token_latency': max(non_first_token_latency),
            'median_token_latency': statistics.median(non_first_token_latency),
            'avg_token_latency': statistics.mean(non_first_token_latency),
        })
        print()
        print(final_result[-1])
        final_result[-1]['raw_ticks'] = all_ticks
        if batch_no < len(args.batch) - 1:
            await asyncio.sleep(5)

    
    os.makedirs('result', exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    with open(f'result/{now_str}_{args.model}_{args.backend}.json', 'wb') as fp:
        fp.write(orjson.dumps({
            'model': args.model,
            'backend': args.backend,
            "time": now_str,
            "results": final_result,
            "duration": args.duration
        }))

if __name__ == '__main__':
    asyncio.run(main())
