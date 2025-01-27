import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='../data')
parser.add_argument('--chunk_num', type=int, default=8)
args = parser.parse_args()


dic = {}
for idx in range(args.chunk_num):
    data = json.load(open(f"{args.input_path}/{idx}.json", 'r'))
    dic.update(data)
    
json.dump(dic, open(f'{args.input_path}/merge.json', 'w'), indent=4)
