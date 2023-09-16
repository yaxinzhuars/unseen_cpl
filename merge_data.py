import argparse

def merge(file1, file2, outfile):
    data = []
    with open(file1) as f:
        for line in f.readlines():
            data.append(line)
    with open(file2) as f:
        for line in f.readlines():
            data.append(line)
    with open(outfile, 'w') as fw:
        for line in data:
            fw.write(line)

parser = argparse.ArgumentParser()
parser.add_argument('--file1', type=str)
parser.add_argument('--file2', type=str)
parser.add_argument('--outfile', type=str)

args = parser.parse_args()
merge(args.file1, args.file2, args.outfile)