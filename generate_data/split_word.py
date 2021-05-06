import sys 
from pathlib import Path

target = sys.argv[1]

if __name__=="__main__":

    line_cnt = 0
    out = (Path(target).parent / 'word_split_result.txt').open('w', encoding='utf8')
    with Path(target).open('r', encoding='utf8') as f:
        line = f.readline().strip()
        while line:
            line_cnt += 1
            for word in line.split():
                out.write(word+'\n')
            line = f.readline().strip()
    print(line_cnt, 'lines processed')

