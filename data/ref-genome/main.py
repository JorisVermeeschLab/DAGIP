filepath = 'GCA_000001405.15_GRCh38_no_alt_analysis_set.fa'


with open(filepath, 'r') as f:
    i = 0
    while True:
        f.read(1)
        if i % 100000 == 0:
            print(i)
        i += 1
    print(i)
