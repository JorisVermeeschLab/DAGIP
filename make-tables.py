
def create_table(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    results = []
    for line in lines:
        line = line.rstrip()
        if ':' not in line:
            continue
        i = line.find(':')
        data = eval(line[i+1:])
        results.append(data)
    assert len(results) == 12

    METHODS = ['No correction', 'Centering-scaling', 'GC correction', 'Domain adaptation']

    f = lambda x: f'{(int(round(1000 * x)) * 0.1):.1f}'

    s = ''
    for i in range(4):
        s += METHODS[i]
        for j in [2, 1, 0]:
            k = i * 3 + j
            sensitivity = f(results[k]['sensitivity-best'])
            specificity = f(results[k]['specificity-best'])
            auroc = f(results[k]['auroc'])
            aupr = f(results[k]['aupr'])
            mcc = f(results[k]['mcc-best'])
            s += f' & {sensitivity} \\% & {specificity} \\% & {mcc} \\% & {auroc} \\% & {aupr} \\%'
        s += ' \\\\\n'

    return s

#print(create_table('results-hl.txt'))

#print(create_table('results-dlbcl.txt'))

#print(create_table('results-mm.txt'))

print(create_table('results-ov.txt'))
