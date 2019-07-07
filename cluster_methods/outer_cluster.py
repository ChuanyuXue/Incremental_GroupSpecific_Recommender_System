def load_bipartite(path):
    file = open(path)
    ug = [[] for x in range(10)]
    ig = [[] for x in range(10)]
    depth = False
    nr = -1
    item_or_user = -1
    for line in file:
        if line[:-1] == 'Depth: 1':
            depth = True
            continue

        if line[:-2] == 'Nr of module: ':
            nr = int(line[-2])
            continue

        if line[:-1] == 'Rownames: ':
            item_or_user = 0
            continue

        if line[:-1] == 'Colnames: ':
            item_or_user = 1
            continue

        if depth and item_or_user == 0:
            if line[9:-1] != '' and line[9:-1] != '_________________':
                ug[nr].append(int(line[9:-1]))

        if depth and item_or_user == 1:
            if line[9:-1] != ''and line[9:-1] != '_________________':
                ig[nr].append(int(line[9:-1]))
    file.close()
    return ug, ig


def load_bipartite_louvain(path, data):
    file = open(path)
    ug = []
    ig = []
    for line in file:
        parts = line.split(':')
        if parts[0] == '\n':
            break
        if parts[0].split('[')[1].split(']')[0] == 'V1':
            ug.append(list(map(int, parts[1].replace('\n', '').replace('u', '').replace('i', '').split(','))))
        else:
            ig.append(list(map(int, parts[1].replace('\n', '').replace('u', '').replace('i', '').split(','))))
    file.close()

    uset, iset = set(data[0]), set(data[1])
    for g in ug:
        for u in g:
            if u not in uset:
                g.remove(u)
    for g in ig:
        for i in g:
            if i not in iset:
                g.remove(i)

    return ug, ig
