import os, fnmatch
def find_files(pattern, path):
    roots = []
    names = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                roots.append(root)
                names.append(name)
                # result.append(os.path.join(root, name))
    return [roots, names]

