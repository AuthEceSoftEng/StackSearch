import sys
import pickle


def paths_to_dict(paths_file, export_path):
    paths = {}
    with open(paths_file, 'r') as p:
        for line in p:
            bitstr, token, _ = line.strip().split('\t')
            paths[token] = bitstr

    with open(export_path, 'wb') as out:
        pickle.dump(paths, out)


if __name__ == '__main__':
    paths_file = sys.argv[1]

    paths_to_dict(paths_file, 'paths.pkl')