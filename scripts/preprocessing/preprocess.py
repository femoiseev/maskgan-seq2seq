import click
import os
from sklearn.model_selection import train_test_split


def tokenize(s, space='<SPACE>'):
    return ' '.join(list(s.replace(' ', '\t'))).replace('\t', space)

def write_list_to_file(filename, ar):
    with open(filename, 'w') as f:
        f.writelines("{}\n".format(s) for s in ar)

@click.command()
@click.option('--input_file')
@click.option('--output_dir')
@click.option('--test_size', default=0.2)
@click.option('--val_size', default=0.2)
@click.option('--source_lang', default='en')
@click.option('--target_lang', default='he')
@click.option('--seed', default=42)
@click.option('--space', default='<SPACE>')
def preprocess(input_file, output_dir, test_size, val_size, source_lang, target_lang, seed, space):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(input_file, 'r') as f:
        lines = f.readlines()

    splitted_lines = [line.strip().lower().split('\t') for line in lines]
    source, target = list(zip(*splitted_lines))

    source = [tokenize(s, space) for s in source]
    target = [tokenize(s, space) for s in target]

    source_train_and_val, source_test, target_train_and_val, target_test = train_test_split(source, target, test_size=test_size, random_state=seed)
    source_train, source_val, target_train, target_val = train_test_split(source_train_and_val, target_train_and_val, test_size=val_size, random_state=seed)

    output_names = ['train.{}', 'valid.{}', 'test.{}']

    for name, (src, tgt) in zip(output_names, [(source_train, target_train), (source_val, target_val), (source_test, target_test)]):
        write_list_to_file(output_dir + name.format(source_lang), src)
        write_list_to_file(output_dir + name.format(target_lang), tgt)


if __name__ == '__main__':
    preprocess()
