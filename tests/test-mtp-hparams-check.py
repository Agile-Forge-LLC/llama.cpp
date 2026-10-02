#!/usr/bin/env python3
"""Exercise the source guard without editing the source header."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(checker, header, expected_code, expected_text):
    result = subprocess.run([sys.executable, str(checker), str(header)], text=True, capture_output=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode != expected_code or expected_text not in output:
        raise AssertionError(f'expected exit {expected_code} and {expected_text!r}; got exit {result.returncode}: {output}')
    print(output.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checker', type=Path)
    parser.add_argument('header', type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix='mtp-hparams-') as temp_dir:
        copy = Path(temp_dir) / 'llama-hparams.h'
        shutil.copyfile(args.header, copy)
        original = copy.read_text()

        run(args.checker, copy, 0, 'all data members classified')

        copy.write_text(original.replace('    uint32_t n_ctx_train;', '    bool unclassified_padding_byte;\n    uint32_t n_ctx_train;', 1))
        run(args.checker, copy, 1, 'unclassified llama_hparams member: unclassified_padding_byte')

        copy.write_text(original.replace('    uint32_t n_ctx_train;', '    // n_ctx_train removed for checker test\n    ', 1))
        run(args.checker, copy, 1, 'classified llama_hparams member no longer exists: n_ctx_train')

        # Cover syntax that does not occur in every revision of the real header.
        copy.write_text('''
struct llama_hparams {
    int a[2], b = 1; // comment with fake member: int ignored;
    std::array<int, 2> c{}, d{1, 2};
    struct Nested { int inner; } nested;
    struct TypeOnly { int ignored; };
    static constexpr int constant = 3;
    int get() const;
    void set() { int local = 0; }
    bool e = false;
};
#define LLAMA_HPARAMS_MTP_FIELDS(X) \\
    X(a, EQUAL) \\
    X(b, ALLOW) \\
    X(c, LAYER) \\
    X(d, EQUAL) \\
    X(nested, NESTED) \\
    X(e, EQUAL)
''')
        run(args.checker, copy, 0, 'all data members classified')


if __name__ == '__main__':
    main()
