#!/usr/bin/env python3
"""Exercise the source guard without editing the source header."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(checker, header, flags, label, expected_code, expected_text):
    result = subprocess.run([sys.executable, str(checker), str(header), *flags], text=True, capture_output=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode != expected_code or expected_text not in output:
        raise AssertionError(f'{label}: expected exit {expected_code} and {expected_text!r}; got exit {result.returncode}: {output}')
    print(f'{label}: exit {result.returncode}: {output.strip()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checker', type=Path)
    parser.add_argument('header', type=Path)
    parser.add_argument('--clang', type=Path, required=True)
    parser.add_argument('--include-dirs', default='')
    parser.add_argument('--defines', default='')
    args = parser.parse_args()
    flags = ['--clang', str(args.clang), '--include-dirs', args.include_dirs, '--defines', args.defines]

    with tempfile.TemporaryDirectory(prefix='mtp-hparams-') as temp_dir:
        copy = Path(temp_dir) / 'llama-hparams.h'
        shutil.copyfile(args.header, copy)
        original = copy.read_text()
        marker = '    uint32_t n_ctx_train;'
        if original.count(marker) != 1:
            raise AssertionError('expected one n_ctx_train declaration')

        run(args.checker, copy, flags, 'real header', 0, 'all data members classified')

        copy.write_text(original.replace(marker, '    alignas(1) bool aligned_padding_member;\n' + marker, 1))
        run(args.checker, copy, flags, 'aligned padding member', 1, 'unclassified llama_hparams member: aligned_padding_member')

        copy.write_text(original.replace(marker, '    bool unclassified_padding_byte;\n' + marker, 1))
        run(args.checker, copy, flags, 'plain padding member', 1, 'unclassified llama_hparams member: unclassified_padding_byte')

        copy.write_text(original.replace(marker, '', 1))
        run(args.checker, copy, flags, 'removed classified member', 1, 'classified llama_hparams member no longer exists: n_ctx_train')

        copy.write_text(original.replace(marker,
                                         '    inline static constexpr int some_constant = 1;\n    int some_method() const { return some_constant; }\n' + marker, 1))
        run(args.checker, copy, flags, 'static constant and method', 0, 'all data members classified')

        # Cover syntax that does not occur in every revision of the real header.
        copy.write_text('''
#include <array>
struct llama_hparams {
    int a[2], b = 1; // comment with fake member: int ignored;
    std::array<int, 2> c{}, d{1, 2};
    struct Nested { int inner; } nested;
    struct TypeOnly { int ignored; };
    struct { int anonymous_inner; };
    inline static constexpr int constant = 3;
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
    X(anonymous_inner, EQUAL) \\
    X(e, EQUAL)
''')
        run(args.checker, copy, flags, 'declaration syntax', 0, 'all data members classified')


if __name__ == '__main__':
    main()
