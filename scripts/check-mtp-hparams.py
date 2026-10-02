#!/usr/bin/env python3
"""Check that the MTP field classification covers every llama_hparams data member."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


KINDS = {'EQUAL', 'ALLOW', 'LAYER', 'NESTED'}


def members(header, clang, includes, defines):
    command = [str(clang), '-std=c++17', '-fsyntax-only', '-Xclang', '-ast-dump=json',
               '-Xclang', '-ast-dump-filter=llama_hparams']
    command += [f'-I{path}' for path in (header.parent, *includes)]
    command += [f'-D{definition}' for definition in defines]
    command += ['-x', 'c++', '-']
    try:
        result = subprocess.run(command, input='#include "llama-hparams.h"\n', text=True,
                                capture_output=True, check=False)
    except OSError as exc:
        raise ValueError(f'cannot run clang++: {exc}') from exc
    if result.returncode:
        raise ValueError(f'clang++ failed to parse llama_hparams.h:\n{result.stderr.strip()}')

    decoder = json.JSONDecoder()
    offset = 0
    definitions = []
    try:
        while offset < len(result.stdout):
            while offset < len(result.stdout) and result.stdout[offset].isspace():
                offset += 1
            if offset == len(result.stdout):
                break
            node, offset = decoder.raw_decode(result.stdout, offset)
            if node.get('kind') == 'CXXRecordDecl' and node.get('name') == 'llama_hparams' and node.get('completeDefinition'):
                definitions.append(node)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f'invalid clang++ JSON AST: {exc}') from exc
    if len(definitions) != 1:
        raise ValueError(f'expected one llama_hparams definition in clang++ AST, found {len(definitions)}')

    children = definitions[0].get('inner', [])
    fields = []
    anonymous = False
    for child in children:
        if child.get('kind') == 'FieldDecl':
            if child.get('name'):
                fields.append(child['name'])
            else:
                anonymous = True
        elif child.get('kind') == 'IndirectFieldDecl':
            # Clang exposes the members of anonymous structs/unions through these nodes.
            if not child.get('name'):
                raise ValueError('unnamed indirect llama_hparams member in clang++ AST')
            fields.append(child['name'])
    if anonymous and not any(child.get('kind') == 'IndirectFieldDecl' for child in children):
        raise ValueError('unnamed llama_hparams member without indirect fields in clang++ AST')
    if not fields or len(fields) != len(set(fields)):
        raise ValueError('missing or duplicate llama_hparams data members in clang++ AST')
    return set(fields)


def classifications(header):
    match = re.search(r'^#define LLAMA_HPARAMS_MTP_FIELDS\(X\)\s*\\\s*\n', header, re.M)
    if not match:
        raise ValueError('LLAMA_HPARAMS_MTP_FIELDS list not found')
    fields = []
    for line in header[match.end():].splitlines():
        entry = re.fullmatch(r'\s*X\((\w+),\s*(\w+)\)\s*(\\)?\s*', line)
        if not entry:
            raise ValueError(f'invalid MTP field classification: {line}')
        name, kind, continuation = entry.groups()
        if kind not in KINDS:
            raise ValueError(f'unknown MTP field classification: {name}: {kind}')
        fields.append(name)
        if not continuation:
            break
    if len(fields) != len(set(fields)):
        raise ValueError('duplicate MTP field classification')
    return set(fields)


def check(header, clang, includes, defines):
    declared = members(header, clang, includes, defines)
    classified = classifications(header.read_text())
    errors = []
    for name in sorted(declared - classified):
        errors.append(f'unclassified llama_hparams member: {name}')
    for name in sorted(classified - declared):
        errors.append(f'classified llama_hparams member no longer exists: {name}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('header', type=Path)
    parser.add_argument('--clang', type=Path, required=True)
    parser.add_argument('--include-dirs', default='', help='CMake target include directories, separated by |')
    parser.add_argument('--defines', default='', help='CMake target compile definitions, separated by |')
    args = parser.parse_args()
    try:
        errors = check(args.header.resolve(), args.clang,
                       [Path(path) for path in args.include_dirs.split('|') if path],
                       [definition for definition in args.defines.split('|') if definition])
    except (OSError, ValueError) as exc:
        errors = [str(exc)]
    if errors:
        for error in errors:
            print(f'MTP hparams check: {error}', file=sys.stderr)
        return 1
    print('MTP hparams check: all data members classified')
    return 0


if __name__ == '__main__':
    sys.exit(main())
