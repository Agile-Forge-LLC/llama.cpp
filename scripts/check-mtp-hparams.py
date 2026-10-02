#!/usr/bin/env python3
"""Check that the MTP field classification covers every llama_hparams data member."""

import argparse
import re
import sys
from pathlib import Path


TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\d+|::|[^\s]')
COMMENT = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
IDENT = re.compile(r'[A-Za-z_]\w*\Z')
KINDS = {'EQUAL', 'ALLOW', 'LAYER', 'NESTED'}


def strip_comments(text):
    return COMMENT.sub(lambda match: ' ' if match.group().startswith('/') else match.group(), text)


def matching_brace(tokens, start):
    depth = 0
    for i in range(start, len(tokens)):
        if tokens[i] == '{':
            depth += 1
        elif tokens[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError('unterminated llama_hparams struct')


def top_level_split(tokens, separator):
    parts = []
    part = []
    depth = {'<': 0, '(': 0, '[': 0, '{': 0}
    closing = {'>': '<', ')': '(', ']': '[', '}': '{'}
    for token in tokens:
        if token == separator and not any(depth.values()):
            parts.append(part)
            part = []
            continue
        if token in depth:
            depth[token] += 1
        elif token in closing:
            key = closing[token]
            depth[key] -= 1
            if depth[key] < 0:
                raise ValueError(f'unbalanced declaration: {tokens}')
        part.append(token)
    if any(depth.values()):
        raise ValueError(f'unbalanced declaration: {tokens}')
    parts.append(part)
    return parts


def top_level_prefix(tokens):
    depth = {'<': 0, '(': 0, '[': 0, '{': 0}
    closing = {'>': '<', ')': '(', ']': '[', '}': '{'}
    prefix = []
    for i, token in enumerate(tokens):
        if token in ('=', '{', '[') and not any(depth.values()):
            break
        if token == '(' and not any(depth.values()):
            if i + 1 < len(tokens) and tokens[i + 1] in ('*', '&', '&&'):
                raise ValueError(f'unsupported llama_hparams function-pointer declaration: {tokens}')
            return None  # member function
        if token in depth:
            depth[token] += 1
        elif token in closing:
            depth[closing[token]] -= 1
        prefix.append(token)
    return prefix


def members(header):
    text = strip_comments(header)
    match = re.search(r'\bstruct\s+llama_hparams\s*\{', text)
    if not match:
        raise ValueError('llama_hparams struct not found')
    tokens = TOKEN.findall(text[match.end() - 1:])
    body = tokens[1:matching_brace(tokens, 0)]
    fields = []
    statement = []
    braces = 0
    for token in body:
        if token == '{':
            braces += 1
        elif token == '}':
            braces -= 1
        if token == ';' and braces == 0:
            if statement and statement[0] not in ('static', 'using', 'typedef', 'friend'):
                if statement[0] in ('struct', 'class', 'union') and '{' in statement:
                    opening = statement.index('{')
                    statement = statement[matching_brace(statement, opening) + 1:]
                    if not statement:
                        statement = []
                        continue
                for declarator in top_level_split(statement, ','):
                    prefix = top_level_prefix(declarator)
                    if prefix is None:
                        continue
                    if not prefix or not IDENT.fullmatch(prefix[-1]):
                        raise ValueError(f'unsupported llama_hparams declaration: {" ".join(statement)}')
                    # A nested type declaration without an instance is not a data member.
                    if prefix[-1] in ('struct', 'class', 'enum'):
                        continue
                    fields.append(prefix[-1])
            statement = []
        else:
            statement.append(token)
        # An inline function body ends without a semicolon. Do not merge it with the next member.
        if token == '}' and braces == 0 and top_level_prefix(statement) is None:
            statement = []
    if statement:
        raise ValueError('unterminated llama_hparams declaration')
    if len(fields) != len(set(fields)):
        raise ValueError('duplicate llama_hparams data member')
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


def check(header):
    declared = members(header)
    classified = classifications(header)
    errors = []
    for name in sorted(declared - classified):
        errors.append(f'unclassified llama_hparams member: {name}')
    for name in sorted(classified - declared):
        errors.append(f'classified llama_hparams member no longer exists: {name}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('header', type=Path)
    args = parser.parse_args()
    try:
        errors = check(args.header.read_text())
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
