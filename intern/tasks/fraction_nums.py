#!/usr/local/bin/python3
import re
import sys

EVEN_NUMBERS_REGEX = re.compile(r"[-+]?\d*\d\.\d\d*")

for line in ["-123 -16.6 89 90.654", "0.3", "0.9"," 12.56", "67"]:
    all_even_numbers = EVEN_NUMBERS_REGEX.findall(line.strip())
    print(' '.join(all_even_numbers))