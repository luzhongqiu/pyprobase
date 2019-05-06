# -*- encoding: utf-8 -*-
import csv
import io
import sys


def extract_csv(num):
    num = int(num)
    for line in sys.stdin:
        data = line.rstrip('\n')
        for row in csv.reader(io.StringIO(data)):
            print(row[num])


sub_command = sys.argv[1]
if sub_command == 'csv':
    extract_csv(sys.argv[2])
