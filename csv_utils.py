import csv

import os


def read(filename, is_headers=True):
    if is_headers:
        with open(filename, 'r') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(10))
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile, dialect=dialect)
            rows = list(reader)
            return rows
    else:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            return rows


def write(filename, fieldnames, rows):
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames)
        if os.stat(filename).st_size == 0:
            writer.writeheader()
        writer.writerows(rows)
