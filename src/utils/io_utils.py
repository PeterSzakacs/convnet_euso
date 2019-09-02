import os
import csv


def load_TSV(filename, selected_columns=None, output_list=None):
    output_list = output_list or []
    if selected_columns is not None:
        process_row = lambda row, cols: {col:row[col] for col in cols}
    else:
        process_row = lambda row, cols: row
    with open(filename, 'r', encoding='UTF-8') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        for row in reader:
            output_list.append(process_row(row, selected_columns))
    return output_list


def save_TSV(filename, rows, column_order, file_exists_overwrite=False):
    if os.path.isfile(filename) and not file_exists_overwrite:
        raise FileExistsError('Cannot overwrite existing file'
                              .format(filename))
    with open(filename, 'w', encoding='UTF-8') as outfile:
        writer = csv.DictWriter(outfile, column_order, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
