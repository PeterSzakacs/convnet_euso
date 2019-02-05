import io
import os
import math
import operator as op

import numpy as np

import utils.data_utils as dat
import utils.metadata_utils as meta
import visualization.html_writers as html

# writing the html reports reuires a lot of shared context, so a class with
# methods to write parts of the report file, as opposed to standalone functions
# is very much appropriate for this task.
class report_writer:


    DEFAULT_MAX_TABLE_SIZE=2000


    def __init__(self, savedir, table_size=None, extra_fields=None,
                 css_rules=None):
        # properties
        self.savedir = savedir
        self.table_size = table_size
        self.extra_fields = extra_fields
        self.css_rules = css_rules
        # html writers
        self._fil = html.file_writer()
        self._tbl = html.table_writer()
        self._img = html.image_writer()
        self._txt = html.text_writer()


    @property
    def savedir(self):
        return self._savedir


    @savedir.setter
    def savedir(self, value):
        if not os.path.isdir(value):
            raise Exception('Not a directory: {}'.format(value))
        self._savedir = value


    @property
    def extra_fields(self):
        return self._extra_fields


    @extra_fields.setter
    def extra_fields(self, value=None):
        self._extra_fields = value or []


    @property
    def table_size(self):
        return self._tbl_size


    @table_size.setter
    def table_size(self, value=None):
        if value != None:
            self._tbl_size = int(value)
        else:
            self._tbl_size = self.DEFAULT_MAX_TABLE_SIZE


    @property
    def css_rules(self):
        return self._css


    @css_rules.setter
    def css_rules(self, value=None):
        self._css = value or """
                                table { border-collapse: collapse; }
                                th, td {
                                    border: 1px solid black;
                                    padding: 0 6px 0 6px;
                                    text-align: center }
                                .l { float: left; }
                                .r { float: right; }
                                .parspan { width: 100%; min-height: 20px; }
                             """


    def _get_file_header(self, logs, context):
        num_items = len(logs)
        hits = sum(1 if log['output'] == log['target'] else 0 for log in logs)
        misses = num_items - hits
        acc = (hits * 100 / num_items)
        err = 100 - acc

        name = context['dataset'] or 'unknown'
        buffer = io.StringIO()
        buffer.write(self._txt.get_list_start())
        buffer.write(self._txt.get_heading("Statistics", level=2))
        buffer.write(self._txt.get_list_item("Used dataset: {}".format(name)))
        buffer.write(self._txt.get_list_item("Network architecture: {}"
                                             .format(context['net_arch'])))
        buffer.write(self._txt.get_list_item("Trained model file: {}"
                                             .format(context['model_file'])))
        buffer.write(self._txt.get_list_item("Number of items checked: {}"
                                             .format(num_items)))
        buffer.write(self._txt.get_list_item("Correct predictions: {}"
                                             .format(hits)))
        buffer.write(self._txt.get_list_item("Incorrect predictions: {}"
                                             .format(misses)))
        buffer.write(self._txt.get_list_item("Accuracy: {}%".format(acc)))
        buffer.write(self._txt.get_list_item("Error rate: {}%".format(err)))
        buffer.write(self._txt.get_list_end())
        retval = buffer.getvalue()
        buffer.close()
        return retval


    def _write_table_header(self, curr_report, num_reports):
        self._txt.add_heading("Table of dataset items", level=2)
        prevfile = ("report_{}.html".format(curr_report - 1)
                    if curr_report > 0 else "#")
        nextfile = ("report_{}.html".format(curr_report + 1)
                    if curr_report < num_reports - 1 else "#")
        l = self._txt.get_link(href=prevfile, link_contents="prev",
                               styleclass="l")
        r = self._txt.get_link(href=nextfile, link_contents="next",
                               styleclass="r")
        self._fil.add_div(l + r, styleclass="parspan")


    def _write_table_row(self, log, item_types):
        idx = log['item_idx']
        imgs = [self._img.get_image('../img/{}/frame-{}.svg'.format(k, idx),
                                    width="184px", height="138px")
                for k in dat.ALL_ITEM_TYPES
                if item_types[k]]

        shower_prob = round(float(log['shower_prob']) * 100, 2)
        noise_prob = round(float(log['noise_prob']) * 100, 2)
        out, targ = log['output'], log['target']

        self._tbl.append_table_row(row_contents=[
            *imgs, "{}%".format(shower_prob), "{}%".format(noise_prob),
            out, targ, *[log[key] for key in self._extra_fields]]
        )


    def write_reports(self, logs, context):
        num_records = len(logs)
        num_reports = math.ceil(num_records / self._tbl_size)
        item_types = context['item_types']
        image_headings = ["{} proj".format(k) for k in dat.ALL_ITEM_TYPES
                          if item_types[k]]
        table_headings = [*image_headings, "Shower %", "Noise %", "Output",
                          "Target", *self._extra_fields]
        file_header = self._get_file_header(logs, context)

        # for every self._tbl_size records of logs create a separate report
        for report_idx in range(num_reports):
            first_record = report_idx * self._tbl_size
            last_record = (report_idx + 1) * self._tbl_size
            if last_record > num_records:
                last_record = num_records
            save_filename = 'report_{}.html'.format(report_idx)
            self._fil.begin_html_file(
                title="Report for network {}".format(context['net_arch']),
                css_rules=self._css
            )
            self._fil.add_raw_html(file_header)
            self._write_table_header(report_idx, num_reports)
            self._tbl.begin_table(table_headings=table_headings)
            for log in logs[slice(first_record, last_record)]:
                self._write_table_row(log, item_types)
            self._tbl.end_table()
            self._fil.end_html_file(os.path.join(self._savedir, save_filename))

