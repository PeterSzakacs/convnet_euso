import os
import math

import numpy as np

import utils.dataset_utils as ds
import visualization.html_writers as html

# writing the html reports reuires a lot of shared context, so a class with
# methods to write parts of the report file, as opposed to standalone functions
# is very much appropriate for this task.
class report_writer:


    DEFAULT_MAX_TABLE_SIZE=2000


    def __init__(self, item_types, savedir, max_table_size=None,
                 metadata_text_transformer=None):
        self._types = item_types
        self.savedir = savedir
        # html writers
        self._fil = html.file_writer()
        self._tbl = html.table_writer()
        self._img = html.image_writer()
        self._txt = html.text_writer()
        # set maximum table size
        self._tbl_size = max_table_size or report_writer.DEFAULT_MAX_TABLE_SIZE
        self.metadata_text_transformer = metadata_text_transformer


    @property
    def metadata_text_transformer(self):
        return self._meta_trans


    @metadata_text_transformer.setter
    def metadata_text_transformer(self, value=None):
        self._meta_trans = value or (lambda metadata: None)


    @property
    def savedir(self):
        return self._savedir


    @savedir.setter
    def savedir(self, value):
        if not os.path.isdir(value):
            raise Exception('Not a directory: {}'.format(value))
        self._savedir = value


    def _write_file_header(self, logs_dict, dataset):
        hits, misses = logs_dict['hits'], logs_dict['misses']
        num_items = hits + misses
        acc = (hits * 100 / num_items)
        err = 100 - acc

        self._txt.begin_list()
        self._txt.add_heading("Statistics", level=2)
        self._txt.add_list_item("Time run: {}".format(logs_dict['time']))
        self._txt.add_list_item("Dataset name: {}".format(dataset.name))
        self._txt.add_list_item("Network architecture: {}"
                          .format(logs_dict['net_arch']))
        self._txt.add_list_item("Trained model file: {}"
                          .format(logs_dict['model_file']))
        self._txt.add_list_item("Number of items predicted as shower: {}"
                          .format(logs_dict['showers']))
        self._txt.add_list_item("Number of items predicted as noise: {}"
                          .format(logs_dict['noise']))
        self._txt.add_list_item("Total number of items checked: {}"
                          .format(num_items))
        self._txt.add_list_item("Correct predictions: {}".format(hits))
        self._txt.add_list_item("Incorrect predictions: {}".format(misses))
        self._txt.add_list_item("Accuracy: {}%".format(acc))
        self._txt.add_list_item("Error rate: {}%".format(err))
        self._txt.end_list()


    def _write_table_header(self, curr_report, num_reports):
        self._txt.add_heading("Table of frames", level=2)
        prevfile = ("report_{}.html".format(curr_report - 1)
                    if curr_report > 0 else "#")
        nextfile = ("report_{}.html".format(curr_report + 1)
                    if curr_report < num_reports - 1 else "#")
        l = self._txt.get_link(href=prevfile, link_contents="prev",
                               styleclass="l")
        r = self._txt.get_link(href=nextfile, link_contents="next",
                               styleclass="r")
        self._fil.add_div(l + r, styleclass="parspan")


    def _write_table_row(self, log, dataset):
        frame_idx, prediction = log[0], log[1]
        rnd_prediction = np.round(prediction).astype(np.uint8)
        target = dataset.get_targets(frame_idx)
        metadata = dataset.get_metadata(frame_idx)
        shower_prob = round(prediction[0] * 100, 2)
        noise_prob = round(prediction[1] * 100, 2)
        out = ('noise' if np.array_equal(rnd_prediction, [0, 1]) else 'shower')
        targ = ('noise' if np.array_equal(target, [0, 1]) else 'shower')
        imgs = [self._img.get_image('../img/{}/frame-{}.svg'.format(k, frame_idx),
                                    width="184px", height="138px")
                for k in ds.ALL_ITEM_TYPES
                if self._types[k]]
        self._tbl.append_table_row(row_contents=[
            *imgs, "{}%".format(shower_prob), "{}%".format(noise_prob),
            out, targ, self._meta_trans(metadata)]
        )


    def write_report(self, logs_dict, dataset, curr_report=0,
                     save_filename=None):
        log_data = logs_dict['logs']
        num_records = len(log_data)
        first_record = curr_report * self._tbl_size
        last_record = (curr_report + 1) * self._tbl_size
        if last_record > num_records:
            last_record = num_records
        num_reports = math.ceil(num_records / self._tbl_size)
        image_headings = ["{} proj".format(k) for k in ds.ALL_ITEM_TYPES
                          if self._types[k]]
        table_headings = [*image_headings, "Shower %", "Noise %", "Output",
                          "Target", "Metadata"]
        save_filename = save_filename or 'report_{}.html'.format(curr_report)

        self._fil.begin_html_file(
            title="Report for network {}".format(logs_dict['net_arch']),
            css_rules="""
                        table { border: 1px solid black; }
                        .l { float: left; }
                        .r { float: right; }
                        .parspan { width: 100%; min-height: 20px; }
                      """
        )
        self._write_file_header(logs_dict, dataset)
        self._write_table_header(curr_report, num_reports)
        self._tbl.begin_table(table_headings=table_headings)
        for log in log_data[slice(first_record, last_record)]:
            self._write_table_row(log, dataset)
        self._tbl.end_table()
        self._fil.end_html_file(os.path.join(self._savedir, save_filename))


    def write_reports(self, logs_dict, dataset):
        num_records = len(logs_dict['logs'])
        num_reports = math.ceil(num_records / self._tbl_size)
        curr_report = 0

        # for every self._tbl_size records of logs create a separate html report
        for report_idx in range(num_reports):
            self.write_report(logs_dict, dataset, curr_report=report_idx)
