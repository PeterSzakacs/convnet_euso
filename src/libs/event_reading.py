#!/usr/bin/python3

import datetime
from enum import Enum
import numpy as np
import ROOT

from libs.eusotrees.exptree import ExpTree


class GtuPdmData(object):
    photon_count_data = None
    gtu = -1
    gtu_time = -1
    # gtu_time1 = -1
    gtu_datetime = datetime.datetime(1910, 1, 1)

    # gtu_global = -1         #  "gtuGlobal/I"
    trg_box_per_gtu = -1    #  "trgBoxPerGTU/I"
    trg_pmt_per_gtu = -1    #  "trgPMTPerGTU/I"
    trg_ec_per_gtu = -1     #  "trgECPerGTU/I"
    n_persist = -1          #  "&nPersist/I"
    gtu_in_persist = -1     #  "&gtuInPersist/I"
    sum_l1_pdm = -1         #  "sumL1PDM/I"
    sum_l1_ec = None                #  "sumL1EC[9]/I"
    sum_l1_pmt = None   #  "sumL1PMT[18][2]/I"

    gps_datetime = datetime.datetime(1910, 1, 1)
    gps_date__raw = -1
    gps_time__raw = -1
    gps_lat = -999.
    gps_lon = -999.
    gps_alt = -999.
    gps_speed = -999.
    gps_course = -999.

    l1trg_events = None

    def __init__(self, photon_count_data, gtu, gtu_time, #gtu_time1,
                 trg_box_per_gtu, trg_pmt_per_gtu, trg_ec_per_gtu,
                 n_persist, gtu_in_persist, sum_l1_pdm, sum_l1_ec, sum_l1_pmt,
                 l1trg_events=[], use_raw_photon_count_data=False,
                 gps_date=-1, gps_time=-1, gps_lat=-999, gps_lon=-999, gps_alt=-999,
                 gps_speed=-999, gps_course=-999):

        if not use_raw_photon_count_data:
            self.photon_count_data = np.zeros_like(photon_count_data, dtype=photon_count_data.dtype)

            for ccb_index in range(0,len(photon_count_data)):
                for pdm_index in range(0,len(photon_count_data[ccb_index])):
                    self.photon_count_data[ccb_index, pdm_index] = np.transpose(np.fliplr(photon_count_data[ccb_index, pdm_index])) # np.fliplr(np.transpose(photon_count_data[ccb_index, pdm_index]))
        else:
            self.photon_count_data = np.array(photon_count_data)

        self.gtu = np.asscalar(gtu) if isinstance(gtu, np.ndarray) else gtu
        self.gtu_time = np.asscalar(gtu_time) if isinstance(gtu_time, np.ndarray) else gtu_time
        # self.gtu_time1 = np.asscalar(gtu_time1) if isinstance(gtu_time1, np.ndarray) else gtu_time1

        if self.gtu_time is not None:
            self.gtu_datetime = datetime.datetime.utcfromtimestamp(self.gtu_time)

        self.trg_box_per_gtu = np.asscalar(trg_box_per_gtu) if isinstance(trg_box_per_gtu, np.ndarray) else trg_box_per_gtu 
        self.trg_pmt_per_gtu = np.asscalar(trg_pmt_per_gtu) if isinstance(trg_pmt_per_gtu, np.ndarray) else trg_pmt_per_gtu
        self.trg_ec_per_gtu = np.asscalar(trg_ec_per_gtu) if isinstance(trg_ec_per_gtu, np.ndarray) else trg_ec_per_gtu
        self.n_persist = np.asscalar(n_persist) if isinstance(n_persist, np.ndarray) else n_persist
        self.gtu_in_persist = np.asscalar(gtu_in_persist) if isinstance(gtu_in_persist, np.ndarray) else gtu_in_persist
        self.sum_l1_pdm = np.asscalar(sum_l1_pdm) if isinstance(sum_l1_pdm, np.ndarray) else sum_l1_pdm
        self.sum_l1_ec = sum_l1_ec
        self.sum_l1_pmt = sum_l1_pmt

        self.gps_date__raw = np.asscalar(gps_date) if isinstance(gps_date, np.ndarray) else gps_date
        self.gps_time__raw = np.asscalar(gps_time) if isinstance(gps_time, np.ndarray) else gps_time
        self.gps_lat = np.asscalar(gps_lat) if isinstance(gps_lat, np.ndarray) else gps_lat
        self.gps_lon = np.asscalar(gps_lon) if isinstance(gps_lon, np.ndarray) else gps_lon
        self.gps_alt = np.asscalar(gps_alt) if isinstance(gps_alt, np.ndarray) else gps_alt
        self.gps_speed = np.asscalar(gps_speed) if isinstance(gps_speed, np.ndarray) else gps_speed
        self.gps_course = np.asscalar(gps_course) if isinstance(gps_course, np.ndarray) else gps_course

        # based on ETOS' eusotrees/datatree.py
        if gps_date is not None and gps_time is not None and gps_date > 0.1 and gps_time > 0.1:
            self.gps_datetime = datetime.datetime(int(gps_date) % 100 + 2000, int((gps_date % 10000) / 100), int(gps_date / 10000),
                                      int(gps_time / 10000), int((gps_time % 10000) / 100), int(gps_time) % 100)
        else:
            self.gps_datetime = datetime.datetime(1910, 1, 1)

        self.l1trg_events = l1trg_events


class L1TrgEvent(object):
    gtu_pdm_data = None
    ec_id = -1
    pmt_row = -1  # should be converted to Lech
    pmt_col = -1  # sholud be converted to Lech
    pix_row = -1
    pix_col = -1
    sum_l1 = -1
    thr_l1 = -1
    persist_l1 = -1

    packet_id = -1      # ideally this should be in GtuPdmData, but this is from l1trg tree
    gtu_in_packet = -1

    def __init__(self, gtu_pdm_data, ec_id, pmt_row, pmt_col, pix_row, pix_col, sum_l1, thr_l1, persist_l1,
                 packet_id = -1, gtu_in_packet = -1):
        self.gtu_pdm_data = gtu_pdm_data
        self.ec_id = np.asscalar(ec_id) if isinstance(ec_id, np.ndarray) else ec_id
        self.pmt_row = np.asscalar(pmt_row) if isinstance(pmt_row, np.ndarray) else pmt_row
        self.pmt_col = np.asscalar(pmt_col) if isinstance(pmt_col, np.ndarray) else pmt_col
        self.pix_row = np.asscalar(pix_row) if isinstance(pix_row, np.ndarray) else pix_row
        self.pix_col = np.asscalar(pix_col) if isinstance(pix_col, np.ndarray) else pix_col
        self.sum_l1 = np.asscalar(sum_l1) if isinstance(sum_l1, np.ndarray) else sum_l1
        self.thr_l1 = np.asscalar(thr_l1) if isinstance(thr_l1, np.ndarray) else thr_l1
        self.persist_l1 = np.asscalar(persist_l1) if isinstance(persist_l1, np.ndarray) else persist_l1

        self.packet_id = np.asscalar(packet_id) if isinstance(packet_id, np.ndarray) else packet_id
        self.gtu_in_packet = np.asscalar(gtu_in_packet) if isinstance(gtu_in_packet, np.ndarray) else gtu_in_packet

    @classmethod
    def from_mario_format(cls, gtu_pdm_data, ec_id, pmt_row, pmt_col, pix_row, pix_col, sum_l1, thr_l1, persist_l1,
                          packet_id = -1, gtu_in_packet = -1):
        e = L1TrgEvent(gtu_pdm_data, ec_id, pmt_row, pmt_col, pix_row, pix_col, sum_l1, thr_l1, persist_l1, packet_id, gtu_in_packet)
        e.o_pmt_col = e.pmt_col; e.o_pmt_row = e.pmt_row; e.o_pix_col = e.pix_col; e.o_pix_row = e.pix_row
        e.pmt_col, e.pmt_row, e.pix_col, e.pix_row = cls.mario2pdm_for_mpl(e.pmt_col, e.pmt_row, e.pix_col, e.pix_row)
        return e

    @classmethod
    def mario2pdm_for_mpl(cls, mpmt_col=0, mpmt_row=0, pix_x=0, pix_y=0):
        if mpmt_row >= 18 or mpmt_row < 0:
            raise Exception("Something is rotten in the state of row#. mario")

        if mpmt_col >= 2 or mpmt_col < 0:
            raise Exception("Something is rotten in the state of col#. mario")

        ec_index = mpmt_row // 2
        pmt_row_in_ec = mpmt_row % 2
        lech_pmt_row = (ec_index // 3) * 2 + pmt_row_in_ec
        lech_pmt_col = (ec_index % 3) * 2 + mpmt_col

        abspix_col = lech_pmt_col*8 + pix_x
        abspix_row = lech_pmt_row*8 + pix_y

        return lech_pmt_col, lech_pmt_row, abspix_col, abspix_row


# there should be an additional base event reader class


class L1EventReader(object):
    t_l1trg = None
    t_gtusry = None
    t_thrtable = None

    kenji_l1_file = None

    _current_l1trg_entry = -1
    _current_gtusry_entry = -1

    kenji_l1trg_entries = -1
    t_gtusry_entries = -1
    t_thrtable_entries = -1

    _l1trg_ecID = None # np.array([-1], dtype=np.int32)
    _l1trg_pmtRow = None # np.array([-1], dtype=np.int32)
    _l1trg_pmtCol = None # np.array([-1], dtype=np.int32)
    _l1trg_pixRow = None # np.array([-1], dtype=np.int32)
    _l1trg_pixCol = None # np.array([-1], dtype=np.int32)
    _l1trg_gtuGlobal = None # np.array([-1], dtype=np.int32)
    _l1trg_packetID = None # np.array([-1], dtype=np.int32)
    _l1trg_gtuInPacket = None # np.array([-1], dtype=np.int32)
    _l1trg_sumL1 = None # np.array([-1], dtype=np.int32)
    _l1trg_thrL1 = None # np.array([-1], dtype=np.int32)
    _l1trg_persistL1 = None # np.array([-1], dtype=np.int32)

    _gtusry_gtuGlobal = None # np.array([-1], dtype=np.int32) #  "gtuGlobal/I"
    _gtusry_trgBoxPerGTU = None # np.array([-1], dtype=np.int32) #  "trgBoxPerGTU/I"
    _gtusry_trgPMTPerGTU = None # np.array([-1], dtype=np.int32) #  "trgPMTPerGTU/I"
    _gtusry_trgECPerGTU = None # np.array([-1], dtype=np.int32) #  "trgECPerGTU/I"
    _gtusry_nPersist = None # np.array([-1], dtype=np.int32) #  "&nPersist/I"
    _gtusry_gtuInPersist = None # np.array([-1], dtype=np.int32) #  "&gtuInPersist/I"

    _gtusry_sumL1PDM = None # np.array([-1], dtype=np.int32) #  "sumL1PDM/I"
    _gtusry_sumL1EC = None # np.array([-1]*9, dtype=np.int32) #  "sumL1EC[9]/I"
    _gtusry_sumL1PMT = None # np.negative(np.ones((18,2), dtype=np.int32)) #  "sumL1PMT[18][2]/I"
    _gtusry_trgPMT = None # np.negative(np.ones((18,2), dtype=np.int32)) #  "sumL1PMT[18][2]/I"

    @classmethod
    def _get_branch_or_raise(cls, file, tree, name):
        br = tree.GetBranch(name)
        if br is None:
            raise Exception("{} > {} is missing branch \"{}\"".format(file, tree.GetName(), name))
        return br

    @classmethod
    def _get_leaf_or_raise(cls, file, tree, branch_name, leaf_name):
        br = cls._get_branch_or_raise(file, tree, branch_name)
        leaf = br.GetLeaf(leaf_name)
        return leaf

    @classmethod
    def open_kenji_l1(cls, pathname):
        f = ROOT.TFile.Open(pathname, "read")
        if f:
            t_l1trg = f.Get("l1trg")                # each PDM is analyzed
            t_gtusry = f.Get("gtusry")              # each GTU is saved (records / 2304)
            t_thrtable = f.Get("thrtable")          # 1 entry - single table
            return f, t_l1trg, t_gtusry, t_thrtable
        else:
            raise Exception("Cannot open kenji's l1 triggers file {}".format(pathname))

    def __init__(self, kenji_l1_pathname):

        if kenji_l1_pathname:
            self.kenji_l1_file, self.t_l1trg, self.t_gtusry, self.t_thrtable = self.open_kenji_l1(kenji_l1_pathname)

        if self.kenji_l1_file:
            self._l1trg_ecID = np.array([-1], dtype=np.int32)
            self._l1trg_pmtRow = np.array([-1], dtype=np.int32)
            self._l1trg_pmtCol = np.array([-1], dtype=np.int32)
            self._l1trg_pixRow = np.array([-1], dtype=np.int32)
            self._l1trg_pixCol = np.array([-1], dtype=np.int32)
            self._l1trg_gtuGlobal = np.array([-1], dtype=np.int32)
            self._l1trg_packetID = np.array([-1], dtype=np.int32)
            self._l1trg_gtuInPacket = np.array([-1], dtype=np.int32)
            self._l1trg_sumL1 = np.array([-1], dtype=np.int32)
            self._l1trg_thrL1 = np.array([-1], dtype=np.int32)
            self._l1trg_persistL1 = np.array([-1], dtype=np.int32)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "ecID").SetAddress(self._l1trg_ecID)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "pmtRow").SetAddress(self._l1trg_pmtRow)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "pmtCol").SetAddress(self._l1trg_pmtCol)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "pixRow").SetAddress(self._l1trg_pixRow)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "pixCol").SetAddress(self._l1trg_pixCol)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "gtuGlobal").SetAddress(self._l1trg_gtuGlobal)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "packetID").SetAddress(self._l1trg_packetID)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "gtuInPacket").SetAddress(
                self._l1trg_gtuInPacket)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "sumL1").SetAddress(self._l1trg_sumL1)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "thrL1").SetAddress(self._l1trg_thrL1)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "persistL1").SetAddress(self._l1trg_persistL1)

            # l1trg->Branch("ecID", & ecID, "ecID/I");
            # l1trg->Branch("pmtRow", & pmtRow, "pmtRow/I");
            # l1trg->Branch("pmtCol", & pmtCol, "pmtCol/I");
            # l1trg->Branch("pixRow", & pixRow, "pixRow/I");
            # l1trg->Branch("pixCol", & pixCol, "pixCol/I");
            # l1trg->Branch("gtuGlobal", & gtuGlobal, "gtuGlobal/I");
            # l1trg->Branch("packetID", & packetID, "packetID/I");
            # l1trg->Branch("gtuInPacket", & gtuInPacket, "gtuInPacket/I");
            # l1trg->Branch("sumL1", & sumL1, "sumL1/I");
            # l1trg->Branch("thrL1", & thrL1, "thrL1/I");
            # l1trg->Branch("persistL1", & persistL1, "persistL1/I");

            # !!!
            #
            # thrtable->Branch("triggerThresholds", triggerThresholds, "triggerThresholds[100][5]/F");
            #
            # !!!

            self._gtusry_gtuGlobal = np.array([-1], dtype=np.int32)  # "gtuGlobal/I"
            self._gtusry_trgBoxPerGTU = np.array([-1], dtype=np.int32)  # "trgBoxPerGTU/I"
            self._gtusry_trgPMTPerGTU = np.array([-1], dtype=np.int32)  # "trgPMTPerGTU/I"
            self._gtusry_trgECPerGTU = np.array([-1], dtype=np.int32)  # "trgECPerGTU/I"
            self._gtusry_nPersist = np.array([-1], dtype=np.int32)  # "&nPersist/I"
            self._gtusry_gtuInPersist = np.array([-1], dtype=np.int32)  # "&gtuInPersist/I"

            self._gtusry_sumL1PDM = np.array([-1], dtype=np.int32)  # "sumL1PDM/I"
            self._gtusry_sumL1EC = np.array([-1] * 9, dtype=np.int32)  # "sumL1EC[9]/I"
            self._gtusry_sumL1PMT = np.negative(np.ones((18, 2), dtype=np.int32))  # "sumL1PMT[18][2]/I"
            self._gtusry_trgPMT = np.negative(np.ones((18, 2), dtype=np.int32))  # "sumL1PMT[18][2]/I"

            self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "gtuGlobal").SetAddress(self._gtusry_gtuGlobal)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "trgBoxPerGTU").SetAddress(
                self._gtusry_trgBoxPerGTU)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "trgPMTPerGTU").SetAddress(
                self._gtusry_trgPMTPerGTU)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "trgECPerGTU").SetAddress(
                self._gtusry_trgECPerGTU)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "nPersist").SetAddress(self._gtusry_nPersist)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "gtuInPersist").SetAddress(
                self._gtusry_gtuInPersist)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "sumL1PDM").SetAddress(self._gtusry_sumL1PDM)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "sumL1EC").SetAddress(self._gtusry_sumL1EC)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "sumL1PMT").SetAddress(self._gtusry_sumL1PMT)
            self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "trgPMT").SetAddress(self._gtusry_trgPMT)

            # gtusry->Branch("gtuGlobal", & gtuGlobal, "gtuGlobal/I");
            # gtusry->Branch("trgBoxPerGTU", & trgBoxPerGTU, "trgBoxPerGTU/I");
            # gtusry->Branch("trgPMTPerGTU", & trgPMTPerGTU, "trgPMTPerGTU/I");
            # gtusry->Branch("trgECPerGTU", & trgECPerGTU, "trgECPerGTU/I");
            # gtusry->Branch("nPersist", & nPersist, "&nPersist/I");
            # gtusry->Branch("gtuInPersist", & gtuInPersist, "&gtuInPersist/I");
            #
            # gtusry->Branch("sumL1PDM", & sumL1PDM, "sumL1PDM/I");
            # gtusry->Branch("sumL1EC", sumL1EC, "sumL1EC[9]/I");
            # gtusry->Branch("sumL1PMT", sumL1PMT, "sumL1PMT[18][2]/I");
            #
            # gtusry->Branch("trgPMT", trgPMT, "trgPMT[18][2]/I");

            self.kenji_l1trg_entries = self.t_l1trg.GetEntries()  # 23331
            self.t_gtusry_entries = self.t_gtusry.GetEntries()  # 16512
            self.t_thrtable_entries = self.t_thrtable.GetEntries()  # 1

    def __del__(self):
        self.close_files()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_files()

    def close_files(self):
        if self.kenji_l1_file:
            self.kenji_l1_file.Close()

    def _search_for_gtusry_by_gtu(self, gtu):
        if not self.kenji_l1_file:
            return None
        while (self._current_gtusry_entry < 0 or self._gtusry_gtuGlobal != gtu) and \
                        self._current_gtusry_entry < self.t_gtusry_entries:
            self._current_gtusry_entry += 1
            self.t_gtusry.GetEntry(self._current_gtusry_entry)

        return self._current_gtusry_entry

    def _search_for_l1trg_events_by_gtu(self, gtu, gtu_pdm_data=None, presume_sorted=True):
        if not self.kenji_l1_file:
            return None
        if not presume_sorted: # or self._l1trg_gtuGlobal > gtu:
            self._current_l1trg_entry = -1

        events_list = []

        while self._current_l1trg_entry < self.kenji_l1trg_entries:
            if self._current_l1trg_entry == -1 or (presume_sorted and self._l1trg_gtuGlobal < gtu) or (not presume_sorted and self._l1trg_gtuGlobal != gtu) :
                self._current_l1trg_entry += 1
                self.t_l1trg.GetEntry(self._current_l1trg_entry)

            # if self._current_l1trg_entry == 168:
            #     print(">"*30, "Entry 168 read", "<"*30)

            if self._l1trg_gtuGlobal == gtu:
                events_list.append(L1TrgEvent.from_mario_format(gtu_pdm_data, self._l1trg_ecID,
                      self._l1trg_pmtRow, self._l1trg_pmtCol,
                      self._l1trg_pixRow, self._l1trg_pixCol, self._l1trg_sumL1, self._l1trg_thrL1, self. _l1trg_persistL1,
                      self._l1trg_packetID, self._l1trg_gtuInPacket))

                # _l1trg_packetID = None  # np.array([-1], dtype=np.int32)
                # _l1trg_gtuInPacket = None  # np.array([-1], dtype=np.int32)

                self._current_l1trg_entry += 1
                self.t_l1trg.GetEntry(self._current_l1trg_entry)
            elif presume_sorted and self._l1trg_gtuGlobal > gtu:
                break

        return events_list


class AcqL1EventReader(L1EventReader):
    acquisition_file = None

    t_texp = None
    t_tevent = None

    exp_tree = None

    texp_entries = -1
    tevent_entries = -1

    _current_tevent_entry = -1
    _tevent_photon_count_data = None
    _tevent_gtu = None # np.array([-1], dtype=np.int32)
    _tevent_gtu_time = None # np.array([-1], dtype=np.double)
    #_tevent_gtu_time1 = None # np.array([-1], dtype=np.double)
    # ...
    _tevent_gps_date = -1
    _tevent_gps_time = -1
    _tevent_gps_lat = -999.
    _tevent_gps_lon = -999.
    _tevent_gps_alt = -999.
    _tevent_gps_speed = -999.
    _tevent_gps_course = -999.

    last_gtu_pdm_data = None

    entry_is_gtu_optimization = False
    first_gtu = None
    last_gtu = None

    def __init__(self, acquisition_pathname, kenji_l1_pathname, first_gtu=None, last_gtu=None,
                 entry_is_gtu_optimization=False, load_texp=True):
        super(AcqL1EventReader, self).__init__(kenji_l1_pathname)

        self.acquisition_file, self.t_texp, self.t_tevent = self.open_acquisition(acquisition_pathname, load_texp)

        self.exp_tree = ExpTree(self.t_texp, self.acquisition_file)

        self._tevent_photon_count_data = np.zeros((self.exp_tree.ccbCount, self.exp_tree.pdmCount,
                                                   self.exp_tree.pmtCountX * self.exp_tree.pixelCountX,
                                                   self.exp_tree.pmtCountY * self.exp_tree.pixelCountY), dtype=np.ubyte)

        self._tevent_gtu = np.array([-1], dtype=np.int32)
        self._tevent_gtu_time = np.array([-1], dtype=np.double)

        #self._tevent_gtu_time1 = np.array([-1], dtype=np.double)

        self._tevent_gps_date = np.array([-1], dtype=np.float32)
        self._tevent_gps_time = np.array([-1], dtype=np.float32)
        self._tevent_gps_lat = np.array([-1], dtype=np.float32)
        self._tevent_gps_lon = np.array([-1], dtype=np.float32)
        self._tevent_gps_alt = np.array([-1], dtype=np.float32)
        self._tevent_gps_speed = np.array([-1], dtype=np.float32)
        self._tevent_gps_course = np.array([-1], dtype=np.float32)

        self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "photon_count_data").SetAddress(self._tevent_photon_count_data)
        self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "gtu").SetAddress(self._tevent_gtu)
        self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "gtu_time").SetAddress(self._tevent_gtu_time)
        # self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "gtu_time1").SetAddress(self._tevent_gtu_time1)
        self._get_leaf_or_raise(acquisition_pathname, self.t_tevent, "clkb_event_gps", "gps_date").SetAddress(self._tevent_gps_date)
        self._get_leaf_or_raise(acquisition_pathname, self.t_tevent, "clkb_event_gps", "gps_time").SetAddress(self._tevent_gps_time)
        self._get_leaf_or_raise(acquisition_pathname, self.t_tevent, "clkb_event_gps", "gps_lat").SetAddress(self._tevent_gps_lat)
        self._get_leaf_or_raise(acquisition_pathname, self.t_tevent, "clkb_event_gps", "gps_lon").SetAddress(self._tevent_gps_lon)
        self._get_leaf_or_raise(acquisition_pathname, self.t_tevent, "clkb_event_gps", "gps_alt").SetAddress(self._tevent_gps_alt)
        self._get_leaf_or_raise(acquisition_pathname, self.t_tevent, "clkb_event_gps", "gps_speed").SetAddress(self._tevent_gps_speed)
        self._get_leaf_or_raise(acquisition_pathname, self.t_tevent, "clkb_event_gps", "gps_course").SetAddress(self._tevent_gps_course)

        if self.t_texp is not None:
            self.texp_entries = self.t_texp.GetEntries()             # 1

        self.tevent_entries = self.t_tevent.GetEntries()         # 16512

        self.first_gtu = first_gtu
        self.last_gtu = last_gtu
        self.entry_is_gtu_optimization = entry_is_gtu_optimization

    def __del__(self):
        self.close_files()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_files()

    @classmethod
    def open_acquisition(cls, pathname, load_texp=True):
        f = ROOT.TFile.Open(pathname, "read")
        if f:
            if load_texp:
                t_texp = f.Get("texp")
            else:
                t_texp = None
            t_tevent = f.Get("tevent")
            return f, t_texp, t_tevent
        else:
            raise Exception("Cannot open acquisition file {}".format(pathname))

        #self.tree.SetBranchAddress("photon_count_data", self.pcd)
        #
        # for e in t :
        #     print(e.__dir__())

    def close_files(self):
        if self.acquisition_file:
            self.acquisition_file.Close()
        super(AcqL1EventReader, self).close_files()

    # def get_tevent_entry(self, num = None):
    #     if num is None:
    #         self.t_tevent.GetEntry(self._current_tevent_entry)
    #     else:
    #         self.t_tevent.GetEntry(num)
    #         self._current_l1trg_entry = num # practically unnecesarry
    #     self._tevent_photon_count_data = np.filplr(np.transpose(self._tevent_photon_count_data))

    def _search_for_tevent_by_gtu(self, gtu):
        while (self._current_tevent_entry < 0 or self._tevent_gtu != gtu) and \
                        self._current_tevent_entry < self.tevent_entries:
            self._current_tevent_entry += 1
            self.t_tevent.GetEntry(self._current_tevent_entry)

    class L1TrgEventIterator(object):
        ack_ev_reader = None

        def __init__(self, ack_ev_reader):
            if not ack_ev_reader.kenji_l1_file:
                raise Exception('kenji_l1_file file is required')
            self.ack_ev_reader = ack_ev_reader

        def __iter__(self):
            self.ack_ev_reader._current_l1trg_entry = -1
            self.ack_ev_reader._current_tevent_entry = -1
            self.ack_ev_reader._current_gtusry_entry = -1
            return self

        def __next__(self):
            aer = self.ack_ev_reader

            aer._current_l1trg_entry += 1

            if aer._current_l1trg_entry >= aer.kenji_l1trg_entries:
                raise StopIteration

            aer.t_l1trg.GetEntry(aer._current_l1trg_entry)

            if aer.last_gtu_pdm_data is None or aer.last_gtu_pdm_data.gtu != aer._l1trg_gtuGlobal:

                aer._search_for_tevent_by_gtu(aer._l1trg_gtuGlobal)

                if aer._tevent_gtu != aer._l1trg_gtuGlobal:
                    aer._current_tevent_entry = -1
                    aer._search_for_tevent_by_gtu(aer._l1trg_gtuGlobal)
                    if aer._tevent_gtu != aer._l1trg_gtuGlobal:
                        raise Exception("GTU {} from trigger data file (tree l1trg) was not found in acquisition file (tree tevent)".format(aer._l1trg_gtuGlobal))

                aer._search_for_gtusry_by_gtu(aer._l1trg_gtuGlobal)

                if aer._gtusry_gtuGlobal != aer._l1trg_gtuGlobal:
                    aer._current_gtusry_entry = -1
                    aer._search_for_gtusry_by_gtu(aer._l1trg_gtuGlobal)
                    if aer._gtusry_gtuGlobal != aer._l1trg_gtuGlobal:
                        raise Exception("GTU {} from trigger data file (tree l1trg) was not found in trigger data file (tree gtusry)".format(aer._l1trg_gtuGlobal))

                aer.last_gtu_pdm_data = GtuPdmData(aer._tevent_photon_count_data, aer._tevent_gtu, aer._tevent_gtu_time, #aer._tevent_gtu_time1,
                                                   aer._gtusry_trgBoxPerGTU, aer._gtusry_trgPMTPerGTU, aer._gtusry_trgECPerGTU,
                                                   aer._gtusry_nPersist, aer._gtusry_gtuInPersist,
                                                   aer._gtusry_sumL1PDM, aer._gtusry_sumL1EC, aer._gtusry_sumL1PMT,
                                                   [], False,
                                                   aer._tevent_gps_date, aer._tevent_gps_time,
                                                   aer._tevent_gps_lat, aer._tevent_gps_lon, aer._tevent_gps_alt,
                                                   aer._tevent_gps_speed, aer._tevent_gps_course)

            l1trg_ev = L1TrgEvent.from_mario_format(aer.last_gtu_pdm_data, aer._l1trg_ecID,
                          aer._l1trg_pmtRow, aer._l1trg_pmtCol,
                          aer._l1trg_pixRow, aer._l1trg_pixCol, aer._l1trg_sumL1, aer._l1trg_thrL1, aer. _l1trg_persistL1)
            aer.last_gtu_pdm_data.l1trg_events.append(l1trg_ev) # not very correct in this form - not all events are going to be associated to the GTU

            return l1trg_ev

        def next(self):
            return self.__next__()

    class GtuPdmDataIterator(object):
        ack_ev_reader = None
        presume_sorted = True

        def __init__(self, ack_ev_reader, presume_sorted = True):
            self.ack_ev_reader = ack_ev_reader
            self.presume_sorted = presume_sorted

        def __iter__(self):
            aer = self.ack_ev_reader

            aer._current_tevent_entry = aer.first_gtu-1 if aer.first_gtu is not None and aer.entry_is_gtu_optimization else -1

            aer._current_l1trg_entry = -1
            aer._current_gtusry_entry = -1

            return self

        def __next__(self):
            aer = self.ack_ev_reader

            while True:
                aer._current_tevent_entry += 1

                if aer._current_tevent_entry >= aer.tevent_entries:
                    raise StopIteration

                aer.t_tevent.GetEntry(aer._current_tevent_entry)

                tevent_gtu_scalar = np.asscalar(aer._tevent_gtu)

                if  aer.last_gtu is not None and tevent_gtu_scalar > aer.last_gtu:
                    raise StopIteration

                if aer.entry_is_gtu_optimization and tevent_gtu_scalar != aer._current_tevent_entry:
                    raise Exception('entry_is_gtu optimization requires event gtu to be equal to tree entry number, event_gtu = {}, entry = {}'.format(aer._tevent_gtu, aer._current_tevent_entry))

                if aer.first_gtu is None or tevent_gtu_scalar >= aer.first_gtu:
                    break

            if aer.kenji_l1_file:
                aer._search_for_gtusry_by_gtu(aer._tevent_gtu)

                if aer._gtusry_gtuGlobal != aer._tevent_gtu:
                    aer._current_gtusry_entry = -1
                    aer._search_for_gtusry_by_gtu(aer._tevent_gtu)
                    if aer._gtusry_gtuGlobal != aer._tevent_gtu:
                        raise Exception(
                            "GTU {} from acquisition data file (tree tevent) was not found in trigger data file (tree gtusry)".format(aer._tevent_gtu))

            gtu_pdm_data = GtuPdmData(aer._tevent_photon_count_data, aer._tevent_gtu, aer._tevent_gtu_time, #aer._tevent_gtu_time1,
                                        aer._gtusry_trgBoxPerGTU, aer._gtusry_trgPMTPerGTU, aer._gtusry_trgECPerGTU,
                                        aer._gtusry_nPersist, aer._gtusry_gtuInPersist,
                                        aer._gtusry_sumL1PDM, aer._gtusry_sumL1EC, aer._gtusry_sumL1PMT,
                                        None, False,
                                        aer._tevent_gps_date, aer._tevent_gps_time,
                                        aer._tevent_gps_lat, aer._tevent_gps_lon, aer._tevent_gps_alt,
                                        aer._tevent_gps_speed, aer._tevent_gps_course
                                      )

            if aer.kenji_l1_file:
                l1trg_events = aer._search_for_l1trg_events_by_gtu(aer._tevent_gtu, gtu_pdm_data)

                gtu_pdm_data.l1trg_events = l1trg_events

            return gtu_pdm_data

        def next(self):
            return self.__next__()

    def iter_l1trg_events(self):
        return self.L1TrgEventIterator(self)

    def iter_gtu_pdm_data(self):
        return self.GtuPdmDataIterator(self)


class EventFilterOptions(object):
    class Cond(Enum):
        lt = 1
        le = 2
        eq = 3
        ge = 4
        gt = 5

    n_persist = -1
    n_persist_cond = Cond.lt
    sum_l1_pdm = -1
    sum_l1_pdm_cond = Cond.lt
    sum_l1_ec_one = -1
    sum_l1_ec_one_cond = Cond.lt
    sum_l1_pmt_one = -1
    sum_l1_pmt_one_cond = Cond.lt

    def cmp(self, val1, val2, cmp_type):
        if val1 is None or val2 is None:
            return True
        if cmp_type == EventFilterOptions.Cond.lt:
            return val1 < val2
        elif cmp_type == EventFilterOptions.Cond.le:
            return val1 <= val2
        elif cmp_type == EventFilterOptions.Cond.eq:
            return val1 == val2
        elif cmp_type == EventFilterOptions.Cond.ge:
            return val1 >= val2
        else:
            val1 > val2

    def has_one_cell_valid(self, v, m, cond):
        for m_v in m.flatten():
            if self.cmp(v, m_v, cond): #TODO one or all?
                return True
        return False

    def check_pdm_gtu(self, pdm_gtu_data):
        if not (self.cmp(self.n_persist, pdm_gtu_data.n_persist, self.n_persist_cond) and self.cmp(self.sum_l1_pdm, pdm_gtu_data.sum_l1_pdm, self.sum_l1_pdm_cond)):
            return False
        if isinstance(pdm_gtu_data.sum_l1_ec, np.ndarray) and not self.has_one_cell_valid(self.sum_l1_ec_one, pdm_gtu_data.sum_l1_ec, self.sum_l1_ec_one_cond):
            return False
        if isinstance(pdm_gtu_data.sum_l1_pmt, np.ndarray) and not self.has_one_cell_valid(self.sum_l1_pmt_one, pdm_gtu_data.sum_l1_pmt, self.sum_l1_pmt_one_cond):
            return False
        return True
