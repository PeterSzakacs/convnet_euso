import numpy as np
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
# import collections

import utility_funtions as utl
import processing_functions as proc

from collections import Iterable as collections_Iterable

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')


def draw_pmts(ax, exp_tree, alpha=.4, edgecolor='black', linewidth=1, facecolor='none',
              pmt_width_overwrite=None, pmt_height_overwrite=None, pmt_count_x_overwrite=None, pmt_count_y_overwrite=None):
    pmt_width =  exp_tree.pdmPixelCountX / exp_tree.pmtCountX  if pmt_width_overwrite is None else pmt_width_overwrite
    pmt_height = exp_tree.pdmPixelCountY / exp_tree.pmtCountY if pmt_height_overwrite is None else pmt_height_overwrite
    pmt_count_x = exp_tree.pmtCountX if pmt_count_x_overwrite is None else pmt_count_x_overwrite
    pmt_count_y = exp_tree.pmtCountY if pmt_count_y_overwrite is None else pmt_count_y_overwrite
    draw_pmts_simple(ax, pmt_width, pmt_height, pmt_count_x, pmt_count_y, alpha, edgecolor, linewidth, facecolor)

def draw_pmts_simple(ax, pmt_width, pmt_height, pmt_count_x, pmt_count_y, alpha=.4, edgecolor='black', linewidth=1, facecolor='none'):
    for tmp_pmt_x in range(pmt_count_x):
        for tmp_pmt_y in range(pmt_count_y):
            rect = mpl_patches.Rectangle((tmp_pmt_x*pmt_width, tmp_pmt_y*pmt_height), pmt_width, pmt_height,
                                         linewidth=linewidth, alpha=alpha, edgecolor=edgecolor, facecolor=facecolor)
            ax.add_patch(rect)


def draw_l1trg_events(ax, l1trg_events, alpha=1, edgecolor='r', facecolor='none', linewidth=1):
    for l1trg_ev in l1trg_events:
        # if l1trg_ev.pix_row == 1:
        x = y = -1
        if isinstance(l1trg_ev,(tuple, list, np.ndarray, set)):
            x = l1trg_ev[1]
            y = l1trg_ev[0]
        else:
        # if isinstance(l1trg_ev, _event_reading_L1TrgEvent):
            x = l1trg_ev.pix_col
            y = l1trg_ev.pix_row

        rect = mpl_patches.Rectangle((x, y), 1, 1,
                                     linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
        ax.add_patch(rect)


def draw_watermark(fig, watermark):
    if not watermark or not fig:
        return
    # fig.text(0, 0, watermark, fontsize=5, color='gray', ha='left', va='top', alpha=.5)
    fig.text(.01, .99, watermark, fontsize=6, color='gray', ha='left', va='top', alpha=1)
    # fig.text(0.95, 0.05, watermark, fontsize=10, color='gray', ha='right', va='bottom', alpha=.5)


def visualize_frame(image, exp_tree=None, l1trg_events=[], title=None, show=True, vmin=None, vmax=None,
                    xlabel="x [pixel]", ylabel="y [pixel]", zlabel="max number of counts within the window",
                    ax=None,
                    watermark=None,
                    save_fig_pathname=None, fig_dimensions=(6, 6),
                    pmt_width_overwrite=None, pmt_height_overwrite=None,
                    pmt_count_x_overwrite=None, pmt_count_y_overwrite=None,
                    draw_colorbar=True, savefig_info_out_file=None,
                    lines=None, value_lines_groups={}, skip_zero_value_line_group=True,
                    lines_colormap_name='cool', line_alpha=None, do_pixel_centering=True,
                    lines_colors=None):
    if not ax:
        fig, ax1 = plt.subplots(1)
    else:
        ax1 = ax

    # det_width =  exp_tree.pmtCountX * exp_tree.pixelCountX
    # det_height = exp_tree.pmtCountY * exp_tree.pixelCountY

    cax4 = ax1.imshow(image, extent=[0, image.shape[1], image.shape[0], 0], vmin=vmin, vmax=vmax)

    if exp_tree:
        draw_pmts(ax1, exp_tree,
                  pmt_width_overwrite=pmt_width_overwrite, pmt_height_overwrite=pmt_height_overwrite,
                  pmt_count_x_overwrite=pmt_count_x_overwrite, pmt_count_y_overwrite=pmt_count_y_overwrite)

    if l1trg_events is not None and len(l1trg_events) > 0:
        draw_l1trg_events(ax1, l1trg_events, alpha=.8)

    if title:
        ax1.set_title(title)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if draw_colorbar:
        ax1.figure.colorbar(cax4, ax=ax1, label=zlabel)

    if isinstance(lines,collections_Iterable) and len(lines) > 0:
        # lines_colors = None
        if value_lines_groups is not None and len(value_lines_groups) > 1:
            # value_lines_groups_cpy = collections.OrderedDict(value_lines_groups)
            value_lines_groups_keys = value_lines_groups.keys()
            if skip_zero_value_line_group:
                value_lines_groups_keys = [k for k in value_lines_groups.keys() if k > 0]
            if len(value_lines_groups_keys) > 1:
                cmap = plt.get_cmap(lines_colormap_name)
                max_key = max(value_lines_groups_keys)
                min_key = min(value_lines_groups_keys)
                line_values = utl.key_vals2val_keys(value_lines_groups, exclusive=True)
                lines_colors = utl.translate_struct(line_values,
                                                    lambda v: cmap(float(v - min_key) / float(max_key - min_key)))

        if line_alpha is None:
            # line_alpha = (0.7*10)/len(lines)
            # a = (0.6 - 0.7) / 10 # == -0.01
            # b = 0.7 - a * 10 # == 0.7 - ((0.6-0.7)/10)*10 # == 0.8
            # line_alpha = max(a*len(lines)+b, 0.15)
            line_alpha = max(-0.01*len(lines) + 0.8, 0.15)

        for line in lines:
            if do_pixel_centering:
                p = proc.calc_line_coords_float(line[1], line[0], image.shape[1] - 0.9999, image.shape[0] - 0.9999)
                if p is None:
                    continue
                # p[,{y,x}]
                # centering coords to middle of pixels
                for i in range(len(p)):  # iterate over points
                    if p[i, 0] + 0.5 <= image.shape[0] and p[i, 1] + 0.5 <= image.shape[1]:
                        p[i, 0] += 0.5
                        p[i, 1] += 0.5
                    else:
                        raise RuntimeError('Unexpected line position')
            else:
                p = proc.calc_line_coords_float(line[1], line[0], image.shape[1], image.shape[0])
                if p is None:
                    continue

            if np.allclose(p[0], p[1]):
                marker_char = '.'
            else:
                marker_char = '-'

            if lines_colors is not None and tuple(line) in lines_colors:
                ax1.plot((p[:, 1]), (p[:, 0]), marker_char, color=lines_colors[tuple(line)], alpha=line_alpha)
                # line_values[line]/max_key)
            else:
                ax1.plot((p[:, 1]), (p[:, 0]), marker_char + 'r', alpha=line_alpha)

    # det_array = np.zeros_like(pcd)
    # pdm_pcd = pcd[0,0]
    # for l1trg_ev in gtu_pdm_data.l1trg_events:
    #     det_array[0][0][l1trg_ev.pix_row, l1trg_ev.pix_col] = gtu_pdm_data.photon_count_data[0][0][l1trg_ev.pix_row, l1trg_ev.pix_col]

    if ax is None and isinstance(fig, mpl.figure.Figure):    # in next version allow for custom axes
        fig.set_size_inches(fig_dimensions[0], fig_dimensions[1], forward=True)
        draw_watermark(fig, watermark)
        if save_fig_pathname:
            fig.savefig(save_fig_pathname)
            if savefig_info_out_file:
                print('Saved frame figure at "{}"'.format(save_fig_pathname), file=savefig_info_out_file)
    if show:
        plt.show()

    return ax1.figure, ax1  #TODO return cax4


def visualize_frame_gtu_x(image, exp_tree=None, l1trg_events=[], title=None, show=True, vmin=None, vmax=None,
                          xlabel="GTU", ylabel="y [pixel]", zlabel="max number of counts in a column",
                          ax=None,
                          watermark=None,
                          save_fig_pathname=None, fig_dimensions=(6, 6),
                          draw_colorbar=True, savefig_info_out_file=None):
    return visualize_frame(image, exp_tree, l1trg_events, title, show, vmin, vmax,
                           xlabel, ylabel, zlabel, ax, watermark, save_fig_pathname, fig_dimensions,
                           image.shape[1], None,
                           1, None,
                           draw_colorbar, savefig_info_out_file)


def visualize_frame_gtu_y(image, exp_tree=None, l1trg_events=[], title=None, show=True, vmin=None, vmax=None,
                          xlabel="GTU", ylabel="x [pixel]", zlabel="max number of counts in a row",
                          ax=None,
                          watermark=None,
                          save_fig_pathname=None, fig_dimensions=(6, 6),
                          draw_colorbar=True, savefig_info_out_file=None):
    return visualize_frame(image, exp_tree, l1trg_events, title, show, vmin, vmax,
                           xlabel, ylabel, zlabel, ax, watermark, save_fig_pathname, fig_dimensions,
                           image.shape[1], None,
                           1, None,
                           draw_colorbar, savefig_info_out_file)


def visualize_multiple_frames(frames, exp_tree=None, l1trg_events=[], titles=None, show=True, vmins=None, vmaxs=None,
                              xlabels="x [pixel]", ylabels="y [pixel]", zlabels="max number of counts within the window",
                              save_fig_pathname=None, watermark=None, single_ax_dimensions=(4.8,4.0),
                              pmt_width_overwrite=None, pmt_height_overwrite=None,
                              pmt_count_x_overwrite=None, pmt_count_y_overwrite=None, savefig_info_out_file=None):
    num_cols = int(np.ceil(np.sqrt(len(frames))))
    num_rows = int(np.ceil(len(frames)/num_cols))

    fig, axs = plt.subplots(num_rows, num_cols, squeeze=False)
    for i, frame in enumerate(frames):
        row = int(np.floor(i / num_cols))
        col = int(i % num_cols)
        ax = axs[row, col]
        title = None
        vmin = vmins
        vmax = vmaxs
        xlabel = xlabels
        ylabel = ylabels
        zlabel = zlabels
        
        if titles and isinstance(titles,(list,tuple)) and len(titles) > i:
            title = titles[i]
        if vmins and isinstance(vmins,(list,tuple)) and len(vmins) > i:
            vmin = vmins[i]
        if vmaxs and isinstance(vmaxs,(list,tuple)) and len(vmaxs) > i:
            vmax = vmaxs[i]
        if xlabels and isinstance(xlabels,(list,tuple)) and len(xlabels) > i:
            xlabel = xlabels[i]
        if ylabels and isinstance(ylabels,(list,tuple)) and len(ylabels) > i:
            ylabel = ylabels[i]
        if zlabels and isinstance(zlabels,(list,tuple)) and len(zlabels) > i:
            zlabel = zlabels[i]
            
        visualize_frame(frame,exp_tree, l1trg_events, title, False, vmin, vmax, xlabel, ylabel, zlabel, ax,
                  pmt_width_overwrite=pmt_width_overwrite, pmt_height_overwrite=pmt_height_overwrite,
                  pmt_count_x_overwrite=pmt_count_x_overwrite, pmt_count_y_overwrite=pmt_count_y_overwrite,
                  savefig_info_out_file=savefig_info_out_file)

    for j in range(i,num_cols*num_rows):
        row = int(np.floor(j / num_cols))
        col = int(j % num_cols)
        axs[row, col].axis('off')

    if isinstance(fig, mpl.figure.Figure):    # in next version allow for custom axes maybe
        fig_width = num_cols*single_ax_dimensions[0]
        fig_height = num_rows*single_ax_dimensions[1]
        fig.set_size_inches(fig_width, fig_height, forward=True)
        draw_watermark(fig, watermark)
        if save_fig_pathname:
            fig.savefig(save_fig_pathname)
            if savefig_info_out_file:
                print('Saved multiple frames figure at "{}"'.format(save_fig_pathname), file=savefig_info_out_file)

    if show:
        plt.show()


def visualize_frame_num_relation(frame_num_x, l1trg_events_by_frame_num=[], att_name="pix_col", title=None, show=True,
                                 vmin=None, vmax=None,
                                 xlabel="GTU", ylabel="x [pixel]", zlabel="max number of counts in the row",
                                 ax=None, save_fig_pathname=None, watermark=None, fig_dimensions=(6,6),
                                 exp_tree=None, savefig_info_out_file=None):
    # if title is not None:
    #     ax.set_title(title)
    #
    # cax = ax.imshow(frame_num_x, extent=[0, frame_num_x.shape[1], frame_num_x.shape[0], 0], vmin=vmin, vmax=vmax)
    # fig.colorbar(cax)

    pmt_width_overwrite = frame_num_x.shape[1] if att_name=='pix_row' else None # if rows are plotted, columns are integrated
    pmt_height_overwrite = frame_num_x.shape[1] if att_name=='pix_col' else None # if rows are plotted, rows are integrated
    pmt_count_x_overwrite = 1 if att_name=='pix_row' else None
    pmt_count_y_overwrite = 1 if att_name=='pix_col' else None

    fig, ax1 = visualize_frame(frame_num_x, exp_tree, None, title, show, vmin, vmax, xlabel, ylabel, zlabel, ax,
                              pmt_width_overwrite=pmt_width_overwrite, pmt_height_overwrite=pmt_height_overwrite,
                              pmt_count_x_overwrite=pmt_count_x_overwrite, pmt_count_y_overwrite=pmt_count_y_overwrite)

    if l1trg_events_by_frame_num is not None:
        for frame_num, l1trg_events in enumerate(l1trg_events_by_frame_num):
            # if l1trg_ev.pix_row == 1:
            for l1trg_ev in l1trg_events:
                if isinstance(l1trg_ev, (tuple, list, np.ndarray, set)):
                    # x = l1trg_ev[1]
                    y = l1trg_ev[0 if att_name=='pix_row' else 1]
                else:
                    # if isinstance(l1trg_ev, _event_reading_L1TrgEvent):
                    # x = l1trg_ev.pix_col
                    y = getattr(l1trg_ev, att_name)

                rect = mpl_patches.Rectangle((frame_num, y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
                ax1.add_patch(rect)

    if ax is None and isinstance(fig, mpl.figure.Figure):    # in next version allow for custom axes
        fig.set_size_inches(fig_dimensions[0], fig_dimensions[1], forward=True)
        draw_watermark(fig, watermark)
        if save_fig_pathname:
            fig.savefig(save_fig_pathname)
            if savefig_info_out_file:
                print('Saved frame number relation figure at "{}"'.format(save_fig_pathname), file=savefig_info_out_file)

    if show:
        plt.show()

    return fig, ax1


def print_frame_info(gtu_pdm_data, file=sys.stdout):
    print("GTU {} ({}); trgBoxPerGTU: {}, trgPmtPerGTU: {}, trgPmtPerGTU: {}; nPersist: {}, gtuInPersist: {}" \
          .format(gtu_pdm_data.gtu, gtu_pdm_data.gtu_time,
                  gtu_pdm_data.trg_box_per_gtu, gtu_pdm_data.trg_pmt_per_gtu, gtu_pdm_data.trg_ec_per_gtu,
                  gtu_pdm_data.n_persist, gtu_pdm_data.gtu_in_persist),
          file=file)
    for l1trg_ev in gtu_pdm_data.l1trg_events:
        print("    pix: {},{}; PMT: {},{}; EC: {}; sumL1: {}, thrL1: {}, persistL1: {} ;"
              " original pmt: {},{} ; original pix: {},{}" \
                .format(l1trg_ev.pix_col,
                    l1trg_ev.pix_row, l1trg_ev.pmt_col, l1trg_ev.pmt_row, l1trg_ev.ec_id,
                    l1trg_ev.sum_l1, l1trg_ev.thr_l1, l1trg_ev.persist_l1,
                    l1trg_ev.o_pmt_col, l1trg_ev.o_pmt_row, l1trg_ev.o_pix_col, l1trg_ev.o_pix_row),
              file=file)


def visualize_hough_lines(image, lines, title=None, value_lines_groups=None, exp_tree=None, l1trg_events=None,
                          xlabel="x [pixel]", ylabel="y [pixel]", zlabel="max number of counts within the window",
                          value_lines_colorbar_label='Num. of normalized counts for the line',
                          save_fig_pathname=None, watermark=None, fig_dimensions=(8, 8),
                          skip_zero_value_line_group=True,
                          pmt_width_overwrite=None, pmt_height_overwrite=None,
                          pmt_count_x_overwrite=None, pmt_count_y_overwrite=None,
                          extra_lines=None, trigg_alpha=0.3, line_alpha=None,
                          lines_colormap_name='cool', extra_lines_colormap_name='brg',
                          do_extra_lines_legend=True, extra_line_labels=[], extra_lines_legend_ncol=4,
                          gridspec_height_ratios=(20, 1, 1), gridspec_hspace=.4,
                          do_pixel_centering=False, savefig_info_out_file=None):
    do_extra_lines_legend = do_extra_lines_legend and extra_lines and isinstance(extra_lines, (list, tuple)) and len(
        extra_lines) > 0
    fig = None
    ax2 = None
    lines_colors = None
    if value_lines_groups is not None and len(value_lines_groups) > 1:
        # value_lines_groups_cpy = collections.OrderedDict(value_lines_groups)
        value_lines_groups_keys = value_lines_groups.keys()
        if skip_zero_value_line_group:
            value_lines_groups_keys = [k for k in value_lines_groups.keys() if k > 0]
        if len(value_lines_groups_keys) > 1:
            cmap = plt.get_cmap(lines_colormap_name)
            max_key = max(value_lines_groups_keys)
            min_key = min(value_lines_groups_keys)
            line_values = utl.key_vals2val_keys(value_lines_groups, exclusive=True)
            lines_colors = utl.translate_struct(line_values, lambda v: cmap(float(v - min_key) / float(max_key - min_key)))

            gridspec_nrows = 2
            if do_extra_lines_legend:
                gridspec_nrows += 1

            gs = mpl.gridspec.GridSpec(gridspec_nrows, 1, hspace=.4,
                                       height_ratios=gridspec_height_ratios[:gridspec_nrows])
            fig = plt.figure()
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = None
            if do_extra_lines_legend:
                ax3 = plt.subplot(gs[2])

            norm = mpl.colors.Normalize(vmin=min_key, vmax=max_key)
            cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation='horizontal')
            cb1.set_label(value_lines_colorbar_label)

    if not fig:
        if do_extra_lines_legend:
            gs = mpl.gridspec.GridSpec(2, 1, hspace=gridspec_hspace,
                                       height_ratios=(gridspec_height_ratios[0],gridspec_height_ratios[-1]))
            fig = plt.figure()
            ax1 = plt.subplot(gs[0])
            ax3 = plt.subplot(gs[1])
        else:
            fig, ax1 = plt.subplots(1)

    cax4 = ax1.imshow(image, extent=[0, image.shape[1], image.shape[0], 0])

    if exp_tree:
        draw_pmts(ax1, exp_tree,
                  pmt_width_overwrite=pmt_width_overwrite, pmt_height_overwrite=pmt_height_overwrite,
                  pmt_count_x_overwrite=pmt_count_x_overwrite, pmt_count_y_overwrite=pmt_count_y_overwrite)

    if l1trg_events:
        draw_l1trg_events(ax1, l1trg_events, alpha=trigg_alpha)

    if title:
        ax1.set_title(title)  # "Hough input img (phi normalization)"
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    fig.colorbar(cax4, ax=ax1, label=zlabel)

    if line_alpha is None:
        # line_alpha = (0.7*10)/len(lines)
        # a = (0.6 - 0.7) / 10 # == -0.01
        # b = 0.7 - a * 10 # == 0.7 - ((0.6-0.7)/10)*10 # == 0.8
        # line_alpha = max(a*len(lines)+b, 0.15)
        line_alpha = max(-0.01*len(lines) + 0.8, 0.15)

    for line in lines:
        if do_pixel_centering:
            p = proc.calc_line_coords_float(line[1], line[0], image.shape[1] - 0.9999, image.shape[0] - 0.9999)
            if p is None:
                continue
            # p[,{y,x}]
            # centering coords to middle of pixels
            for i in range(len(p)):  # iterate over points
                if p[i, 0] + 0.5 <= image.shape[0] and p[i, 1] + 0.5 <= image.shape[1]:
                    p[i, 0] += 0.5
                    p[i, 1] += 0.5
                else:
                    raise RuntimeError('Unexpected line position')
        else:
            p = proc.calc_line_coords_float(line[1], line[0], image.shape[1], image.shape[0])
            if p is None:
                continue

        if np.allclose(p[0], p[1]):
            marker_char = '.'
        else:
            marker_char = '-'

        if lines_colors is not None and tuple(line) in lines_colors:
            ax1.plot((p[:, 1]), (p[:, 0]), marker_char, color=lines_colors[tuple(line)], alpha=line_alpha)
            # line_values[line]/max_key)
        else:
            ax1.plot((p[:, 1]), (p[:, 0]), marker_char + 'r', alpha=line_alpha)

    if extra_lines:
        extra_lines_colormap = plt.get_cmap(extra_lines_colormap_name)  # 'autumn'
        colors = [extra_lines_colormap(each)
                  for each in np.linspace(0, 1, len(extra_lines))]

        plots = [None] * len(extra_lines)
        for li, (line, color) in enumerate(zip(extra_lines, colors)):
            # p = proc.calc_line_coords(line[1], line[0], image.shape[1], image.shape[0])
            if do_pixel_centering:
                p = proc.calc_line_coords_float(line[1], line[0], image.shape[1] - 0.9999, image.shape[0] - 0.9999)
                if p is None:
                    continue
                for i in range(len(p)):  # iterate over points
                    if p[i, 0] + 0.5 <= image.shape[0] and p[i, 1] + 0.5 <= image.shape[1]:
                        p[i, 0] += 0.5
                        p[i, 1] += 0.5
                    else:
                        raise RuntimeError('Unexpected line position')
            else:
                p = proc.calc_line_coords_float(line[1], line[0], image.shape[1], image.shape[0])
                if p is None:
                    continue
            plot, = ax1.plot((p[:, 1]), (p[:, 0]), ':', color=color)
            plots[li] = plot
        # endfor

        if do_extra_lines_legend:  # and ax3:
            if extra_line_labels is None:
                extra_line_labels = []
            labels = []
            notnull_plots = []
            for i, (line, plot) in enumerate(zip(extra_lines, plots)):
                if plot is None:
                    continue
                if i < len(extra_line_labels):
                    labels.append(extra_line_labels[i])
                else:
                    labels.append("({:.2f}, {:.2f})".format(*line))
                notnull_plots.append(plot)
            ax3.legend(notnull_plots, labels, loc='center', mode='expand', frameon=False, borderaxespad=0, borderpad=0,
                       ncol=extra_lines_legend_ncol)
            ax3.set_axis_off()
        # if ax3 ... # legend # TODO

    if fig:  # in next version allow for custom axes
        fig.set_size_inches(fig_dimensions[0], fig_dimensions[1], forward=True)
        draw_watermark(fig, watermark)
        if save_fig_pathname:
            fig.savefig(save_fig_pathname)
            if savefig_info_out_file:
                print('Saved hough lines figure at "{}"'.format(save_fig_pathname), file=savefig_info_out_file)

    return fig, (ax1, ax2)


# deprecated
def visualize_hough_lines_old(image, lines, title=None, value_lines_groups=None, exp_tree=None, l1trg_events=None,
                          xlabel="x [pixel]", ylabel="y [pixel]", zlabel="max number of counts within the window",
                          value_lines_colorbar_label='Num. of normalized counts for the line',
                          save_fig_pathname=None, watermark=None, fig_dimensions=(8, 8),
                          skip_zero_value_line_group=True,
                          pmt_width_overwrite=None, pmt_height_overwrite=None,
                          pmt_count_x_overwrite=None, pmt_count_y_overwrite=None,
                          extra_lines=None, trigg_alpha=0.3, line_alpha=None,
                          lines_colormap_name='cool', extra_lines_colormap_name='brg',
                          do_extra_lines_legend=True, extra_line_labels=[], extra_lines_legend_ncol=4,
                          gridspec_height_ratios=(20, 1, 1), gridspec_hspace=.4,
                          do_pixel_centering=False, savefig_info_out_file=None):

    do_extra_lines_legend = do_extra_lines_legend and extra_lines and isinstance(extra_lines, (list, tuple)) and len(extra_lines) > 0
    fig = None
    ax2 = None
    lines_colors = None
    if value_lines_groups is not None and len(value_lines_groups) > 1:
        # value_lines_groups_cpy = collections.OrderedDict(value_lines_groups)
        value_lines_groups_keys = value_lines_groups.keys()
        if skip_zero_value_line_group:
            value_lines_groups_keys = [k for k in value_lines_groups.keys() if k > 0]
        if len(value_lines_groups_keys) > 1:
            cmap = plt.get_cmap(lines_colormap_name)
            max_key = max(value_lines_groups_keys)
            min_key = min(value_lines_groups_keys)
            line_values = utl.key_vals2val_keys(value_lines_groups, exclusive=True)
            lines_colors = utl.translate_struct(line_values, lambda v: cmap(float(v - min_key) / float(max_key - min_key)))

            gridspec_nrows = 2
            if do_extra_lines_legend:
                gridspec_nrows += 1

            gs = mpl.gridspec.GridSpec(gridspec_nrows, 1, hspace=gridspec_hspace,
                                       height_ratios=gridspec_height_ratios[:gridspec_nrows])
            fig = plt.figure()
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = None
            if do_extra_lines_legend:
                ax3 = plt.subplot(gs[2])

            norm = mpl.colors.Normalize(vmin=min_key, vmax=max_key)
            cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                            norm=norm,
                                            orientation='horizontal')
            cb1.set_label(value_lines_colorbar_label)

    if not fig:
        if do_extra_lines_legend:
            gs = mpl.gridspec.GridSpec(2, 1, hspace=gridspec_hspace,
                                       height_ratios=(gridspec_height_ratios[0],gridspec_height_ratios[-1]))
            fig = plt.figure()
            ax1 = plt.subplot(gs[0])
            ax3 = plt.subplot(gs[1])
        else:
            fig, ax1 = plt.subplots(1)

    cax4 = ax1.imshow(image, extent=[0, image.shape[1], image.shape[0], 0])

    if exp_tree:
        draw_pmts(ax1, exp_tree,
                  pmt_width_overwrite=pmt_width_overwrite, pmt_height_overwrite=pmt_height_overwrite,
                  pmt_count_x_overwrite=pmt_count_x_overwrite, pmt_count_y_overwrite=pmt_count_y_overwrite)

    if l1trg_events:
        draw_l1trg_events(ax1, l1trg_events, alpha=trigg_alpha)

    if title:
        ax1.set_title(title)  # "Hough input img (phi normalization)"
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    # y0 = (acc_matrix_max_rho - 0 * np.cos(acc_matrix_max_phi)) / np.sin(angle)
    # y1 = (acc_matrix_max_rho - image.shape[1] * np.cos(angle)) / np.sin(angle)

    fig.colorbar(cax4, ax=ax1, label=zlabel)

    if line_alpha is None:
        # line_alpha = (0.7*10)/len(lines)
        # a = (0.6 - 0.7) / 10 # == -0.01
        # b = 0.7 - a * 10 # == 0.7 - ((0.6-0.7)/10)*10 # == 0.8
        # line_alpha = max(a*len(lines)+b, 0.15)
        line_alpha = max(-0.01*len(lines) + 0.8, 0.15)

    for line in lines:
        p = proc.calc_line_coords_float(line[1], line[0], image.shape[1], image.shape[0])
        # p[,{y,x}]
        if p is None:
            continue
        if do_pixel_centering:
            # centering coords to middle of pixels
            for i in range(len(p)):  # iterate over points
                if p[i, 1] == 0 or p[i, 1] == image.shape[1]:  # x == 0
                    # may cause non-required shifts for average lines from hough space
                    p[i, 0] += 0.5  # y+0.5
                else:  # x != 0
                    p[i, 1] += 0.5  # x+0.5

        if lines_colors is not None and tuple(line) in lines_colors:
            ax1.plot((p[:, 1]), (p[:, 0]), '-', color=lines_colors[tuple(line)],
                     alpha=line_alpha)  # line_values[line]/max_key)
        else:
            ax1.plot((p[:, 1]), (p[:, 0]), '-r', alpha=line_alpha)

    if extra_lines:
        extra_lines_colormap = plt.get_cmap(extra_lines_colormap_name)  # 'autumn'
        colors = [extra_lines_colormap(each)
                  for each in np.linspace(0, 1, len(extra_lines))]
        plots = []
        for line, color in zip(extra_lines, colors):
            # p = proc.calc_line_coords(line[1], line[0], image.shape[1], image.shape[0])
            p = proc.calc_line_coords_float(line[1], line[0], image.shape[1], image.shape[0])
            plot = None
            if p is not None:
                if do_pixel_centering:
                    # centering coords to middle of pixels
                    for i in range(len(p)):  # iterate over points
                        if p[i, 1] == 0 or p[i, 1] == image.shape[1]:  # x == 0
                            p[i, 0] += 0.5  # y+0.5
                        else:  # x != 0
                            p[i, 1] += 0.5  # x+0.5
                # print("line (y,x) [{},{}] , [{},{}]".format(p[0,0],p[0,1],p[1,0],p[1,1]))
                plot, = ax1.plot((p[:, 1]), (p[:, 0]), ':', color=color)
            plots.append(plot)

        if do_extra_lines_legend:  # and ax6:
            if extra_line_labels is None:
                extra_line_labels = []
            labels = []
            notnull_plots = []
            for i, (line, plot) in enumerate(zip(extra_lines, plots)):
                if plot is None:
                    continue
                if i < len(extra_line_labels):
                    labels.append(extra_line_labels[i])
                else:
                    labels.append("({:.2f}, {:.2f})".format(*line))
                notnull_plots.append(plot)
            ax3.legend(notnull_plots, labels, loc='center', mode='expand', frameon=False, borderaxespad=0, borderpad=0,
                       ncol=extra_lines_legend_ncol)
            ax3.set_axis_off()
        # if ax6 ... # legend # TODO

    if fig:  # in next version allow for custom axes
        fig.set_size_inches(fig_dimensions[0], fig_dimensions[1], forward=True)
        draw_watermark(fig, watermark)
        if save_fig_pathname:
            fig.savefig(save_fig_pathname)
            if savefig_info_out_file:
                print('Saved hough lines figure at "{}"'.format(save_fig_pathname), file=savefig_info_out_file)

    return fig, (ax1, ax2)


def _force_aspect(ax,aspect=1):
    # https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def visualize_hough_space(acc_matrix, phi_linspace, rho_range_opts, title=None,
                          xlabel=r"$\phi$ [deg]", ylabel=r"$\rho$ [pixels]", zlabel="Sum of normalized counts for the line",
                          ax = None,
                          save_fig_pathname=None, watermark=None, fig_dimensions=(8, 5), square_pixels=False,
                          savefig_info_out_file=None):
    #  ({} = {}*{} - ({} = -{} + {}) + {}/2)
    #        rho_step,rho_index, rho_correction_lower, size, max_distance, size,


    # fig2, (ax1, ax1b, ax2) = plt.subplots(3)
    if ax is None:
        fig, ax1 = plt.subplots(1)
    else:
        ax1 = ax
        fig = ax1.figure

    # lower indexes = lower values => flipud

    cax1 = ax1.imshow(np.flipud(acc_matrix), extent=[np.rad2deg(phi_linspace[0]), np.rad2deg(phi_linspace[-1]), rho_range_opts[0], rho_range_opts[1]], aspect='auto')
    # ax1b.imshow(rho_acc_matrix, extent=[np.rad2deg(phi_range[0]),np.rad2deg(phi_range[-1]), -max_distance, max_distance], aspect='auto')
    # ax2.imshow(nc_acc_matrix, extent=[np.rad2deg(phi_range[0]),np.rad2deg(phi_range[-1]), -max_distance, max_distance], aspect='auto')

    # fig3, ax3 = plt.subplots(1)
    # cax3 = ax3.imshow(acc_matrix, aspect='auto')
    # fig3.colorbar(cax3)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    fig.colorbar(cax1, label=zlabel, ax=ax1)

    if title:
        ax1.set_title(title)

    if square_pixels:
        _force_aspect(ax1, acc_matrix.shape[1]/acc_matrix.shape[0])

    if ax is None and isinstance(fig, mpl.figure.Figure):
        fig.set_size_inches(fig_dimensions[0],fig_dimensions[1],forward=True)
        draw_watermark(fig, watermark)
        if save_fig_pathname:
            fig.savefig(save_fig_pathname)
            if savefig_info_out_file:
                print('Saved hough space figure at "{}"'.format(save_fig_pathname), file=savefig_info_out_file)

    return fig, ax1, cax1


def visualize_hough_spaces(acc_matrix_list, phi_linspace_list, rho_range_opts_list, title_list=None,
                           xlabels=r"$\phi$ [deg]", ylabels=r"$\rho$ [pixels]", zlabels="Sum of normalized counts for the line",
                           save_fig_pathname=None, watermark=None, single_plot_dimensions=(8, 5), num_columns=4,
                           fallback_phi_linspace=(0, np.pi), fallback_rho_range=np.hypot(48, 48), square_pixels=False,
                           savefig_info_out_file=None):

    if isinstance(xlabels, str):
        xlabels = [xlabels]
    if isinstance(ylabels, str):
        ylabels = [ylabels]
    if isinstance(zlabels, str):
        zlabels = [zlabels]

    if len(acc_matrix_list) > num_columns:
        num_rows = int(np.ceil(len(acc_matrix_list)/num_columns))
    else:
        num_rows = 1
        num_columns = len(acc_matrix_list)

    fig, axs = plt.subplots(num_rows, num_columns)
    if num_columns > 1 or num_rows > 1:
        axs_flattened = axs.flatten()
    else:
        axs_flattened = [axs]

    last_xlabel = None
    last_ylabel = None
    last_zlabel = None

    for i, ax in enumerate(axs_flattened):
        acc_matrix = acc_matrix_list[i] if i < len(acc_matrix_list) else None
        phi_linspace = phi_linspace_list[i] if i < len(phi_linspace_list) else fallback_phi_linspace
        rho_range_opts = rho_range_opts_list[i] if i < len(rho_range_opts_list) else fallback_rho_range
        xlabel = xlabels[i] if i < len(xlabels) else last_xlabel
        ylabel = ylabels[i] if i < len(ylabels) else last_ylabel
        zlabel = zlabels[i] if i < len(zlabels) else last_zlabel
        title = title_list[i] if i < len(title_list) else "{}".format(i)

        if acc_matrix is None or phi_linspace is None or rho_range_opts is None:
            continue

        visualize_hough_space(acc_matrix, phi_linspace, rho_range_opts, title, xlabel, ylabel, zlabel, ax, square_pixels=square_pixels)

        last_xlabel = xlabel
        last_ylabel = ylabel
        last_zlabel = zlabel

    fig.set_size_inches(single_plot_dimensions[0]*num_columns,single_plot_dimensions[1]*num_rows,forward=True)
    draw_watermark(fig, watermark)
    if save_fig_pathname:
        fig.savefig(save_fig_pathname)
        if savefig_info_out_file:
            print('Saved multiple hough spaces figure at "{}"'.format(save_fig_pathname), file=savefig_info_out_file)

    return fig


def visualize_clusters_from_cluster_images(im_clusters, phi_linspace, rho_range_opts__pi_norm,
                                           title="Estimated clusters", xlabel="", ylabel="",
                                           save_fig_pathname=None, watermark=None, fig_dimensions=(8, 7),
                                           cmap='gist_rainbow', noise_color=[0, 0, 0, 1],
                                           core_sample__marker='o', not_core_sample__marker='s',
                                           core_sample__markersize=4, not_core_sample__markersize=2,
                                           plot_indexes=(1,0), axes_lim=(None, None),
                                           alpha=0.5, aspect='auto', force_aspect=False,
                                           line_labels=[],
                                           core_sample_indeces=True,
                                           savefig_info_out_file=None):
    im_clusters_list = []
    im__pi_norm_lines_list = []

    for cluster_i, (seed, im) in enumerate(im_clusters.items()):
        im__htpos_list = np.transpose(np.where(im > 0))
        im_clusters_list += [cluster_i for ii in range(len(im__htpos_list))]
        im__pi_norm_lines_list += proc.hs_indexes2val(im__htpos_list, phi_linspace, rho_range_opts__pi_norm)

    if None in axes_lim:
        axes_lim_c = list(axes_lim)
        if axes_lim_c[0] is None:
            axes_lim_c[0] = (0,np.pi)
        if axes_lim_c[1] is None:
            axes_lim_c[1] = (rho_range_opts__pi_norm[0], rho_range_opts__pi_norm[1])
    else:
        axes_lim_c = axes_lim

    visualize_clusters(np.array(im__pi_norm_lines_list), np.array(im_clusters_list), core_sample_indeces,
                       title, xlabel, ylabel,
                       save_fig_pathname, watermark, fig_dimensions,
                       cmap, noise_color,
                       core_sample__marker, not_core_sample__marker,
                       core_sample__markersize, not_core_sample__markersize,
                       plot_indexes, axes_lim_c, alpha, aspect, force_aspect,
                       line_labels, savefig_info_out_file)


def visualize_clusters(data, labels, core_sample_indices,
                       title="Estimated clusters", xlabel="", ylabel="",
                       save_fig_pathname=None, watermark=None, fig_dimensions=(8, 7),
                       cmap='gist_rainbow', noise_color=[0, 0, 0, 1],
                       core_sample__marker='o', not_core_sample__marker='s',
                       core_sample__markersize=4, not_core_sample__markersize=2,
                       plot_indexes=(1,0), axes_lim=(None, None),
                       alpha=0.5, aspect='auto', force_aspect=False,
                       line_labels=[], savefig_info_out_file=None):
    pi0 = plot_indexes[0]
    pi1 = plot_indexes[1]

    core_samples_mask = None
    if not isinstance(core_sample_indices,bool):
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[core_sample_indices] = True

    fig, ax = plt.subplots(1)

    scatter_cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    # core_samples_data = data[core_samples_mask]
    # not_core_samples_data = data[~core_samples_mask]
    # ax.scatter(core_samples_data[:, pi0], core_samples_data[: pi1], c=labels, cmap=scatter_cmap_obj, marker=core_sample__marker, markersize=core_sample__markersize)
    # ax.scatter(not_core_samples_data[:, pi0], not_core_samples_data[: pi1], c=labels, cmap=scatter_cmap_obj, marker=not_core_sample__marker, markersize=not_core_sample__markersize)
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [scatter_cmap_obj(each) for each in np.linspace(0, 1, len(unique_labels))] # plt.cm.Spectral(each)

    plot_lines = []
    plot_line_labels = []

    for i, (k, col) in enumerate(zip(unique_labels, colors)):
        if k == -1:
            # Black used for noise.
            col = noise_color #[0, 0, 0, 1]

        class_member_mask = (labels == k)

        if core_samples_mask is None:
            xy = data[class_member_mask]
        else:
            xy = data[class_member_mask & core_samples_mask]

        x_vals = xy[:, pi0]
        l, = ax.plot(x_vals, xy[:, pi1], core_sample__marker, markerfacecolor=tuple(col),
                 markeredgecolor=None, markeredgewidth=0,
                 markersize=core_sample__markersize, alpha=alpha)
        if (len(line_labels) > i or isinstance(line_labels, str)) and len(x_vals) > 0:
            plot_lines.append(l)
            line_label_str = line_labels if isinstance(line_labels, str) else line_labels[i]
            plot_line_labels.append(line_label_str.format(k, 'core sample'))

        if core_samples_mask is not None:
            xy = data[class_member_mask & ~core_samples_mask]
            x_vals = xy[:, pi0]
            l, = ax.plot(x_vals, xy[:, pi1], not_core_sample__marker, markerfacecolor=tuple(col),
                     markeredgecolor=None, markeredgewidth=0,
                     markersize=not_core_sample__markersize, alpha=alpha)
            if (len(line_labels) > i or isinstance(line_labels, str)) and len(x_vals) > 0:
                plot_lines.append(l)
                line_label_str = line_labels if isinstance(line_labels, str) else line_labels[i]
                plot_line_labels.append(line_label_str.format(k, 'not core sample'))

    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if axes_lim[0] is not None:
        ax.set_xlim(*axes_lim[0])
    if axes_lim[1] is not None:
        ax.set_ylim(*axes_lim[1])

    if force_aspect:
        _force_aspect(ax, aspect)
    else:
        ax.set_aspect(aspect)

    if len(plot_line_labels) > 0:
        ax.legend(plot_lines, plot_line_labels)

    fig.set_size_inches(fig_dimensions[0], fig_dimensions[1], forward=True)
    draw_watermark(fig, watermark)
    if save_fig_pathname:
        fig.savefig(save_fig_pathname)
        if savefig_info_out_file:
            print('Saved clusters figure at "{}"'.format(save_fig_pathname), file=savefig_info_out_file)


def show_plots():
    plt.show()


def close_plots():
    plt.close('all')
