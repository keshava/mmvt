from dell_panel import DellPanel


def run(mmvt):
    for group in DellPanel.groups:
        color = next(DellPanel.colors)
        for elc_ind in group:
            mmvt.coloring.object_coloring(DellPanel.names[elc_ind], tuple(color))
