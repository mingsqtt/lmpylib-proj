import numpy as np
import pandas as pd

data = pd.read_csv("data/data.csv")
import_data = data
month_lbls = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_lbls_map = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
wday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
data["month_lbl"] = [month_lbls[m-1] for m in data.month.values]
grpbyprod = data[["prod_cat", "month", "po_1"]].groupby(["prod_cat", "month"]).count()

barplot2(grpbyprod, subplots_ncol=4, horizontal=False, x_scale_rotation=90, title="Main Title", xlab="X", ylab="Y", x_scale_label_mapper=month_lbls, show=False, sort_by="y", ascending=False, y_scale_range="fixed")
barplot2(grpbyprod, subplots_ncol=4, horizontal=True, x_scale_rotation=0, subplot_adjust=(0.9, 0.1, 0.4, 0.2), title="Main Title", xlab="X", ylab="Y", sort_by="y", ascending=False)
barplot(grpbyprod, horizontal=False, x_scale_rotation=90, title="Main Title", xlab="X", ylab="Y", x_axis_label_mapper=month_lbls, show=False)



barplot(data[["month", "po_1"]].groupby(["month"]).count(), sort_by="x", ascending=False, x_scale_label_mapper=month_lbls_map, horizontal=True)
barplot(data[["month", "po_1"]].groupby(["month"]).count(), sort_by="x", ascending=False, x_scale_label_mapper=month_lbls_map, horizontal=False, x_scale_rotation=45)
barplot(data[["mode", "po_1"]].groupby(["mode"]).count(), sort_by="x", ascending=False, horizontal=True)


plot(np.linspace(0.04, 0.283, 100), np.linspace(0.04, 0.283, 100)**2, color="green", marker_size=5, x_scale_rotation=90, style="point", show=True)
plot(np.linspace(0.04, 0.283, 100), np.linspace(0.04, 0.283, 100), color="red", marker_size=5, x_scale_ticks=[0.04, 0.05, 0.07, 0.1, 0.15, 0.27, 0.283], x_scale_rotation=90, style="point")

type(data[["ata", "nday_gate_to_plant"]].groupby("ata").sum().index[0])

data.loc[data.dest == "CY", "dest"] = "YC"
x = data.loc[data.dest.isin(["Plant", "DC", "YC", "Bonded Location"]) ,["month", "nday_gate_to_plant", "dest"]]
barplot(x, group_by="dest", sort_by="y", ascending=False, x_scale_label_mapper=month_lbls_map, horizontal=True, legend_title="Port", sort_groups_by_value=True, groups_ascending=False, group_position="stack", width=0.7)

barplot(grpbyprod, group_by="", x_scale_rotation=90, x_scale_label_mapper=month_lbls, sort_by="y", group_position="dodge")

grpby3 = data.loc[data.dest.isin(["Plant", "DC", "CY", "Bonded Location"]) & data.dest_port.isin(["Manila", "Manila North Harbour", "Batangas Port"]), ["dest_port", "month", "dest", "po_1"]].groupby(["dest_port", "dest", "month"]).count()
barplot2(grpby3, x_scale_rotation=0, title="xxxx", sort_by="y", subplots_ncol=2)



grp_by_month_wday = import_data.loc[(import_data.dest == "Plant") & (import_data.dest_port == "Manila"), ["po_1", "ata_month", "ata_wday"]].groupby(["ata_month", "ata_wday"]).count()
grp_by_month_wday = grp_by_month_wday.fillna(0)
barplot2(grp_by_month_wday, subplots_ncol=7, y_scale_range="fixed", x_scale_rotation=45, facet_label_mapper=month_lbls, x_scale_label_mapper=wday_labels, show=False, color="red", title="Importation Volume Arrived at Manila Port on Each Day by Month (Delivered to Plant)")

auto_figure_size(2, 6, 0.93, 0.07, 0.2, 0.2, title_height=0.5)