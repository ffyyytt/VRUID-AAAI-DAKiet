import pickle
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser("VRUID")
parser.add_argument("-datapath", help="Path to dataset", nargs='?', type=str, default="/kaggle/input/aaai-25-visually-rich-document-vrd-iu-leaderboard")

test, val = pickle.load(open(f"{args.datapath}/test_data.pkl", "rb")), pickle.load(open(f"{args.datapath}/val_data.pkl", "rb"))
test_data = pickle.load(open(f"{args.datapath}/val_data.pkl", "rb")) | pickle.load(open(f"{args.datapath}/test_data.pkl", "rb"))
figure_child_ids, figure_parent_ids, figure_predictions = pickle.load(open("output_figure.pickle", "rb"))
table_child_ids, table_parent_ids, table_predictions = pickle.load(open("output_table.pickle", "rb"))
form_child_ids, form_parent_ids, form_predictions = pickle.load(open("output_form.pickle", "rb"))
list_child_ids, list_parent_ids, list_predictions = pickle.load(open("output_list.pickle", "rb"))
form_body_child_ids, form_body_parent_ids, form_body_predictions = pickle.load(open("output_form_body.pickle", "rb"))

id_to_page = {}
for file_id, k in enumerate(test_data.keys()):
    for obj in test_data[k]["components"]:
        id_to_page[obj["object_id"]] = obj["page"]

ID = []
PARENT = []
allobjects = []

no_parent = ['abstract', 'appendix_list', 'cross', 'figure', 'form_title', 'list_of_figures', 'list_of_tables', 'other', 'references', 'report_title', 'section', 'summary', 'table', 'table_of_contents', 'title']
section_tree = []

for file_id, k in enumerate(test_data.keys()):
    objects = sorted(test_data[k]["components"], key=lambda x: test_data[k]["ordered_id"].index(x["object_id"]))
    cid = test_data[k]["ordered_id"]
    cparent = ["-1" if object["category"] in no_parent else "?" for object in objects]

    for i in range(len(objects)):
        if "subsection" in objects[i]["category"]:
            c = i-1
            while c > -1:
                if ("section" in objects[c]["category"]) and (len(objects[c]["category"]) < len(objects[i]["category"])):
                    if len(objects[i]["category"])-len(objects[c]["category"]) == 3:
                        cparent[i] = objects[c]["object_id"]
                    break
                c -= 1
            if cparent[i] == "?":
                cparent[i] = "-1"
        if objects[i]["category"] == "paragraph":
            c = i-1
            while c > -1:
                if ("section" in objects[c]["category"]):
                    cparent[i] = objects[c]["object_id"]
                    break
                c -= 1
            if cparent[i] == "?":
                for j in range(i):
                    c = i-1-j
                    if (objects[c]["category"] in ["summary", "abstract"]):
                        cparent[i] = objects[c]["object_id"]
                        break
            
            if cparent[i] == "?":
                cparent[i] = "-1"

        if objects[i]["category"] == "figure_caption":
            cparent[i] = int(figure_parent_ids[file_id][np.argmax(figure_predictions[file_id], axis=1)[figure_child_ids[file_id].index(objects[i]["object_id"])]][0])
            # if cparent[i] != -1 and id_to_page[cparent[i]] < id_to_page[objects[i]["object_id"]] or cparent[i] != -1 and id_to_page[cparent[i]] > id_to_page[objects[i]["object_id"]]+1:
            #     # if k in test:
            #     #     print(k, objects[i]["object_id"], id_to_page[cparent[i]], id_to_page[objects[i]["object_id"]])
            #     cparent[i] = -1
        
        if objects[i]["category"] == "table_caption":
            cparent[i] = int(table_parent_ids[file_id][np.argmax(table_predictions[file_id], axis=1)[table_child_ids[file_id].index(objects[i]["object_id"])]][0])
            # if cparent[i] != -1 and id_to_page[cparent[i]] < id_to_page[objects[i]["object_id"]] or cparent[i] != -1 and id_to_page[cparent[i]] > id_to_page[objects[i]["object_id"]]+1:
            #     # if k in test:
            #     #     print(k, objects[i]["object_id"], id_to_page[cparent[i]], id_to_page[objects[i]["object_id"]])
            #     cparent[i] = -1
        
        if objects[i]["category"] == "form":
            cparent[i] = form_parent_ids[file_id][np.argmax(form_predictions[file_id], axis=1)[form_child_ids[file_id].index(objects[i]["object_id"])]][0]

        if objects[i]["category"] == "form_body":
            cparent[i] = form_body_parent_ids[file_id][np.argmax(form_body_predictions[file_id], axis=1)[form_body_child_ids[file_id].index(objects[i]["object_id"])]][0]

        if objects[i]["category"] == "list":
            cparent[i] = list_parent_ids[file_id][np.argmax(list_predictions[file_id], axis=1)[list_child_ids[file_id].index(objects[i]["object_id"])]][0]

        
    ID += cid
    PARENT += cparent
    allobjects += objects

df = pd.DataFrame()
df["ID"] = ID
df["Parent"] = PARENT
df.to_csv("submission.csv", index=False)
