import streamlit as st
import pandas as pd
from io import StringIO
import networkx as nx
import random
import pandas as pd
import subprocess 
import numpy
import time
from tqdm import tqdm
from collections import defaultdict
import csv
import networkx as nx
import datetime
import math
from math import exp
import unicodecsv
import csv
import detect
import _pickle as cPickle

def run_algorithm(df,NETWORKNAME, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3):
    if gamma1 == 0 and gamma2 == 0 and gamma3 == 0:
        sys.exit(0)

    print ("Loading %s network" % NETWORKNAME)
    G = nx.DiGraph()

    with open(r'..\data\amazon\amazon_network.csv', 'r') as f:
        data = csv.reader(f)
        headers = next(data)
        for row in data:
            G.add_node(row[0]) 
            G.add_node(row[1]) 
            G.add_edge(row[0], row[1], weight=row[2])

    print ("Loaded")

    nodes = G.nodes()
    edges = G.edges(data=True)
    print (("%s network has %d nodes and %d edges") % (NETWORKNAME, len(nodes),len(edges)))

    u_name = set()
    p_name = set()

    for node in nodes:
        if node[0]=='A' or node[0:3]=="#oc":
            u_name.add(node)
        else:
            p_name.add(node)

    user_names = list(u_name)
    product_names = list(p_name)

    num_users = len(user_names)
    num_products = len(product_names)
    user_map = dict(zip(user_names, range(len(user_names))))
    product_map = dict(zip(product_names, range(len(product_names))))

    full_birdnest_user = cPickle.load(open("../data/%s_birdnest_user.pkl" % (NETWORKNAME), "rb"), encoding="latin1")
    full_birdnest_product = cPickle.load(open("../data/%s_birdnest_product.pkl" % (NETWORKNAME), "rb"), encoding="latin1")

    full_birdnest_edge = []

    try:
        print ('loading birdnest pickle')

        full_birdnest_edge = cPickle.load(open("../data/%s_edge_birdnest.pkl" % NETWORKNAME,"rb"), encoding="latin1")
        edge_map = cPickle.load(open("../data/%s_edge_map.pkl" % NETWORKNAME,"rb"), encoding="latin1")

        full_birdnest_edge = numpy.array(full_birdnest_edge)
        mn = min(full_birdnest_edge)
        mx = max(full_birdnest_edge)
        full_birdnest_edge = (full_birdnest_edge - mn)*1.0/(mx-mn+0.001)

    except:
        print (("Didnt find edge birdnest scores for %s network") % NETWORKNAME)
        full_birdnest_edge = [0.0]*len(edges)
        ae = zip(numpy.array(edges)[:,0], numpy.array(edges)[:, 1])
        edge_map = dict(zip(ae, range(len(edges))))

    for node in nodes:
        if node[0]=="A" or node[0:3]=="#oc":
            G._node[node]["fairness"] = 1 - full_birdnest_user[user_map[node]]
        else:
            G._node[node]["goodness"] = (1 - full_birdnest_product[product_map[node]] - 0.5) * 2

    for edge in edges:
        G[edge[0]][edge[1]]["fairness"] = 1 - full_birdnest_edge[edge_map[("u"+edge[0], "p"+edge[1])]]
    
    iter = 0
    du = dp = dr = 0

    # iteration for rev2
    while iter < 5:
        print ('-----------------')
        print (("Epoch number %d with du = %f, dp = %f, dr = %f, for (%d,%d,%d,%d,%d,%d,%d)") % (iter, du, dp, dr, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3))
        if numpy.isnan(du) or numpy.isnan(dp) or numpy.isnan(dr):
            break

        du = dp = dr = 0

        # Updating goodness of product
        currentgvals = []
        for node in nodes:
            if "B" not in node[0]:
                continue
            currentgvals.append(G._node[node]["goodness"])

        median_gvals = numpy.median(currentgvals) 

        for node in nodes:
            if "B" not in node[0]:
                continue

            inedges = G.in_edges(node, data=True)
            ftotal = 0.0
            gtotal = 0.0
            for edge in inedges:
                gtotal += float(edge[2]["fairness"]) * float(edge[2]["weight"])
            ftotal += 1.0

            kl_timestamp = ((1 - full_birdnest_product[product_map[node]]) - 0.5) * 2

            if ftotal > 0.0:
                mean_rating_fairness = (beta1 * median_gvals + beta2 * kl_timestamp + gtotal) / (beta1 + beta2 + ftotal)
            else:
                mean_rating_fairness = 0.0

            x = mean_rating_fairness

            if x < -1.0:
                x = -1.0
            if x > 1.0:
                x = 1.0
            dp += abs(G._node[node]["goodness"] - x)
            G._node[node]["goodness"] = x

        # Updating fairness of ratings
        for edge in edges:
            rating_distance = 1 - (abs(float(edge[2]["weight"]) - float(G._node[edge[1]]["goodness"])) / 2.0)
            user_fairness = G._node[edge[0]]["fairness"]
            ee = ("u"+edge[0], "p"+edge[1])
            kl_text = 1.0 - full_birdnest_edge[edge_map[ee]]

            x = (gamma2 * rating_distance + gamma1 * user_fairness + gamma3 * kl_text) / (gamma1 + gamma2 + gamma3)

            if x < 0.00:
                x = 0.0
            if x > 1.0:
                x = 1.0

            dr += abs(edge[2]["fairness"] - x)
            G[edge[0]][edge[1]]["fairness"] = x

        # Updating fairness of users
        currentfvals = []
        for node in nodes:
            if "A" not in node[0] and "#oc" not in node[0:3]:
                continue
            currentfvals.append(G._node[node]["fairness"])
        median_fvals = numpy.median(currentfvals)

        for node in nodes:
            if "A" not in node[0] and "#oc" not in node[0:3]:
                continue

            outedges = G.out_edges(node, data=True)
            f = 0
            rating_fairness = []
            for edge in outedges:
                rating_fairness.append(edge[2]["fairness"])

            for x in range(0, alpha1):
                rating_fairness.append(median_fvals)

            kl_timestamp = 1.0 - full_birdnest_user[user_map[node]]

            for x in range(0, alpha2):
                rating_fairness.append(kl_timestamp)

            mean_rating_fairness = numpy.mean(rating_fairness)

            x = mean_rating_fairness
            if x < 0.00:
                x = 0.0
            if x > 1.0:
                x = 1.0

            du += abs(G._node[node]["fairness"] - x)
            G._node[node]["fairness"] = x

        iter += 1
        if du < 0.01 and dp < 0.01 and dr < 0.01:
            break

    # SAVE THE RESULT
    currentfvals = []
    for node in nodes:
        if "A" not in node[0] and "#oc" not in node[0:3]: 
            continue
        currentfvals.append(G._node[node]["fairness"])
    median_fvals = numpy.median(currentfvals)

    all_node_vals = []
    for node in nodes:
        if "A" not in node[0] and "#oc" not in node[0:3]: 
            continue
        f = G._node[node]["fairness"]
        all_node_vals.append([node, (f - median_fvals) * numpy.log(G.out_degree(node) + 1), f, G.out_degree(node)])
    all_node_vals = numpy.array(all_node_vals)

    all_node_vals_sorted = sorted(all_node_vals, key=lambda x: (float(x[1]), float(x[2]), -1 * float(x[3])))[::-1]
    fw = open("../results/%s/%s-fng-sorted-users-%d-%d-%d-%d-%d-%d-%d.csv" % (NETWORKNAME, NETWORKNAME, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3), "w")

    goodusers = set()
    badusers = set()
    f = open("../data/%s_gt.csv" % NETWORKNAME, "r")

    for l in f:
        l = l.strip().split(",")
        if l[1] == "-1":
            badusers.add(l[0])
        else:
            goodusers.add(l[0])
    f.close()

    for i, sl in enumerate(all_node_vals_sorted):
        if sl[0] in badusers or sl[0] in goodusers:
            fw.write("%s,%s,%s,%s\n" % (str(sl[0]), str(sl[1]), str(sl[2]), str(sl[3])))

    fw.close()

def save_result_csv(df, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3):
    result_csv = f"../results/result-{alpha1}-{alpha2}-{beta1}-{beta2}-{gamma1}-{gamma2}-{gamma3}.csv"
    df.to_csv(result_csv, index=False)
    return result_csv


def main():
    st.title("Your Algorithm Streamlit App")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        st.write("File Uploaded Successfully!")
        st.write("Preview of the uploaded file:")
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        # Get algorithm parameters from the user
        st.sidebar.header("Algorithm Parameters")
        NETWORKNAME = "amazon"
        alpha1 = st.sidebar.slider("Alpha1", min_value=0, max_value=10, value=5)
        alpha2 = st.sidebar.slider("Alpha2", min_value=0, max_value=10, value=5)
        beta1 = st.sidebar.slider("Beta1", min_value=0, max_value=10, value=5)
        beta2 = st.sidebar.slider("Beta2", min_value=0, max_value=10, value=5)
        gamma1 = st.sidebar.slider("Gamma1", min_value=0, max_value=10, value=5)
        gamma2 = st.sidebar.slider("Gamma2", min_value=0, max_value=10, value=5)
        gamma3 = st.sidebar.slider("Gamma3", min_value=0, max_value=10, value=5)

        # # Run the algorithm
        # st.write("Running the algorithm...")
        # result_df = run_algorithm(df,NETWORKNAME, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3)
        # st.write("Algorithm completed!")

        # # Display the result
        # st.write("Preview of the result:")
        # st.write(result_df.head())

        # # Save result to CSV and provide download link
        # result_csv = result_df.to_csv(index=False)
        # st.download_button("Download Result CSV", data=result_csv, file_name="result.csv", key="result_csv")

        if st.sidebar.button("Run Algorithm"):
            st.write("Running the algorithm...")
            result_df = run_algorithm(df,NETWORKNAME, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3)
            st.write("Algorithm completed! 🎊🎉")

            # Display the result
            # st.write("Preview of the result:")
            # if result_df is not None:
            #     st.write(result_df.head())

            # Save result to CSV and provide download link
            # result_csv = save_result_csv(result_df, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3)
            # st.download_button("Download Result CSV", data=result_csv, file_name="result.csv", key="result_csv")

if __name__ == "__main__":
    main()
