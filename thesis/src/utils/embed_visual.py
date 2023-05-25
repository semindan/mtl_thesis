from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import umap
import numpy as np

# import pickle
import joblib
from matplotlib import pyplot as plt
from scipy import spatial
import copy
import igraph as ig

EMBEB_PATH = "/home/semindan/baka/thesis/src/embeddings/"


def visual_compare_points(points1, points2, tasks):
    for i in range(len(points1)):
        x = points1[i][0]
        y = points1[i][1]
        plt.plot(x, y, "bo", color="blue")
        plt.text(x * (1 + 0.001), y * (1 + 0.001), tasks[i], fontsize=12)

    for i in range(len(points2)):
        x = points2[i][0]
        y = points2[i][1]
        plt.plot(x, y, "bo", color="red")
        plt.text(x * (1 + 0.001), y * (1 + 0.001), tasks[i], fontsize=12)

    # plt.xlim((-0.1, 0.1))
    # plt.ylim((-0.1, 0.1))
    plt.show()


def visualize_points(points, tasks):
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]

        plt.plot(x, y, "bo")
        ax = plt.gca()
        ax.patch.set_alpha(0.5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # ax.set(yticklabels=[])  # remove the tick labels
        # ax.tick_params(left=False)
        ax.text(x * (1 + 0.01), y * (1 + 0.01), tasks[i], fontsize=12)
        # fig = plt.figure()
        # fig.savefig("tmp.png", transparent=True)
    plt.show()


def rrf(rankings):
    return np.sum(1 / (60 + rankings))


def rank(src_tasks, src_embs, dst_task, dst_emb):
    # get cosine similarities of src_tasks to dst_task
    distances = cosine(src_tasks, src_embs, dst_emb)

    argsorted = np.argsort(distances)
    ranks = np.arange(1, len(src_tasks) + 1)
    sorted_distances = distances[argsorted]

    sorted_tasks = np.array(src_tasks)[argsorted]

    return list(zip(sorted_tasks, ranks, sorted_distances))


def scores(components, src_tasks, src_dicts, dst_task, dst_dict):
    rankings = {}
    for component in components:
        src_vecs = [x[component] for x in src_dicts]

        dst_vec = dst_dict[component]
        res = rank(src_tasks, src_vecs, dst_task, dst_vec)
        for task, r, dist in res:
            rankings[task] = rankings.get(task, [])
            rankings[task].append(r)

    rrf_scores = {}
    for task in rankings:
        rrf_scores[task] = rrf(np.array(rankings[task]))
    rrf_scores = dict(
        sorted(rrf_scores.items(), key=lambda entry: entry[1], reverse=True)
    )
    for i, src_task in enumerate(rrf_scores):
        rrf_scores[src_task] = (rrf_scores[src_task], i + 1)
    return rrf_scores


def all_scores(components, tasks, vecs):
    scores_d = {}
    for i, task_dst in enumerate(tasks):
        tasks_src = tasks[:i] + tasks[i + 1 :]
        vecs_src = vecs[:i] + vecs[i + 1 :]
        scores_d[task_dst] = scores(components, tasks_src, vecs_src, task_dst, vecs[i])
    return scores_d


def cosine(src_tasks, src_vecs, dst_vec):
    distances = []
    for i, src in enumerate(src_vecs):
        distances.append(spatial.distance.cosine(src, dst_vec))
    return np.array(distances)


def cosine_rankings(tasks, vecs):
    distances = {}
    for i, src in enumerate(vecs):
        for j, dst in enumerate(vecs):
            if tasks[i] == tasks[j]:
                continue
            distances[tasks[i]] = distances.get(tasks[i], {})
            dist = spatial.distance.cosine(src, dst)
            distances[tasks[i]][tasks[j]] = dist
    distances = {
        task_name: sorted(distances[task_name].items(), key=lambda entry: entry[1])
        for task_name in tasks
    }
    return distances


def load(tasks, model, type):
    vecs = []
    for task_name in tasks:
        with open(
            EMBEB_PATH + model + "_" + task_name + type + ".pkl",
            "rb",
        ) as f:
            vecs.append(joblib.load(f))
    return vecs


def load_task_embeds(tasks, model):
    return load(tasks, model, "_taskemb")


def load_text_embeds(tasks, model):
    return load(tasks, model, "_textemb")


def avg_vecs(components, task_dict):
    for key in task_dict.keys():
        if len(task_dict[key].shape) == 2:
            entry = np.mean(task_dict[key], axis=0)
        else:
            entry = task_dict[key]
        task_dict[key] = entry
    return task_dict


def get_long_vecs(components, task_dict):
    vec = []
    num_layers = 0
    for key in components:
        if len(task_dict[key].shape) == 2:
            entry = np.mean(task_dict[key], axis=0)
        else:
            entry = task_dict[key]
        vec = np.hstack((vec, entry))
        num_layers += 1
    return vec / num_layers


def fruchterman_reingold(scores_dict):
    forces = {}
    for task_src in scores_dict:
        forces[task_src] = forces.get(task_src, {})
        for task_dst in scores_dict:
            if task_dst == task_src:
                continue
            force = (1 / scores_dict[task_dst][task_src][1]) + (
                1 / scores_dict[task_src][task_dst][1]
            )
            # if task_dst in forces and task_src in forces[task_dst]:
            #     continue
            forces[task_src][task_dst] = force
    return forces


def max_forces(forces):
    new_forces = {}
    for task_dst in forces:
        task_src = max(forces[task_dst], key=forces[task_dst].get)
        new_forces[task_src] = new_forces.get(task_src, {})
        new_forces[task_src][task_dst] = forces[task_dst][task_src]
        # new_forces.append((task_src, task_dst, forces[task_src][task_dst]))

    return new_forces


def full_pipeline(components, tasks, task_vecs):
    long_vecs = [get_long_vecs(components, task_dict) for task_dict in task_vecs]
    avges = [avg_vecs(components, task_dict) for task_dict in task_vecs]
    scores_res = all_scores(components, tasks, avges)
    cosine_ranks = cosine_rankings(tasks, long_vecs)
    # mapper = umap.UMAP(n_neighbors=2, metric = "cosine").fit(np.array(long_vecs))
    # visualize_points(mapper.embedding_, tasks)
    # pca = PCA(n_components=2)
    # pca.fit(np.array(long_vecs).T)
    # vecs_reduced = pca.components_.T
    # visualize_points(vecs_reduced, tasks)

    forces = fruchterman_reingold(scores_res)
    print(forces)
    new_forces = max_forces(forces)

    return scores_res, cosine_ranks, new_forces


def prepare_results_for_graph(tasks, forces, res_scores):
    vs = []
    es = []
    force = []
    weights = []
    for i, task in enumerate(tasks):
        vs.append(i)
        if task in forces:
            for task_dst in forces[task]:
                es.append([i, tasks.index(task_dst)])
                force.append(forces[task][task_dst])
                weights.append(res_scores[task_dst][task][0])

    return vs, es, force, np.round(weights, 4)


# %%
tasks = ["xnli", "paws-x", "nc", "qadsm", "wpr", "qam", "ctk"]
models = ["mt5", "xlm-r"]
# %%
# tasks = ["paws-x", "qam", "nc"]
text_vecs = load_text_embeds(tasks, "mt5")
task_vecs = load_task_embeds(tasks, "mt5")
# %%
task_text_vecs = copy.deepcopy(task_vecs)
for i in range(len(tasks)):
    task_text_vecs[i]["text"] = text_vecs[i]["avg_feature_vec"]

# %%
components = list(task_text_vecs[0].keys())
# components = list(filter(lambda x: "lm_head" not in x and "decoder" not in x, components))
components = list(filter(lambda x: "head" not in x, components))
# out = scores(components, tasks, task_vecs_flat, tasks[-1], task_vecs_flat[-1])

res_scores, cosine_ranks, forces = full_pipeline(components, tasks, task_text_vecs)
# %%
res_scores
# %%
vs, es, fs, weights = prepare_results_for_graph(tasks, forces, res_scores)
res_dict = {"vs": vs, "es": es, "fs": fs, "weights": weights, "res_scores": res_scores}
joblib.dump(res_dict, "xlm-r_task_text.pkl")

# %%
results = {}
import os

for filename in os.listdir():
    if filename.endswith(".pkl"):
        results[filename[:-4]] = joblib.load(filename)
if not os.path.exists("pdf_embeds"):
    os.mkdir("pdf_embeds")
# %%
layout = ig.Graph.layout_fruchterman_reingold
for filename in results:
    vs = results[filename]["vs"]
    es = results[filename]["es"]
    fs = results[filename]["fs"]
    weights = results[filename]["weights"]

    g = ig.Graph(
        edges=es,
        edge_attrs={"weight": weights, "name": tasks},
        directed=True,
    )
    visual_style = {
        "edge_width": 2.0,
        "vertex_size": 45,
        "palette": "heat",
        "layout": layout(g),
        "vertex_label": tasks,
        "margin": 30,
        "bbox": (300, 300),
        "vertex_color": "steelblue" if "mt5" in filename else "limegreen",
        "edge_background": "transparent",
        "vertex_label_size": 12,
        "edge_curved": 0.7
        # "vertex_order" : tasks
        # "vertex_label_dist" : 1.2
        # "vertex_label_angle" : 90
    }
    ig.plot(g, f"pdf_embeds/{filename}.pdf", **visual_style)
# ig.plot(g,  **visual_style)

# %%
fig, ax = plt.subplots(nrows=2, ncols=2)

for row in ax:
    for col in row:
        col.plot(g)
# %%
