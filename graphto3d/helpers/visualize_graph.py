from graphviz import Digraph
import os
from helpers import viz_util
import json


def visualize_scene_graph(graph, relationships, rel_filter_in = [], rel_filter_out = [], obj_ids = [], title ="", scan_id="",
													outfolder="./vis_graphs/"):
	g = Digraph(comment='Scene Graph' + title, format='png')

	for (i,obj) in enumerate(graph["objects"]):
		if (len(obj_ids) == 0) or (int(obj['global_id']) in obj_ids):
			g.node(str(obj['global_id']), obj["class"], fontname='helvetica', color=obj["color"], style='filled')
	edge_mask = None
	draw_edges(g, graph["relationships"], relationships, rel_filter_in, rel_filter_out, obj_ids, edge_mask)
	g.render(outfolder + scan_id)


def draw_edges(g, graph_relationships, relationships, rel_filter_in, rel_filter_out, obj_ids, edge_mask=None):
	edges = {}
	for (i, rel) in enumerate(graph_relationships):
		rel_text = relationships[rel[2]]
		if (len(rel_filter_in) == 0 or (rel_text.rstrip() in rel_filter_in)) and not rel_text.rstrip() in rel_filter_out:
			if (len(obj_ids) == 0) or ((rel[1] in obj_ids) and (rel[0] in obj_ids)):
				index = str(rel[0]) + "_" + str(rel[1])
				if index not in edges:
					edges[index] = []
				edges[index].append(rel[3])

	for (i,edge) in enumerate(edges):
		edge_obj_sub = edge.split("_")
		rels = ', '.join(edges[edge])

		g.edge(str(edge_obj_sub[0]), str(edge_obj_sub[1]), label=rels, color='grey')


def run(scan_id, data_path, outfolder):


	# use this option to read scene graphs from the dataset
	relationships_json = os.path.join(data_path, 'relationships_validation.json') #"relationships_train.json")
	objects_json = os.path.join(data_path, "objects.json")

	relationships = viz_util.read_relationships(os.path.join(data_path, "relationships.txt"))

	graph = viz_util.load_semantic_scene_graphs(relationships_json, objects_json)

	filter_dict_in = [] 
	filter_dict_out = []
	for scan_id in [scan_id]:
		visualize_scene_graph(graph[scan_id], relationships, filter_dict_in, filter_dict_out, [], "visualize_scene_graph", scan_id=scan_id,
													outfolder=outfolder)

	idx = [str(o['global_id']) for o in graph[scan_id]['objects']]
	color = [o['color'] for o in graph[scan_id]['objects']]
	# return used colors so that they can be used for 3D model visualization
	return dict(zip(idx, color))

