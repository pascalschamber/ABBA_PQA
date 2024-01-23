from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from bokeh.models import HoverTool, MultiLine, NodesAndLinkedEdges, Circle, EdgesAndLinkedNodes, Range1d
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.palettes import Spectral4, Inferno256, Spectral8, Spectral9, RdYlBu8, Spectral11
from bokeh.transform import linear_cmap
from bokeh.plotting import from_networkx
from bokeh.models import Legend, LegendItem, ColorBar ,LabelSet
from bokeh.io import export_svg
from matplotlib.lines import Line2D
from bokeh.models import (BoxZoomTool, HoverTool,
                          Plot, ResetTool, WheelZoomTool, PanTool)
from networkx.exception import NetworkXError

import math
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import operator
from datetime import datetime
today = datetime.now().strftime('%Y_%m%d')


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
from utilities.utils_data_management import AnimalsContainer
import utilities.utils_atlas_region_helper_functions as arhfs
import utilities.utils_general as ug
import utilities.utils_plotting as up
import utilities.utils_image_processing as u2

from quantification import graphing_2023_0603 as graphing
from quantification import PCA_engram_reactivation_pattern_0731 as pca_code







def custom_scale(corr_df):
    # scale a df from range -1 to 1, preserving number of negative correlations
    vals = corr_df.values.flatten()
    vmin, vmax = abs(vals[~np.isnan(vals)].min()), vals[~np.isnan(vals)].max()
    rescaled_vals = np.zeros_like(vals)
    for i_v, v in enumerate(vals):
        if pd.isnull(v):
            rescaled_vals[i_v] = np.nan
        elif v>0:
            rescaled_vals[i_v] = v/vmax
        else:
            rescaled_vals[i_v] = v/vmin
    reshaped = np.reshape(rescaled_vals, corr_df.values.shape)
    rescaled_df = pd.DataFrame(data=reshaped, index=corr_df.index, columns=corr_df.columns)

    return rescaled_df

def filter_regions_by_st_level(ont_slice):
    st_order_ids = [el[1] for el in arhfs.gather_ids_by_st_level(ont_slice, as_dict=False)]
    set_st_order = [] # need to convert to a set because of redundancies in ont_ids during iteration (each region appears by itself and under parent)
    for el in st_order_ids:
        if el not in set_st_order:
            set_st_order.append(el)
    st_order_names = arhfs.get_attributes_for_list_of_ids(ont_ids, set_st_order, 'name')
    return st_order_names

def check_parents(region_names_list, parent_level_colormap):
    parents = arhfs.get_st_parents(ont_ids, region_names_list, PARENT_ST_LEVEL)
    st_level_parents = set(list(parents.values()))
    assert all([p in parent_level_colormap for p in st_level_parents]), print(set(list(parents.values())))

def get_node_of_interest(G, node_of_interest, allow_fail=False):
    try:
        # Find neighbors of the node of interest
        neighbors = list(G.neighbors(node_of_interest))
        # Include the node of interest in the list
        nodes_to_include = neighbors + [node_of_interest]
        # Create subgraph
        subgraph = G.subgraph(nodes_to_include)
        return subgraph
    except NetworkXError as e:
        if allow_fail: 
            return None
        else: 
            raise KeyError(e)

def create_graph(df, significance_thresholds, drop_children=True):
    print(f' og df shape: {df.shape}')
    
    # filter out regions, # restrict to children of Basic cell groups and regions and st_level 8
    check_parents(df.columns.to_list(), parent_level_colormap)
    ont_slice = arhfs.parent_ontology_at_st_level(ont_ids, ST_LEVEL)
    st_order_names = arhfs.filter_regions_by_st_level(ont_ids, ont_slice)

    ordered_index = [el for el in st_order_names if el in df.columns.to_list()]
    df = df[ordered_index]
    new_parent_st_level_names = [d['name'] for d in ont_slice if d['name'] in df.columns.to_list()]
    print(f' filtered df shape: {df.shape}, num of parent regions: {len(new_parent_st_level_names)}')

    adj_reg_ids = [names_dict[el] for el in df.columns.to_list()]
    adj_st_levels = arhfs.get_attributes_for_list_of_ids(ont_ids, adj_reg_ids, 'st_level')
    adj_acronyms = arhfs.get_attributes_for_list_of_ids(ont_ids, adj_reg_ids, 'acronym')


    # Calculate the correlation matrix
    correlation_matrix = df.corr(method='pearson')
    p_values = df.corr(method=lambda x, y: pearsonr(x, y)[1]).fillna(1)

    # Apply a significance filter and a false discovery rate of 5%
    rejected, p_values_fdr_bh, _, _ = multipletests(p_values.values.flatten(), alpha=significance_thresholds['multi_ttest_p'], method='fdr_bh')
    p_values_fdr_bh = p_values_fdr_bh.reshape(correlation_matrix.shape)

    # Binarize the correlation matrix to an adjacency matrix
    mask = (abs(correlation_matrix) > significance_thresholds['corr_thresh']) & (p_values_fdr_bh < significance_thresholds['multi_ttest_p'])
    adjacency_matrix = correlation_matrix.where(mask).notnull().astype('int')
    print(f'adjacency_matrix shape: {adjacency_matrix.shape}, sum edges: {adjacency_matrix.values.flatten().sum()}')

    # Create new adjacency matrices where edges from children are drawn from parents
    child_to_parent_mapping = {k:v for k,v in arhfs.map_children2parent(ont_ids, ont_slice).items() if k in adjacency_matrix.index.values}
    print(f'children present in matrix: {sum([reg in child_to_parent_mapping for reg in adjacency_matrix.index.values])}')

    modified_matrix = adjacency_matrix.copy()
    if drop_children:
        children_to_drop = []
        for child, parent in child_to_parent_mapping.items():
            if child in modified_matrix.columns and parent in modified_matrix.columns:
                modified_matrix.loc[parent] = modified_matrix.loc[parent] | modified_matrix.loc[child]
                modified_matrix[parent] = modified_matrix[parent] | modified_matrix[child]
                children_to_drop.append(child)
        for child in children_to_drop:
            modified_matrix = modified_matrix.drop(child, axis=0).drop(child, axis=1)
    np.fill_diagonal(modified_matrix.values, 0)
    print(f'modified matrix shape: {modified_matrix.shape}, sum edges: {modified_matrix.values.flatten().sum()}, edges/node:{modified_matrix.values.flatten().sum()/modified_matrix.shape[0]}')


    ####################
    # Create the graph
    G = nx.from_pandas_adjacency(modified_matrix)
    # Remove isolated nodes, i.e. those without any edges
    G.remove_nodes_from(list(nx.isolates(G)))
    # Add correlation values as edge attributes
    for u, v, d in G.edges(data=True):
        d['correlation'] = correlation_matrix.loc[u, v]

    num_edges = sum([G.degree(n) for n in G.nodes])
    print(f'G nodes:{len(G.nodes)}, edges: {num_edges}, edges/node: {num_edges/len(G.nodes)}') 

    return G, modified_matrix, correlation_matrix






def visualize_dict(dict_hierarchy):
    # plot atlas ontology heirarchically, # e.g. dict_hierarchy == ont_ids[8]
    G = nx.DiGraph()
    
    root_name = dict_hierarchy['name']
    root_level = dict_hierarchy.get('st_level')
    G.add_node(root_name, st_level=root_level, subset=-1)
    
    root_children = dict_hierarchy.get('children', [])
    add_edges(G, root_name, root_children, root_level)

    # Specify color map for levels
    color_map = linear_cmap(field_name='st_level', palette=Spectral11, low=min(nx.get_node_attributes(G, 'st_level').values()), high=max(nx.get_node_attributes(G, 'st_level').values()))
    
    plot = Plot(plot_width=800, plot_height=900, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    plot.title.text = "Graph Visualization"

    # Add the hover tool
    node_hover_tool = HoverTool(tooltips=[("Name", "@index"), ("Level", "@st_level")])
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool(), WheelZoomTool(), PanTool())

    graph_renderer = from_networkx(G, nx.multipartite_layout, scale=1, center=(0, 0))

    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=color_map)
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=color_map)
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=color_map)

    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=.9, line_width=3)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=1)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=1)

    graph_renderer.selection_policy = EdgesAndLinkedNodes()
    plot.renderers.append(graph_renderer)

    output_notebook()
    show(plot)
    return G


def add_edges(graph, parent_node, children, parent_level=0):
    for child in children:
        child_name = child['name']
        child_level = child.get('st_level') # parent_level=0
        graph.add_node(child_name, st_level=child_level, subset=parent_level)
        graph.add_edge(parent_node, child_name)

        child_nodes = child.get('children', [])
        add_edges(graph, child_name, child_nodes, child_level)



##################################################################################################################################################################
def plot_network(G, correlation_matrix, fig_name, mode, graph_outpath, network_corr_cmap):
    if G is None: # added to allow failing of get_node_of_interest
        print('G is empty, nothing to plot'); return None
    
    assert mode in ['circular', 'spring'], f'cannot interpret mode:{mode}'

    degrees = [math.log(G.degree(n) + 1) * 10 for n in G.nodes]
    node_acronyms = {node:ont_ids[names_dict[node]]['acronym']  for i, node in enumerate(G.nodes)}

    ont_slice_colors = arhfs.parent_ontology_at_st_level(ont_ids, PARENT_ST_LEVEL)
    st_order_dict_colors = arhfs.gather_ids_by_st_level(ont_slice_colors, as_dict=True)
    child2parent_mapping_colors = {k:v for k,v in arhfs.map_children2parent(ont_ids, ont_slice_colors).items() if k in list(G.nodes)}    
    node_colors = {node: parent_level_colormap[child2parent_mapping_colors[node]] for i, node in enumerate(G.nodes)}


    # Create a plot
    ######################################################
    plot = figure(title=f"Brain network graph ({fig_name})", plot_width=800, plot_height=800)

    # Create a hover tool
    node_hover_tool = HoverTool(tooltips=[("index", "@index")])
    plot.add_tools(node_hover_tool)

    # Generate a colormap for the edges
    colormap = linear_cmap('correlation', palette=network_corr_cmap, low=-1, high=1)

    # Create the network graph
    if mode == 'circular':
        graph_renderer = from_networkx(G, nx.circular_layout,  scale=1, center=(0,0))
    elif mode == 'spring':
        graph_renderer = from_networkx(G, nx.spring_layout,  scale=1, center=(0,0))

    # Set the node size and color
    graph_renderer.node_renderer.glyph = Circle(size=30, fill_color='color')

    # Set the edge width and color
    graph_renderer.edge_renderer.glyph = MultiLine(line_color=colormap, line_alpha=0.8, line_width=1)

    # Add the edge correlation values to the graph renderer's data source
    graph_renderer.edge_renderer.data_source.data['correlation'] = [d['correlation'] for u, v, d in G.edges(data=True)]
    x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
    graph_renderer.node_renderer.data_source.data['x'] = x
    graph_renderer.node_renderer.data_source.data['y'] = y
    graph_renderer.node_renderer.data_source.data['degree'] = degrees
    graph_renderer.node_renderer.data_source.data['color'] = [node_colors[node] for node in G.nodes]
    graph_renderer.node_renderer.data_source.data['node_acronyms'] = [node_acronyms[node] for node in G.nodes]
    graph_renderer.node_renderer.data_source.data['node_parents'] = [child2parent_mapping_colors[node] for node in G.nodes]

    # configure the selection or inspection behavior of graphs
    graph_renderer.selection_policy = NodesAndLinkedEdges()

    # Add the network graph to the plot
    plot.renderers.append(graph_renderer)

    # Create a LabelSet
    labels = LabelSet(
        x='x', y='y', text='node_acronyms', level='glyph', 
        #   x_offset=-8, y_offset=-8, 
        y_offset=-4,
        text_align='center',
        text_baseline = 'center',
        text_font_size = '12px',
        source=graph_renderer.node_renderer.data_source, 
        render_mode='canvas'
    )
    plot.add_layout(labels) # Add the labels to the plot

    # # add a legend
    # leg_items = []
    # for leg_lbl_i, (node_parent, node_color) in enumerate(map_parent_2_color.items()):
    #     # get index of first occuring node for this label
    #     try:
    #         idx = graph_renderer.node_renderer.data_source.data['node_parents'].index(node_parent)
    #         leg_items.append(LegendItem(label=node_parent, renderers=[graph_renderer.node_renderer], index=idx))
    #     except ValueError:
    #         print(f'{node_parent} not found in graph');continue

    # legend = Legend(items=leg_items)
    # plot.add_layout(legend)

    # save the plot
    plot.output_backend = "svg"
    if SAVE_GRAPHS: export_svg(plot, filename=graph_outpath)
    # Show the plot
    output_notebook()
    show(plot)
##################################################################################################################################################################
def plot_legend():
    # plot legend, color of acronym to parent region
    sns.set_style('whitegrid', {"grid.color": ".0", "grid.linestyle": "",})
    plt.legend(
        handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markeredgecolor='k', markersize=12) for l,c in parent_level_colormap.items()], 
        labels=parent_level_colormap.keys() )
    if SAVE_GRAPHS:
        plt.savefig(os.path.join(graph_outdir, 'parent_region_color_legend.svg'), dpi=300)
    plt.show()

def plot_network_corr_cmap(bokeh_palette):
    # Create a linear color mapper that will be used for the network edges
    mapper = linear_cmap('correlation', palette=bokeh_palette, low=-1, high=1)
    # Create a figure
    p = figure(width=500, height=500, title="ColorBar")
    # Add a color bar to the figure
    color_bar = ColorBar(color_mapper=mapper['transform'], location=(0, 0))
    p.add_layout(color_bar, 'right')
    p.output_backend = "svg"
    if SAVE_GRAPHS: export_svg(p, filename=os.path.join(graph_outdir, 'network_corr_cmap_legend.svg'))
    output_notebook()
    show(p)

def map_region2parent_color(ont_ids, region_names, parent_stLvl=5, map_val='acronym', return_val='color'):
    ont_slice_colors = arhfs.parent_ontology_at_st_level(ont_ids, parent_stLvl)
    child2parent_mapping_colors = {k:v for k,v in arhfs.map_children2parent(ont_ids, ont_slice_colors).items() if k in list(region_names)}
    node_colors = {ont_ids[names_dict[name]][map_val]: parent_level_colormap[child2parent_mapping_colors[name]] for name in list(region_names)}
    return node_colors



def get_largest_connected_component(G):
    # print(f"component sizes: {[len(el) for el in nx.connected_components(G)]}")
    largest_connected_component = max(nx.connected_components(G), key=len)
    largest_component_graph = G.subgraph(largest_connected_component).copy()
    return largest_component_graph

def get_katz_centrality(graphs, groups, max_iter=100000, normalized=False, alpha=0.01, tol=1.0e-6, beta=1.0):    
    # would be better if alpha could be 0.05, in terms of relating it to the other graph in paper
    katz_centralities = [nx.katz_centrality(g, max_iter=max_iter, normalized=normalized, alpha=alpha, tol=tol, beta=beta) for g in graphs]

    katz_df = []
    for grp, katz_c in zip(groups, katz_centralities):
        for k,v in katz_c.items():
            katz_df.append({'group':grp, 'region_name':k, 'katz_centrality':v})
    return pd.DataFrame(katz_df)


def plot_katz_centrality(ont_ids, names_dict, katz_df, graph_outpath):    
    sns.set_style(
        'whitegrid', {"axes.facecolor": ".9", "grid.color": ".6", "grid.linestyle": "-", 'patch.edgecolor': 'none', 
        'figure.facecolor': 'black', 'text.color': 'white', 'axes.labelcolor': 'white','axes.facecolor': 'white','axes.edgecolor': 'white', 'xtick.color': 'white',
        'ytick.color': 'white',   
    })

    data = katz_df.sort_values(['group', 'katz_centrality'], ascending=[False, False])
    
    palette={'EXT':'blue', 'FC':'red', 'CTX':'gray'}
    fig,ax=plt.subplots(figsize=(25,8))
    ax.set_yscale('log')
    scatterp = sns.scatterplot(
        data=data, x='region_name', y='katz_centrality', hue='group', palette=palette, hue_order=palette.keys(), ax=ax, 
        alpha=0.7,
    )
    fig.canvas.draw_idle()
    # sns.barplot(data=data, x='region_name', y='katz_centrality', hue='group', palette=palette, ax=ax, dodge=bool(0), alpha=0.8)

    # convert tick labels to acronyms and apply color
    init_labels = [t.get_text() for t in ax.get_xticklabels()]
    node_colors = map_region2parent_color(ont_ids, init_labels, parent_stLvl=PARENT_ST_LEVEL)
    ax.set_xticks([i for i in range(len(init_labels))], [ont_ids[names_dict[el]]['acronym'] for el in init_labels], 
                  fontsize=4, rotation=90, ha='center')
    for tick in ax.xaxis.get_ticklabels():
        tick.set_color(node_colors[tick.get_text()])
    if SAVE_GRAPHS: fig.savefig(graph_outpath, dpi=300, bbox_inches='tight')
    plt.show()



def plot_degree_histogram(graphs, groups):
    node_df = []
    for i, g in enumerate(graphs):
        for n in g.nodes:
            node_df.append({'group':groups[i], 'degree':g.degree(n)})
    node_df = pd.DataFrame(node_df)
    palette={'EXT':'blue', 'FC':'red', 'CTX':'gray'}
    sns.histplot(data=node_df, x='degree', hue='group', palette=palette, multiple="dodge", shrink=.8, hue_order=palette.keys())


def plot_GC_network_stats(graphs, groups, graph_outpath):
    # plot misc graph metrics such as nodes,edges, edges/node, efficiency

    def plot_n_vals(ax, vals, groups, var_name, palette):
        # format data as df, and plot fc vs ext
        plot_vals = [v if isinstance(v, list) else [v] for v in vals]
        df_rows = []
        for vs, grp in zip(plot_vals, groups):
            df_rows.extend([{'group':grp, var_name:v} for v in vs])
        pltdf = pd.DataFrame(df_rows)
        sns.barplot(ax=ax, data=pltdf, x='group', order=palette.keys(), y=var_name, hue='group', palette=palette, dodge=bool(0), errorbar='se')
        ax.get_legend().remove()
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(var_name)
    # graphs, groups = [Gext, Gfc, Gctx], ['EXT', 'FC', 'CTX']
    # vals = [nx.global_efficiency(g) for g in graphs]
    # var_name = 'Global efficiency'

    palette={'EXT':'blue', 'FC':'red', 'CTX':'gray'}
    
    fig, axs = plt.subplots(1,5, figsize=(8,4))
    plot_n_vals(axs[0], [nx.global_efficiency(g) for g in graphs], groups, 'Global efficiency', palette)
    plot_n_vals(axs[1], [nx.global_efficiency(get_largest_connected_component(g)) for g in graphs], groups, 'GC efficiency', palette)
    plot_n_vals(axs[2], [len(get_largest_connected_component(g).nodes) for g in graphs], groups, 'Nodes in GC', palette)
    plot_n_vals(axs[3], [[v for k,v in dict(g.degree()).items()] for g in graphs], groups, r'Edges/Node', palette)
    plot_n_vals(axs[4], [nx.density(g) for g in graphs], groups, 'Network density', palette) # nx.density(get_largest_connected_component(Gfc)), 
    plt.tight_layout()
    if SAVE_GRAPHS: fig.savefig(graph_outpath, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_resiliency(G, mode):
    Gc = G.copy()

    # List to hold the percentage of nodes in the largest connected component
    pcts = []

    # While there are nodes in the graph
    while Gc.number_of_nodes() > 0:
        if mode == 'pGC':
            # Calculate the percentage of nodes in the largest connected component
            largest_cc = get_largest_connected_component(Gc)
            pct = len(largest_cc)/len(Gc.nodes)
        elif mode == 'pEff':
            pct = nx.global_efficiency(Gc)
        else: 
            raise ValueError(mode)
        
        pcts.append(pct)

        # Remove the node with the highest degree
        node_degrees = dict(Gc.degree())
        highest_degree_node = max(node_degrees.items(), key=operator.itemgetter(1))[0]
        Gc.remove_node(highest_degree_node)
    return pcts

def plot_network_resiliency(graphs, groups, graph_outpath, mode):
    pcts_all = [calculate_resiliency(g, mode) for g in graphs]
    # pcts_fc, pcts_ext = calculate_resiliency(Gfc, mode), calculate_resiliency(Gext, mode)
    palette={'EXT':'blue', 'FC':'red', 'CTX':'gray'}
    plt.figure(figsize=(10, 6))
    for pcts_grp, grp in zip(pcts_all, groups):
        sns.lineplot(x=range(len(pcts_grp)), y=pcts_grp, color=palette[grp])

    plt.xscale('log')
    plt.xlabel('Number of Nodes')
    if mode == 'pGC':
        plt.ylabel('% GC Size')
    elif mode == 'pEff':
        plt.ylabel('% Efficiency')
    plt.title('Network Resilience')
    if SAVE_GRAPHS: plt.savefig(graph_outpath, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_corr_counts(G, group_name):
    edge_df = {}
    for e in G.edges.data():
        fr, to, d = e
        corr = d['correlation']
        for areg in [fr, to]:
            if areg not in edge_df:
                edge_df[areg] = {'nPos':0, 'nNeg':0}
            if corr > 0:
                edge_df[areg]['nPos'] += 1
            else:
                edge_df[areg]['nNeg'] += 1
    edge_df2 = []
    for k,v in edge_df.items():
        edge_df2.append({'group':group_name, 'region_name':k, 'nPos':v['nPos'], 'nNeg':-v['nNeg'], 'nTot':v['nPos']+v['nNeg']})
    return pd.DataFrame(edge_df2)
    
def get_corr_counts(graphs, groups):
    corr_counts = pd.concat([calculate_corr_counts(g, grp) for g,grp in zip(graphs, groups)], ignore_index=True).assign(
        diff=0, diff_fcctx=0, diff_extctx=0)
    
    # get region order by difference between fc and ext
    corr_counts_region_order = {}
    for areg in corr_counts['region_name'].unique():
        vals = corr_counts.loc[corr_counts['region_name']==areg]
        val_fc = vals.loc[vals['group']=='FC']['nTot']
        val_ext = vals.loc[vals['group']=='EXT']['nTot']
        val_ctx = vals.loc[vals['group']=='CTX']['nTot']
        val_fc = 0 if pd.isnull(val_fc).all() else val_fc.values[0]
        val_ext = 0 if pd.isnull(val_ext).all() else val_ext.values[0]
        val_ctx = 0 if pd.isnull(val_ctx).all() else val_ctx.values[0]
        corr_counts_region_order[areg] = abs(val_fc-val_ext)
        reg_index = corr_counts[corr_counts['region_name']==areg].index
        corr_counts.loc[reg_index, 'diff'] = val_fc-val_ext
        corr_counts.loc[reg_index, 'diff_fcctx'] = val_fc-val_ctx
        corr_counts.loc[reg_index, 'diff_extctx'] = val_ext-val_ctx
    corr_counts_region_order = [el[0] for el in sorted(corr_counts_region_order.items(), key=lambda x: x[1])[::-1]]
    return corr_counts, corr_counts_region_order


def plot_corr_counts(corr_counts, corr_counts_region_order, graph_outpath, ont_ids, names_dict, fig_height=10):
    data = corr_counts.assign(
        acronym = corr_counts['region_name'].apply(lambda x: ont_ids[names_dict[x]]['acronym']),
        tick_color = corr_counts['region_name'].apply(lambda x: list(map_region2parent_color(ont_ids, [x], parent_stLvl=PARENT_ST_LEVEL).values())[0])
    )
    data = data.melt(id_vars=['group', 'region_name', 'nTot', 'diff', 'acronym', 'tick_color'], 
                     value_name='n', value_vars=['nPos', 'nNeg'])
    # data = data.sort_values(['tick_color', 'nTot'], ascending=[True, False]).replace(0, np.nan)
    data = data.sort_values(['diff','nTot'], ascending=[False,False]).replace(0, np.nan)
    y_label_order = []
    for el in data['region_name']:
        if el not in y_label_order:
            y_label_order.append(el)


    # Make the PairGrid
    ##################################################################
    sns.set_style(
        'whitegrid', {"axes.facecolor": ".9", "grid.color": ".6", "grid.linestyle": "-", 'patch.edgecolor': 'none', 
        'figure.facecolor': 'black', 'text.color': 'white', 'axes.labelcolor': 'white','axes.facecolor': 'white','axes.edgecolor': 'white', 'xtick.color': 'white',
        'ytick.color': 'white',   
    })
    palette={'EXT':'blue', 'FC':'red', 'CTX':'gray'}

    # Make the PairGrid
    g = sns.PairGrid(data,
                    x_vars=['n'], y_vars=["region_name"], hue='group', palette=palette,
                    height=fig_height, aspect=.25)

    # Draw a dot plot using the stripplot function
    g.map(sns.stripplot, size=5, orient="h", jitter=False,
        palette=palette, linewidth=1, edgecolor=None, alpha=0.7)

    # Use the same x axis limits on all columns and add better labels
    g.set(xlim=(data['n'].min(), data['n'].max()), xlabel="count", ylabel="")

    # Use semantically meaningful titles for the columns
    for ax, title in zip(g.axes.flat, ["count"]):
        ax.set(title=title)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.axvline(0, c='k')
    sns.despine(left=True, bottom=True)


    # replace axis labels with acronyms and color
    g.axes[0][0].tick_params(labeltop=True, labelright=True)
    for ax in g.axes[0]:
        # ax = g.axes[0][0]
        node_colors = map_region2parent_color(ont_ids, y_label_order, parent_stLvl=PARENT_ST_LEVEL)
        ax.set_yticks([i for i in range(len(y_label_order))], [ont_ids[names_dict[el]]['acronym'] for el in y_label_order], 
                        fontsize=4, rotation=0, ha='center')
        for tick in ax.get_yticklabels():
            tick.set_color(node_colors[tick.get_text()])

    if SAVE_GRAPHS: g.savefig(graph_outpath, dpi=300, bbox_inches='tight')
    plt.show()

    return data

def get_neighbors_within_2(graph, node):
    """Get neighbors of a node within 2 hops."""
    one_hop = set(graph.neighbors(node))
    two_hop = set()
    for neighbor in one_hop:
        two_hop |= set(graph.neighbors(neighbor))
    # Include the node itself and exclude it from the neighbors
    two_hop |= {node}
    two_hop -= {node}
    return one_hop | two_hop

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

def compare_graphs(graph1, graph2):
    """Compare nodes of two graphs based on their 2-hop neighbors."""
    # If Node 1 in both graphs has a similarity score of 1.0, meaning their 2-hop neighborhoods are identical.
    # if Nodes 2 and 3 have a similarity score of 0.5, indicating that only half of their 2-hop neighbors are the same between the two graphs.
    nodes1 = set(graph1.nodes())
    nodes2 = set(graph2.nodes())
    
    # Check for nodes that are in both graphs
    common_nodes = nodes1 & nodes2
    
    similarities = {}
    for node in common_nodes:
        neighbors_g1 = get_neighbors_within_2(graph1, node)
        neighbors_g2 = get_neighbors_within_2(graph2, node)
        similarities[node] = jaccard_similarity(neighbors_g1, neighbors_g2)

    similarities_sorted = sorted(similarities.items(), key=lambda x: x[1], reverse=True)     
    similarities = {k[0]:similarities[k[0]] for k in similarities_sorted}
    return similarities

def plot_similarities(similarities, def_title='', graph_outpath=None):
    """Plot the similarity scores for nodes."""
    nodes = list(similarities.keys())
    scores = list(similarities.values())
    node_colors = map_region2parent_color(ont_ids, nodes, parent_stLvl=PARENT_ST_LEVEL)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(def_title)
    ax.bar(nodes, scores, color='lightblue')
    ax.set_xlabel('Nodes')
    ax.set_ylabel('Similarity Score')
    ax.set_title('Similarity Scores for Nodes based on 2-hop Neighbors')
    ax.set_ylim(0, 1)  # since Jaccard similarity is between 0 and 1
    # for i, v in enumerate(scores):
    #     ax.text(i, v + 0.02, "{:.2f}".format(v), ha='center', va='bottom', fontweight='bold')
    ax.set_xticks([i + 0.5 for i in range(len(nodes))])
    ax.set_xticklabels([ont_ids[names_dict[el]]['acronym'] for el in nodes], fontsize='xx-small', rotation=90)

    for tick in ax.xaxis.get_ticklabels():
        tick.set_color(node_colors[tick.get_text()])
    
    if SAVE_GRAPHS: fig.savefig(graph_outpath, dpi=300, bbox_inches='tight')
    plt.show()



def custom_cmap(cmap_str):
    from matplotlib.colors import LinearSegmentedColormap
    # Obtain the original 'seismic' colormap
    original_cmap = plt.cm.seismic

    # Define the new custom colormap
    colors = [(1, 1, 1), (0, 0, 0), (1, 1, 1)]  # R -> G -> B
    n_bins = [0, 0.5, 1]  # Define bins
    cmap_name = "custom_div_cmap"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

    # Modify the new colormap to blend with the original 'seismic' colormap
    newcolors = np.vstack((original_cmap(np.linspace(0, 0.5, 128)), np.array([0,0,0,1]), original_cmap(np.linspace(0.5, 1, 128))))
    black_midpoint_cmap = LinearSegmentedColormap.from_list("BlackMidpointSeismic", newcolors)

    return black_midpoint_cmap

def custom_cmap2():
    from matplotlib.colors import LinearSegmentedColormap
    original_cmap = plt.cm.seismic
    cmap_compress_range = int((1-0.85)*256)
    compressed_colors_upper = original_cmap(np.linspace(0.5, 1, cmap_compress_range))
    compressed_colors_lower = original_cmap(np.linspace(0.0, 0.5, cmap_compress_range))
    middle_colors = np.tile([1, 1, 1, 1], (256-(compressed_colors_upper.shape[0] + compressed_colors_lower.shape[0]), 1))
    # Create a new colormap where colors from [0, 0.85] are white and [0.85, 1] follows the compressed 'seismic' colormap
    newcolors = np.vstack([compressed_colors_lower, middle_colors, compressed_colors_upper])
    compressed_seismic_cmap = LinearSegmentedColormap.from_list("CompressedSeismic", newcolors)

    return compressed_seismic_cmap

 

def plot_correlation_matrix(corr, ont_ids, names_dict, graph_outpath, cmap='seismic', center=0, vmin=None, vmax=None, fillna=False, mask=False, CORR_TICKS_ALL_SIDES=False, SHOW=True):
    # sns.set_style({'axes.facecolor': 'black'})
    data = corr.transpose().fillna(0) if fillna else corr.transpose()
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask else None
    
    figw, figh = max(int(corr.shape[0]/7.66), 10), max(int(corr.shape[0]/9.58), 8)
    fig, ax = plt.subplots(figsize=(figw, figh))
    hm = sns.heatmap(
        data, annot=False, cmap=cmap, ax=ax, center=center, mask=mask, square=True,
        linewidths=1.0, linecolor='#d9d9d9', robust=True, vmin=vmin, vmax=vmax,
    )
    assert (corr.index.values == corr.columns.values).all()

    node_colors = map_region2parent_color(ont_ids, corr.index, parent_stLvl=PARENT_ST_LEVEL)
    ax = format_axis_correlation_matrix(ax, corr, ont_ids, names_dict, node_colors, CORR_TICKS_ALL_SIDES)
    
    if SAVE_GRAPHS: plt.savefig(graph_outpath, bbox_inches='tight', dpi=300)
    if SHOW: plt.show()
    else: plt.close()


def format_axis_correlation_matrix(ax, corr, ont_ids, names_dict, node_colors, CORR_TICKS_ALL_SIDES):
    if CORR_TICKS_ALL_SIDES:
        ax.tick_params(labeltop=True, labelright=True)
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=0)

    ax.set_yticks([i + 0.5 for i in range(len(corr.index))])
    ax.set_yticklabels([ont_ids[names_dict[el]]['acronym'] for el in corr.index.values], fontsize='xx-small')
    ax.set_xticks([i + 0.5 for i in range(len(corr.index))])
    ax.set_xticklabels([ont_ids[names_dict[el]]['acronym'] for el in corr.index.values], fontsize='xx-small')

    for tick in ax.xaxis.get_ticklabels():
        tick.set_color(node_colors[tick.get_text()])
    for tick in ax.yaxis.get_ticklabels():
        tick.set_color(node_colors[tick.get_text()])
    
    return ax 


def calculate_difference_corr_matrix(correlation_matrix_fc, modified_matrix_fc, correlation_matrix_ext, modified_matrix_ext):
    # take the difference, rescaling first to range 0,2
    corr_fc_scaled, corr_ext_scaled = custom_scale(correlation_matrix_fc.reindex_like(modified_matrix_fc)),  custom_scale(correlation_matrix_ext.reindex_like(modified_matrix_ext))
    print(f"correlation ranges. FC: ({correlation_matrix_fc.min().min()}, {correlation_matrix_fc.max().max()}), EXT: ({correlation_matrix_ext.min().min()}, {correlation_matrix_ext.max().max()})")
    print(f"num og corr. above/below 0: {(correlation_matrix_fc.values.flatten()>0).sum()}, {(correlation_matrix_fc.values.flatten()<0).sum()}")
    print(f"num scaled corr. above/below 0: {(corr_fc_scaled.values.flatten()>0).sum()}, {(corr_fc_scaled.values.flatten()<0).sum()}")
    print(f"old sum > 0: {(correlation_matrix_fc[correlation_matrix_fc<0]).sum().sum()}, <0: {(correlation_matrix_fc[correlation_matrix_fc>0]).sum().sum()}, matrix_size: {correlation_matrix_fc.values.size}")
    print(f"new sum > 0: {(corr_fc_scaled[corr_fc_scaled<0]).sum().sum()}, <0: {(corr_fc_scaled[corr_fc_scaled>0]).sum().sum()}, matrix_size: {corr_fc_scaled.values.size}")

    # # force dimensions to match
    # corr_ext_scaled = corr_ext_scaled.reindex_like(corr_fc_scaled)
    # modified_matrix_ext = modified_matrix_ext.reindex_like(modified_matrix_fc)

    corr_diff = (corr_fc_scaled+1) - (corr_ext_scaled+1)
    # corr_diff_rescaled = custom_scale(corr_diff)
    comb_adj = pd.DataFrame(((modified_matrix_fc>0).values | (modified_matrix_ext>0).values) > 0, index=modified_matrix_fc.index, columns=modified_matrix_fc.columns)

    return corr_diff.dropna(axis=0, how='all').dropna(axis=1, how='all'), comb_adj.dropna(axis=0, how='all').dropna(axis=1, how='all')


if __name__ == '__main__':
    ##################################################################################################################################################################
    # ont visualization
    # ont_ids = arhfs.load_ontology()
    # ont = arhfs.load_raw_ontology()[0]
    # Gont = visualize_dict(ont)

    ##################################################################################################################################################################
    ont_ids = arhfs.load_ontology()
    names_dict = dict(zip([d['name'] for d in ont_ids.values()], ont_ids.keys()))


    ##################################################################################################################################################################
    SAVE_GRAPHS = bool(0)
    SAVE_CORRS = bool(0)
    VALUE = ['zif-density', 'reactivation', 'gfp_density'][0]
    ST_LEVEL = 8
    PARENT_ST_LEVEL = 5
    DROP_CHILDREN = bool(0)
    parent_level_colormap = {
        'Cortical subplate': '#3288bd', 'Isocortex': '#66c2a5', 'Olfactory areas': '#abdda4', 'Pallidum': '#e6f598', 'Hippocampal formation': '#ffffbf', 'Striatum': '#fee08b', 'Thalamus': '#fdae61', 'Hypothalamus': '#f46d43', 'Midbrain': '#d53e4f',
        'Pons':'#980043', 'Medulla':'#756bb1', 'Cerebellar cortex':'#d9d9d9', 'Cerebellar nuclei':'#969696',
    }
    network_corr_cmap = RdYlBu8
    graph_outdir = r'C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\quantification\network_analysis\figures_' + VALUE.replace('-', '_')
    if SAVE_GRAPHS: graph_outdir = graphing.verify_output_dir(os.path.join(graph_outdir, f'{today}_st{ST_LEVEL}'))



    # Load the data - index: animal_ints, cols: brain regions
    data_dir = r"C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\quantification\network_analysis\data"
    # df_fc = pd.read_excel(os.path.join(data_dir, "2023_0806_zif-density_fc.xlsx"), index_col=0)
    # df_ext = pd.read_excel(os.path.join(data_dir, "2023_0806_zif-density_ext.xlsx"), index_col=0)
    # df_fc = pd.read_excel(os.path.join(data_dir, "2023_0807_zif_density_st8_fc.xlsx"), index_col=0)
    # df_ext = pd.read_excel(os.path.join(data_dir, "2023_0807_zif_density_st8_ext.xlsx"), index_col=0)
    if VALUE == 'zif-density':
        df_fc = pd.read_excel(os.path.join(data_dir, "2023_0829_zif_density_st8_fc.xlsx"), index_col=0)
        df_ext = pd.read_excel(os.path.join(data_dir, "2023_0829_zif_density_st8_ext.xlsx"), index_col=0)
        df_ctx = pd.read_excel(os.path.join(data_dir, "2023_0829_zif_density_st8_ctx.xlsx"), index_col=0)
    elif VALUE == 'reactivation':
        df_fc = pd.read_excel(os.path.join(data_dir, "2023_0829_reactivation_st8_fc.xlsx"), index_col=0)
        df_ext = pd.read_excel(os.path.join(data_dir, "2023_0829_reactivation_st8_ext.xlsx"), index_col=0)
        df_ctx = pd.read_excel(os.path.join(data_dir, "2023_0829_reactivation_st8_ctx.xlsx"), index_col=0)
    elif VALUE == 'gfp_density':
        df_fc = pd.read_excel(os.path.join(data_dir, "2023_0830_gfp_density_st8_fc.xlsx"), index_col=0)
        df_ext = pd.read_excel(os.path.join(data_dir, "2023_0830_gfp_density_st8_ext.xlsx"), index_col=0)
        df_ctx = pd.read_excel(os.path.join(data_dir, "2023_0830_gfp_density_st8_ctx.xlsx"), index_col=0)
    else: 
        raise ValueError(VALUE)
    
    # get atlas region centroids for each region in the df columns
    




    for c_threshold in [0.8, 0.85, 0.9][:1]:
        ##########################################################
        # set significance thresholds
        significance_thresholds = dict(
            multi_ttest_p = 0.05,
            corr_thresh = c_threshold,
        )
        param_str = '_'.join([f'{k}-{v}' for k,v in significance_thresholds.items()])

        ##########################################################
        # create network graphs
        Gfc, modified_matrix_fc, correlation_matrix_fc = create_graph(df_fc, significance_thresholds, drop_children=DROP_CHILDREN)
        Gext, modified_matrix_ext, correlation_matrix_ext = create_graph(df_ext, significance_thresholds, drop_children=DROP_CHILDREN)
        Gctx, modified_matrix_ctx, correlation_matrix_ctx = create_graph(df_ctx, significance_thresholds, drop_children=DROP_CHILDREN)

        
        ##########################################################
        # network plots
        plot_legend()
        plot_network_corr_cmap(network_corr_cmap)
        sns.set_style(
            'darkgrid', {"grid.color": ".0", "grid.linestyle": "", 'patch.edgecolor': 'none',
        })
        # plot small world characteristics
        plot_degree_histogram([Gext, Gfc, Gctx], ['EXT', 'FC', 'CTX'])
        plot_GC_network_stats([Gext, Gfc, Gctx], ['EXT', 'FC', 'CTX'], os.path.join(graph_outdir, f'{today}_{param_str}_GC_network_stats.svg'))

        # plot network diagrams
        for MODE in ['spring', 'circular']:    
            plot_network(Gfc, correlation_matrix_fc, f'FC {param_str}', MODE, os.path.join(graph_outdir, f'{today}_{param_str}_{MODE}_FC.svg'), network_corr_cmap)
            plot_network(Gext, correlation_matrix_ext, f'EXT {param_str}', MODE, os.path.join(graph_outdir, f'{today}_{param_str}_{MODE}_EXT.svg'), network_corr_cmap)
            plot_network(Gctx, correlation_matrix_ctx, f'CTX {param_str}', MODE, os.path.join(graph_outdir, f'{today}_{param_str}_{MODE}_CTX.svg'), network_corr_cmap)

        
        # zoom in on a node of interest, plotting its direct connections
        NoI = 'Basolateral amygdalar nucleus' if VALUE == 'zif-density' else 'Basolateral amygdalar nucleus, anterior part' if VALUE=='reactivation' else 'Basolateral amygdalar nucleus'
        plot_network(get_node_of_interest(Gfc, NoI, allow_fail=True), correlation_matrix_fc, f'FC {param_str}', 'spring', os.path.join(graph_outdir, f'{today}_{param_str}_spring_{NoI}_FC.svg'), network_corr_cmap)
        plot_network(get_node_of_interest(Gext, NoI, allow_fail=True), correlation_matrix_ext, f'EXT {param_str}', 'spring', os.path.join(graph_outdir, f'{today}_{param_str}_spring_{NoI}_EXT.svg'), network_corr_cmap)
        plot_network(get_node_of_interest(Gctx, NoI, allow_fail=True), correlation_matrix_ctx, f'CTX {param_str}', 'spring', os.path.join(graph_outdir, f'{today}_{param_str}_spring_{NoI}_CTX.svg'), network_corr_cmap)

        # plot katz centrality
        print('max alphas for each graph:', [1/max(nx.adjacency_spectrum(g)) for g in [Gext, Gfc, Gctx]])
        alpha = 0.035 if VALUE=='reactivation' else 0.042 if VALUE=='zif-density' else 0.01
        katz_df = get_katz_centrality(
            [get_largest_connected_component(Gext), get_largest_connected_component(Gfc), get_largest_connected_component(Gctx)], ['EXT', 'FC', 'CTX'], 
            max_iter=100000, normalized=True, alpha=alpha)    
        plot_katz_centrality(ont_ids, names_dict, katz_df, os.path.join(graph_outdir, f'{today}_{param_str}_katz_centrality.svg'))

        # plot resiliency as either % of GC or % efficiency
        plot_network_resiliency([Gext, Gfc, Gctx], ['EXT', 'FC', 'CTX'], os.path.join(graph_outdir, f'{today}_{param_str}_network_resiliency_pGC.svg'), 'pGC')
        plot_network_resiliency([Gext, Gfc, Gctx], ['EXT', 'FC', 'CTX'], os.path.join(graph_outdir, f'{today}_{param_str}_network_resiliency_pEff.svg'), 'pEff')
        
        # plot count of number of correlations per region (FC vs EXT)
        corr_counts, corr_counts_region_order = get_corr_counts([Gext, Gfc, Gctx], ['EXT', 'FC', 'CTX'])
        plot_corr_counts(corr_counts, corr_counts_region_order, os.path.join(graph_outdir, f'{today}_{param_str}_corrCountsPairgrid.svg'), ont_ids, names_dict, fig_height=20)
        
        

        similarities_fcext = compare_graphs(Gext, Gfc)
        plot_similarities(similarities_fcext, def_title='FC vs EXT', graph_outpath=os.path.join(graph_outdir,f'{today}_{param_str}_JaccardSimilarity_FCvsEXT.svg'))
        similarities_fcctx = compare_graphs(Gctx, Gfc)
        plot_similarities(similarities_fcctx, def_title='FC vs CTX', graph_outpath=os.path.join(graph_outdir,f'{today}_{param_str}_JaccardSimilarity_FCvsCTX.svg'))

        ##########################################################
        # plot correlation matricies
        corr_args = dict(fillna=True, mask=bool(0), CORR_TICKS_ALL_SIDES=bool(1), center=0.0, vmin=-1.0, vmax=1.0, SHOW=bool(1), cmap=custom_cmap2())
        corr_args_diff = dict(vmin=-2.0, vmax=2.0, cmap='seismic')
        corr_args_diff = {**corr_args, **corr_args_diff}
        corr_args_diff_masked = dict(cmap='seismic')
        corr_args_diff_masked = {**corr_args_diff, **corr_args_diff_masked}


        corr_fc = correlation_matrix_fc.reindex_like(modified_matrix_fc)[modified_matrix_fc>0].dropna(axis=0, how='all').dropna(axis=1, how='all')
        corr_ext = correlation_matrix_ext.reindex_like(modified_matrix_ext)[modified_matrix_ext>0].dropna(axis=0, how='all').dropna(axis=1, how='all')
        corr_ctx = correlation_matrix_ctx.reindex_like(modified_matrix_ctx)[modified_matrix_ctx>0].dropna(axis=0, how='all').dropna(axis=1, how='all')
        plot_correlation_matrix(corr_fc, ont_ids, names_dict, os.path.join(graph_outdir,f'{today}_{param_str}_corr_fc.svg'), **corr_args)
        plot_correlation_matrix(corr_ext, ont_ids, names_dict, os.path.join(graph_outdir,f'{today}_{param_str}_corr_ext.svg'), **corr_args) 
        plot_correlation_matrix(corr_ctx, ont_ids, names_dict, os.path.join(graph_outdir,f'{today}_{param_str}_corr_ctx.svg'), **corr_args)

        # take the difference between fc and ext
        corr_diff, comb_adj = calculate_difference_corr_matrix(correlation_matrix_fc, modified_matrix_fc, correlation_matrix_ext, modified_matrix_ext)
        plot_correlation_matrix(corr_diff, ont_ids, names_dict, os.path.join(graph_outdir,f'{today}_{param_str}_corr_diff_not-masked.svg'), **corr_args_diff)
        plot_correlation_matrix(corr_diff[comb_adj], ont_ids, names_dict, os.path.join(graph_outdir,f'{today}_{param_str}_corr_diff_masked.svg'), **corr_args_diff_masked) 
        # take the difference between fc and ctx
        corr_diff_ctx, comb_adj_ctx = calculate_difference_corr_matrix(correlation_matrix_fc, modified_matrix_fc, correlation_matrix_ctx, modified_matrix_ctx)
        plot_correlation_matrix(corr_diff_ctx, ont_ids, names_dict, os.path.join(graph_outdir,f'{today}_{param_str}_corr_diff_ctx_not-masked.svg'), **corr_args_diff)
        plot_correlation_matrix(corr_diff_ctx[comb_adj_ctx], ont_ids, names_dict, os.path.join(graph_outdir,f'{today}_{param_str}_corr_diff_ctx_masked.svg'), **corr_args_diff_masked)
        
        if SAVE_CORRS:
            correlation_matrix_fc.to_csv(os.path.join(graph_outdir,f'{today}_{param_str}_corr_fc_not-masked.csv'))
            correlation_matrix_ext.to_csv(os.path.join(graph_outdir,f'{today}_{param_str}_corr_ext_not-masked.csv'))
            correlation_matrix_ctx.to_csv(os.path.join(graph_outdir,f'{today}_{param_str}_corr_ctx_not-masked.csv'))
            corr_fc.to_csv(os.path.join(graph_outdir,f'{today}_{param_str}_corr_fc.csv'))
            corr_ext.to_csv(os.path.join(graph_outdir,f'{today}_{param_str}_corr_ext.csv'))
            corr_ctx.to_csv(os.path.join(graph_outdir,f'{today}_{param_str}_corr_ctx.csv'))

            corr_diff.to_csv(os.path.join(graph_outdir,f'{today}_{param_str}_corr_diff_not-masked.csv'))
            corr_diff[comb_adj].to_csv(os.path.join(graph_outdir,f'{today}_{param_str}_corr_diff_masked.csv'))
            corr_diff_ctx.to_csv(os.path.join(graph_outdir,f'{today}_{param_str}_corr_diff_ctx_not-masked.csv'))
            corr_diff_ctx[comb_adj_ctx].to_csv(os.path.join(graph_outdir,f'{today}_{param_str}_corr_diff_ctx_masked.csv'))

        

        from sklearn.decomposition import PCA
        import numpy as np

        def do_pca_corr(corr_df, num_components=2):
            # Assuming you've already calculated the correlation_matrix
            X = corr_df.fillna(0).values

            pca = PCA(n_components=num_components)  # We need 2 principal components for a 2D plot
            X_pca = pca.fit_transform(X)
            return pca, X_pca
        
        # Convert cartesian to polar coordinates
        def cartesian_to_polar(x, y):
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            return r, theta

        PLOT_SCREE = bool(0)
        num_components = 25
        pca_indicies = 2 if VALUE=='zif-density' else None
        corr_dfs = [corr_fc, corr_ext, corr_ctx][:pca_indicies]
        pcas = [do_pca_corr(corr_df, num_components=num_components) for corr_df in corr_dfs]
        pca_labels = ['FC', 'EXT', 'CTX']
        
        plt.rcParams["scatter.edgecolors"] = 'k'
        for pca_i, pca in enumerate(pcas[:]):
            '''
                note on what the coordinates mean by quadrant
                -pca0/+pca1  |  +pca0/+pca1
                _____________|______________
                             |
                -pca0/-pca1  |  +pca0/-pca1
            '''

            pca_obj, pca_X = pcas[pca_i]
            if PLOT_SCREE: pca_code.scree_plots(pca_obj)

            og_df = corr_dfs[pca_i]
            pca_label_group = pca_labels[pca_i]

            # Convert PCA results to polar coordinates
            radii, angles = cartesian_to_polar(pca_X[:, 0], pca_X[:, 1])
            palette = map_region2parent_color(ont_ids, og_df.index.to_list(), parent_stLvl=PARENT_ST_LEVEL, map_val='name')
            colors = [palette[lbl] for lbl in og_df.index.to_list()]

            # Plot
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(15,15))
            ax.set_facecolor('#d9d9d9')
            ax.set_title(pca_label_group, fontsize='xx-large')

            # ax.scatter(angles, radii, c=colors)

            # Annotate data points
            for point_i, (label, x, y) in enumerate(zip(og_df.columns, angles, radii)):
                og_ont = ont_ids[names_dict[label]]
                acro = og_ont['acronym']
                y_spacer = y + 0.10
                ax.annotate(acro, xy=(x, y), fontsize=4, c='k', ha='center', va='center', xytext=(x,y_spacer))
                
                pca_val0, pca_val1 = pca_X[point_i, 0], pca_X[point_i, 1]
                ax.plot([0, x], [0, y], color=palette[label], alpha=0.7, zorder=0)
            
            plt.savefig(os.path.join(graph_outdir,f'{today}_{param_str}_PCA_polar_{pca_label_group}.svg'), bbox_inches='tight', dpi=300)
            plt.show()

        
        
        

        # correlate freezing behavior with zif activation
        ##################################################
        if bool(0):
            
            from scipy.stats import spearmanr, pearsonr
            ac = AnimalsContainer()
            ac.init_animals()
            freeze_df = pd.read_excel(r"C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\quantification\network_analysis\data\2023_0808_freezing-data_cohort2-3.xlsx")
            freeze_df = freeze_df.assign(animal_int=freeze_df['animal_id'].apply(lambda x: int(x[3:])))
            ret_df = freeze_df.loc[freeze_df['trial'] == 'RET1']

            sns.barplot(data=ret_df, x='group', y='freeze_avg', hue='cohort')
            sns.swarmplot(data=ret_df, x='group', y='freeze_avg', hue='cohort',dodge=True, color='k')
            plt.show()

            ret_fc = ret_df.loc[ret_df['group'] == 'FC']

            
            df_paths = [os.path.join(r'D:\ReijmersLab\TEL\slides\quant_data\counts', fn) for fn in [
                '2023_0806_quant_data_cohort2_t-75-350-0.45-550-100-5-85-4-31.csv',
                '2023_0806_quant_data_cohort3_t-75-350-0.45-360-100-5-85-4-31.csv',]]
            
            andf = pca_code.load_by_animal_reactivation_data(df_paths, ac)
            df = pca_code.PCA_format(andf, val_col='reactivation')
            
            # filter regions
            # get regions with highest degree for FC
            corr_counts_fc = corr_counts.loc[(corr_counts['group'] == 'FC') ].sort_values('diff', ascending=False)
            fc_regions = [el for el in corr_counts_fc['region_name'].values if el in df.columns.to_list()]

            # filter freeze animal ids
            ret_ids = [el for el in ret_df['animal_int'].values if el in df.index]
            df = df.loc[ret_ids,:].sort_index()
            ret_df = ret_df.set_index('animal_int').loc[ret_ids,:].sort_index()

            freeze_vals = ret_df['freeze_avg'].values 
            region_freeze_corr = []
            for col in df.columns.to_list():
                df_vals = df[col].values 
                correlation_sp, pval_sp = spearmanr(df_vals, freeze_vals)
                correlation_pear, pval_pear = pearsonr(df_vals, freeze_vals)
                region_freeze_corr.append({
                    'region_name':col, 'corr_sp':correlation_sp, 'corr_pear':correlation_pear, 'pval_sp':pval_sp, 'pval_pear':pval_pear
                })
            region_freeze_corr_df = pd.DataFrame(region_freeze_corr)

            PVAL_THRESH = 0.05
            CORR_THRESH = 0.60
            sig_corr_df = region_freeze_corr_df[
                ((region_freeze_corr_df['pval_sp']<PVAL_THRESH) | (region_freeze_corr_df['pval_pear']<PVAL_THRESH)) &
                ((region_freeze_corr_df['corr_sp'].abs()>CORR_THRESH) | (region_freeze_corr_df['corr_pear'].abs()>CORR_THRESH))
            ]
            print(f'number of significant correlations: {len(sig_corr_df)}')

            fig,ax = plt.subplots(figsize=(8,6))
            sns.barplot(
                data = region_freeze_corr_df[(region_freeze_corr_df['pval_pear']<PVAL_THRESH) & (region_freeze_corr_df['corr_pear'].abs()>CORR_THRESH)],
                x='region_name', y='corr_pear', ax=ax,
            )
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), ha='left', rotation=-45)
            plt.show()



    







