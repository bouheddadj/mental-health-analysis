"""
Dashboard interactif pour l'analyse Mental Health & Social Media Balance
Utilise Dash et Plotly pour des visualisations interactives
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import community.community_louvain as community_louvain

# Configuration de l'application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Mental Health & Social Media Analysis"

# Chargement des données
df = pd.read_csv('mental_health_social_media.csv')
df_clean = df.dropna()

# Colonnes numériques
numerical_cols = ['Age', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)',
                  'Stress_Level(1-10)', 'Days_Without_Social_Media',
                  'Exercise_Frequency(week)', 'Happiness_Index(1-10)']

# Normalisation
scaler = StandardScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df_clean[numerical_cols]),
    columns=numerical_cols
)

# Clustering initial
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(df_normalized)

# PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(df_normalized)
df_clean['PCA1'] = pca_results[:, 0]
df_clean['PCA2'] = pca_results[:, 1]

# Layout du dashboard
app.layout = dbc.Container([
    # En-tête
    dbc.Row([
        dbc.Col([
            html.H1("📊 Mental Health & Social Media Balance",
                    className="text-center mb-4 mt-4",
                    style={'color': '#2c3e50', 'font-weight': 'bold'}),
            html.H4("Analyse de données et visualisations interactives",
                    className="text-center mb-4",
                    style={'color': '#7f8c8d'}),
            html.Hr()
        ], width=12)
    ]),

    # Statistiques clés
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("👥 Utilisateurs", className="card-title"),
                    html.H2(len(df_clean), className="text-primary"),
                    html.P(f"Âge moyen: {df_clean['Age'].mean():.1f} ans")
                ])
            ], className="mb-4 shadow")
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("⏱️ Temps d'écran", className="card-title"),
                    html.H2(f"{df_clean['Daily_Screen_Time(hrs)'].mean():.1f}h",
                            className="text-info"),
                    html.P("Moyenne quotidienne")
                ])
            ], className="mb-4 shadow")
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("😊 Indice bonheur", className="card-title"),
                    html.H2(f"{df_clean['Happiness_Index(1-10)'].mean():.1f}/10",
                            className="text-success"),
                    html.P(f"Stress moyen: {df_clean['Stress_Level(1-10)'].mean():.1f}/10")
                ])
            ], className="mb-4 shadow")
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("💪 Exercice", className="card-title"),
                    html.H2(f"{df_clean['Exercise_Frequency(week)'].mean():.1f}x",
                            className="text-warning"),
                    html.P("Par semaine")
                ])
            ], className="mb-4 shadow")
        ], width=3),
    ]),

    # Onglets principaux
    dbc.Tabs([
        # ONGLET 1: Exploration
        dbc.Tab(label="📈 Exploration des données", tab_id="tab-1", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Distribution des variables", className="mt-4 mb-3"),
                    dcc.Dropdown(
                        id='var-dropdown',
                        options=[{'label': col, 'value': col} for col in numerical_cols],
                        value='Happiness_Index(1-10)',
                        className="mb-3"
                    ),
                    dcc.Graph(id='distribution-plot')
                ], width=6),

                dbc.Col([
                    html.H4("Statistiques descriptives", className="mt-4 mb-3"),
                    html.Div(id='stats-table')
                ], width=6)
            ]),

            dbc.Row([
                dbc.Col([
                    html.H4("Matrice de corrélation (Pearson)", className="mt-4 mb-3"),
                    dcc.Graph(id='correlation-heatmap')
                ], width=12)
            ])
        ]),

        # ONGLET 2: Relations entre variables
        dbc.Tab(label="🔗 Relations entre variables", tab_id="tab-2", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Analyse de scatter plots", className="mt-4 mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Variable X:"),
                            dcc.Dropdown(
                                id='scatter-x',
                                options=[{'label': col, 'value': col} for col in numerical_cols],
                                value='Daily_Screen_Time(hrs)'
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Variable Y:"),
                            dcc.Dropdown(
                                id='scatter-y',
                                options=[{'label': col, 'value': col} for col in numerical_cols],
                                value='Happiness_Index(1-10)'
                            )
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Couleur par:", className="mt-3"),
                            dcc.Dropdown(
                                id='scatter-color',
                                options=[
                                    {'label': 'Genre', 'value': 'Gender'},
                                    {'label': 'Plateforme', 'value': 'Social_Media_Platform'},
                                    {'label': 'Cluster', 'value': 'Cluster'}
                                ],
                                value='Gender'
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Taille par:", className="mt-3"),
                            dcc.Dropdown(
                                id='scatter-size',
                                options=[{'label': col, 'value': col} for col in numerical_cols],
                                value='Age'
                            )
                        ], width=6)
                    ]),
                    dcc.Graph(id='scatter-plot', className="mt-3")
                ], width=12)
            ])
        ]),

        # ONGLET 3: Clustering
        dbc.Tab(label="🎯 Clustering", tab_id="tab-3", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Paramètres du clustering", className="mt-4 mb-3"),
                    html.Label("Nombre de clusters:"),
                    dcc.Slider(
                        id='n-clusters-slider',
                        min=2,
                        max=8,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in range(2, 9)}
                    ),
                    html.Div(id='silhouette-score', className="mt-3 mb-3"),
                    html.H5("Courbe d'Elbow (Inertie)", className="mt-4"),
                    dcc.Graph(id='elbow-curve')
                ], width=4),

                dbc.Col([
                    html.H4("Visualisation des clusters (PCA)", className="mt-4 mb-3"),
                    dcc.Graph(id='cluster-pca-plot'),
                    html.H4("Distribution par cluster", className="mt-4 mb-3"),
                    dcc.Graph(id='cluster-distribution')
                ], width=8)
            ]),

            dbc.Row([
                dbc.Col([
                    html.H4("Profils des clusters", className="mt-4 mb-3"),
                    html.Div(id='cluster-profiles')
                ], width=12)
            ])
        ]),

        # ONGLET 4: Réduction dimensionnelle
        dbc.Tab(label="🗺️ Réduction dimensionnelle", tab_id="tab-4", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("PCA - Analyse en Composantes Principales", className="mt-4 mb-3"),
                    dcc.Graph(id='pca-plot'),
                    html.Div([
                        html.P(f"Variance expliquée PC1: {pca.explained_variance_ratio_[0]*100:.2f}%"),
                        html.P(f"Variance expliquée PC2: {pca.explained_variance_ratio_[1]*100:.2f}%"),
                        html.P(f"Variance totale: {sum(pca.explained_variance_ratio_)*100:.2f}%")
                    ])
                ], width=6),

                dbc.Col([
                    html.H4("t-SNE - Visualisation non-linéaire", className="mt-4 mb-3"),
                    html.Label("Perplexité:"),
                    dcc.Slider(
                        id='perplexity-slider',
                        min=5,
                        max=50,
                        step=5,
                        value=30,
                        marks={i: str(i) for i in range(5, 51, 10)}
                    ),
                    dcc.Graph(id='tsne-plot')
                ], width=6)
            ]),

            dbc.Row([
                dbc.Col([
                    html.H4("Contribution des variables aux composantes", className="mt-4 mb-3"),
                    dcc.Graph(id='pca-loadings')
                ], width=12)
            ])
        ]),

        # ONGLET 5: Analyse de réseau
        dbc.Tab(label="🕸️ Analyse de réseau", tab_id="tab-5", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Réseau de similarité entre utilisateurs", className="mt-4 mb-3"),
                    html.Label("Seuil de similarité:"),
                    dcc.Slider(
                        id='similarity-threshold',
                        min=0.5,
                        max=0.95,
                        step=0.05,
                        value=0.7,
                        marks={i/100: f'{i/100:.2f}' for i in range(50, 100, 10)}
                    ),
                    html.Div(id='network-stats', className="mt-3 mb-3")
                ], width=3),

                dbc.Col([
                    dcc.Graph(id='network-plot')
                ], width=9)
            ]),

            dbc.Row([
                dbc.Col([
                    html.H4("Distribution des degrés", className="mt-4 mb-3"),
                    dcc.Graph(id='degree-distribution')
                ], width=6),

                dbc.Col([
                    html.H4("Communautés détectées", className="mt-4 mb-3"),
                    dcc.Graph(id='community-plot')
                ], width=6)
            ])
        ]),

        # ONGLET 6: Insights
        dbc.Tab(label="💡 Insights & Recommandations", tab_id="tab-6", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Principaux enseignements", className="mt-4 mb-3"),

                    dbc.Card([
                        dbc.CardHeader("🔍 Corrélations importantes"),
                        dbc.CardBody(id='key-correlations')
                    ], className="mb-3 shadow"),

                    dbc.Card([
                        dbc.CardHeader("👥 Profils d'utilisateurs identifiés"),
                        dbc.CardBody(id='user-profiles')
                    ], className="mb-3 shadow"),

                    dbc.Card([
                        dbc.CardHeader("📱 Impact des réseaux sociaux"),
                        dbc.CardBody([
                            dcc.Graph(id='platform-analysis')
                        ])
                    ], className="mb-3 shadow"),

                    dbc.Card([
                        dbc.CardHeader("💪 Recommandations"),
                        dbc.CardBody(id='recommendations')
                    ], className="shadow")

                ], width=12)
            ])
        ])
    ], id="tabs", active_tab="tab-1")

], fluid=True)


# ==================== CALLBACKS ====================

# Distribution plot
@app.callback(
    Output('distribution-plot', 'figure'),
    Input('var-dropdown', 'value')
)
def update_distribution(selected_var):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Histogramme', 'Box Plot'))

    fig.add_trace(
        go.Histogram(x=df_clean[selected_var], name='Distribution',
                     marker_color='#3498db'),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(y=df_clean[selected_var], name='Box Plot',
               marker_color='#e74c3c'),
        row=1, col=2
    )

    fig.update_layout(height=400, showlegend=False, title_text=f"Distribution: {selected_var}")
    return fig


# Stats table
@app.callback(
    Output('stats-table', 'children'),
    Input('var-dropdown', 'value')
)
def update_stats(selected_var):
    stats = df_clean[selected_var].describe()

    table = dbc.Table([
        html.Thead([
            html.Tr([html.Th("Statistique"), html.Th("Valeur")])
        ]),
        html.Tbody([
            html.Tr([html.Td("Moyenne"), html.Td(f"{stats['mean']:.2f}")]),
            html.Tr([html.Td("Médiane"), html.Td(f"{stats['50%']:.2f}")]),
            html.Tr([html.Td("Écart-type"), html.Td(f"{stats['std']:.2f}")]),
            html.Tr([html.Td("Min"), html.Td(f"{stats['min']:.2f}")]),
            html.Tr([html.Td("Max"), html.Td(f"{stats['max']:.2f}")]),
            html.Tr([html.Td("Q1"), html.Td(f"{stats['25%']:.2f}")]),
            html.Tr([html.Td("Q3"), html.Td(f"{stats['75%']:.2f}")])
        ])
    ], bordered=True, hover=True, striped=True)

    return table


# Correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('tabs', 'active_tab')
)
def update_correlation_heatmap(active_tab):
    if active_tab == 'tab-1':
        corr = df_clean[numerical_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Corrélation")
        ))

        fig.update_layout(
            title="Matrice de corrélation de Pearson",
            height=600,
            xaxis={'tickangle': -45}
        )

        return fig
    return {}


# Scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-x', 'value'),
     Input('scatter-y', 'value'),
     Input('scatter-color', 'value'),
     Input('scatter-size', 'value')]
)
def update_scatter(x_var, y_var, color_var, size_var):
    fig = px.scatter(
        df_clean,
        x=x_var,
        y=y_var,
        color=color_var,
        size=size_var,
        hover_data=['User_ID', 'Age', 'Gender', 'Social_Media_Platform'],
        title=f"{y_var} vs {x_var}",
        template='plotly_white'
    )

    fig.update_layout(height=600)
    return fig


# Clustering callbacks
@app.callback(
    [Output('cluster-pca-plot', 'figure'),
     Output('silhouette-score', 'children'),
     Output('cluster-distribution', 'figure'),
     Output('cluster-profiles', 'children')],
    Input('n-clusters-slider', 'value')
)
def update_clustering(n_clusters):
    # Appliquer K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_normalized)

    # Score de silhouette
    sil_score = silhouette_score(df_normalized, clusters)

    # Créer df temporaire
    df_temp = df_clean.copy()
    df_temp['Cluster'] = clusters

    # Plot PCA avec clusters
    fig_pca = px.scatter(
        df_temp,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        hover_data=['User_ID', 'Age', 'Happiness_Index(1-10)'],
        title=f"Visualisation PCA avec {n_clusters} clusters",
        color_continuous_scale='Viridis',
        template='plotly_white'
    )
    fig_pca.update_layout(height=500)

    # Distribution par cluster
    cluster_counts = df_temp['Cluster'].value_counts().sort_index()
    fig_dist = go.Figure(data=[
        go.Bar(x=cluster_counts.index, y=cluster_counts.values,
               marker_color='#3498db')
    ])
    fig_dist.update_layout(
        title="Nombre d'utilisateurs par cluster",
        xaxis_title="Cluster",
        yaxis_title="Nombre d'utilisateurs",
        height=400
    )

    # Profils des clusters
    profiles = []
    for cluster_id in range(n_clusters):
        cluster_data = df_temp[df_temp['Cluster'] == cluster_id]

        profile_card = dbc.Card([
            dbc.CardHeader(f"Cluster {cluster_id} - {len(cluster_data)} utilisateurs"),
            dbc.CardBody([
                html.P(f"🕐 Temps d'écran: {cluster_data['Daily_Screen_Time(hrs)'].mean():.1f}h/jour"),
                html.P(f"😴 Qualité sommeil: {cluster_data['Sleep_Quality(1-10)'].mean():.1f}/10"),
                html.P(f"😰 Niveau stress: {cluster_data['Stress_Level(1-10)'].mean():.1f}/10"),
                html.P(f"😊 Indice bonheur: {cluster_data['Happiness_Index(1-10)'].mean():.1f}/10"),
                html.P(f"💪 Exercice: {cluster_data['Exercise_Frequency(week)'].mean():.1f} fois/semaine"),
                html.P(f"👥 Âge moyen: {cluster_data['Age'].mean():.1f} ans")
            ])
        ], className="mb-3")

        profiles.append(profile_card)

    sil_display = dbc.Alert(
        f"Score de Silhouette: {sil_score:.3f}",
        color="success" if sil_score > 0.5 else "warning"
    )

    return fig_pca, sil_display, fig_dist, profiles


# Elbow curve
@app.callback(
    Output('elbow-curve', 'figure'),
    Input('tabs', 'active_tab')
)
def update_elbow(active_tab):
    if active_tab == 'tab-3':
        inertias = []
        silhouettes = []
        K_range = range(2, 9)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(df_normalized)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(df_normalized, labels))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Méthode du coude (Inertie)', 'Score de Silhouette')
        )

        fig.add_trace(
            go.Scatter(x=list(K_range), y=inertias, mode='lines+markers',
                       marker=dict(size=10, color='#e74c3c'),
                       line=dict(width=2)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=list(K_range), y=silhouettes, mode='lines+markers',
                       marker=dict(size=10, color='#2ecc71'),
                       line=dict(width=2)),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Nombre de clusters", row=1, col=1)
        fig.update_xaxes(title_text="Nombre de clusters", row=1, col=2)
        fig.update_yaxes(title_text="Inertie", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=2)

        fig.update_layout(height=400, showlegend=False)
        return fig
    return {}


# PCA plot
@app.callback(
    Output('pca-plot', 'figure'),
    Input('tabs', 'active_tab')
)
def update_pca_plot(active_tab):
    if active_tab == 'tab-4':
        fig = px.scatter(
            df_clean,
            x='PCA1',
            y='PCA2',
            color='Happiness_Index(1-10)',
            size='Daily_Screen_Time(hrs)',
            hover_data=['User_ID', 'Age', 'Gender'],
            title="PCA - Projection en 2D",
            color_continuous_scale='Viridis',
            template='plotly_white'
        )
        fig.update_layout(height=500)
        return fig
    return {}


# t-SNE plot
@app.callback(
    Output('tsne-plot', 'figure'),
    Input('perplexity-slider', 'value')
)
def update_tsne(perplexity):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(df_normalized)

    df_temp = df_clean.copy()
    df_temp['TSNE1'] = tsne_results[:, 0]
    df_temp['TSNE2'] = tsne_results[:, 1]

    fig = px.scatter(
        df_temp,
        x='TSNE1',
        y='TSNE2',
        color='Cluster',
        hover_data=['User_ID', 'Age', 'Happiness_Index(1-10)'],
        title=f"t-SNE (perplexité={perplexity})",
        template='plotly_white'
    )
    fig.update_layout(height=500)
    return fig


# PCA loadings
@app.callback(
    Output('pca-loadings', 'figure'),
    Input('tabs', 'active_tab')
)
def update_pca_loadings(active_tab):
    if active_tab == 'tab-4':
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=numerical_cols
        )

        fig = go.Figure()

        for var in loadings.index:
            fig.add_trace(go.Scatter(
                x=[0, loadings.loc[var, 'PC1']],
                y=[0, loadings.loc[var, 'PC2']],
                mode='lines+markers+text',
                name=var,
                text=['', var],
                textposition='top center',
                line=dict(width=2),
                marker=dict(size=[0, 10])
            ))

        fig.update_layout(
            title="Contribution des variables aux composantes principales",
            xaxis_title="PC1",
            yaxis_title="PC2",
            height=500,
            template='plotly_white'
        )

        return fig
    return {}


# Network analysis
@app.callback(
    [Output('network-plot', 'figure'),
     Output('network-stats', 'children'),
     Output('degree-distribution', 'figure'),
     Output('community-plot', 'figure')],
    Input('similarity-threshold', 'value')
)
def update_network(threshold):
    # Créer le réseau de similarité
    similarity_matrix = cosine_similarity(df_normalized)
    similarity_matrix[similarity_matrix < threshold] = 0
    np.fill_diagonal(similarity_matrix, 0)

    G = nx.from_numpy_array(similarity_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_nodes() == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Aucun réseau à afficher (seuil trop élevé)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return empty_fig, "Pas de réseau", empty_fig, empty_fig

    # Statistiques du réseau
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)

    stats_display = dbc.Card([
        dbc.CardBody([
            html.H5("Statistiques du réseau", className="card-title"),
            html.P(f"Nœuds: {n_nodes}"),
            html.P(f"Arêtes: {n_edges}"),
            html.P(f"Densité: {density:.4f}"),
        ])
    ])

    # Layout du réseau
    pos = nx.spring_layout(G, seed=42)

    # Edges
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Nodes
    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Degré',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_degrees = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        degree = G.degree(node)
        node_degrees.append(degree)
        node_text.append(f"Nœud {node}<br>Degré: {degree}")

    node_trace.marker.color = node_degrees
    node_trace.text = node_text

    # Figure du réseau
    fig_network = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title=f"Réseau de similarité (seuil={threshold})",
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0, l=0, r=0, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=600
                            ))

    # Distribution des degrés
    degrees = [d for n, d in G.degree()]
    fig_degree = go.Figure(data=[
        go.Histogram(x=degrees, marker_color='#3498db', nbinsx=20)
    ])
    fig_degree.update_layout(
        title="Distribution des degrés",
        xaxis_title="Degré",
        yaxis_title="Fréquence",
        height=400
    )

    # Détection de communautés
    try:
        communities = community_louvain.best_partition(G)
        community_list = list(communities.values())

        node_trace_comm = node_trace.copy()
        node_trace_comm.marker.color = community_list
        node_trace_comm.marker.colorbar.title = 'Communauté'

        fig_community = go.Figure(data=[edge_trace, node_trace_comm],
                                  layout=go.Layout(
                                      title=f"Communautés détectées (n={len(set(community_list))})",
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=0, l=0, r=0, t=40),
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      height=400
                                  ))
    except:
        fig_community = go.Figure()
        fig_community.add_annotation(
            text="Impossible de détecter des communautés",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    return fig_network, stats_display, fig_degree, fig_community


# Insights - Key correlations
@app.callback(
    Output('key-correlations', 'children'),
    Input('tabs', 'active_tab')
)
def update_key_correlations(active_tab):
    if active_tab == 'tab-6':
        corr = df_clean[numerical_cols].corr()

        insights = []

        # Trouver les corrélations fortes
        strong_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.3:
                    strong_corr.append((
                        corr.columns[i],
                        corr.columns[j],
                        corr.iloc[i, j]
                    ))

        strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)

        for var1, var2, r in strong_corr[:5]:
            if r > 0:
                text = f"✅ {var1} et {var2} sont positivement corrélés (r={r:.3f})"
            else:
                text = f"⚠️ {var1} et {var2} sont négativement corrélés (r={r:.3f})"
            insights.append(html.P(text))

        return insights
    return []


# User profiles
@app.callback(
    Output('user-profiles', 'children'),
    Input('tabs', 'active_tab')
)
def update_user_profiles(active_tab):
    if active_tab == 'tab-6':
        profiles = []

        for cluster_id in sorted(df_clean['Cluster'].unique()):
            cluster_data = df_clean[df_clean['Cluster'] == cluster_id]

            avg_screen = cluster_data['Daily_Screen_Time(hrs)'].mean()
            avg_happiness = cluster_data['Happiness_Index(1-10)'].mean()
            avg_stress = cluster_data['Stress_Level(1-10)'].mean()

            # Profil textuel
            if avg_screen > 6:
                screen_cat = "fort usage"
                screen_emoji = "🔴"
            elif avg_screen > 4:
                screen_cat = "usage modéré"
                screen_emoji = "🟡"
            else:
                screen_cat = "usage limité"
                screen_emoji = "🟢"

            if avg_happiness > 7:
                happiness_cat = "heureux"
            elif avg_happiness > 5:
                happiness_cat = "moyennement heureux"
            else:
                happiness_cat = "peu heureux"

            profile_text = html.Div([
                html.H6(f"{screen_emoji} Profil {cluster_id} ({len(cluster_data)} utilisateurs)"),
                html.P(f"Caractéristiques: {screen_cat} des réseaux sociaux, {happiness_cat}"),
                html.Hr()
            ])

            profiles.append(profile_text)

        return profiles
    return []


# Platform analysis
@app.callback(
    Output('platform-analysis', 'figure'),
    Input('tabs', 'active_tab')
)
def update_platform_analysis(active_tab):
    if active_tab == 'tab-6':
        platform_stats = df_clean.groupby('Social_Media_Platform').agg({
            'Happiness_Index(1-10)': 'mean',
            'Stress_Level(1-10)': 'mean',
            'Daily_Screen_Time(hrs)': 'mean'
        }).reset_index()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Bonheur moyen',
            x=platform_stats['Social_Media_Platform'],
            y=platform_stats['Happiness_Index(1-10)'],
            marker_color='#2ecc71'
        ))

        fig.add_trace(go.Bar(
            name='Stress moyen',
            x=platform_stats['Social_Media_Platform'],
            y=platform_stats['Stress_Level(1-10)'],
            marker_color='#e74c3c'
        ))

        fig.update_layout(
            title="Impact des plateformes sur le bien-être",
            xaxis_title="Plateforme",
            yaxis_title="Score moyen",
            barmode='group',
            height=400
        )

        return fig
    return {}


# Recommendations
@app.callback(
    Output('recommendations', 'children'),
    Input('tabs', 'active_tab')
)
def update_recommendations(active_tab):
    if active_tab == 'tab-6':
        # Analyser les données pour générer des recommandations
        high_screen = df_clean[df_clean['Daily_Screen_Time(hrs)'] > 6]
        low_happiness = df_clean[df_clean['Happiness_Index(1-10)'] < 5]
        high_stress = df_clean[df_clean['Stress_Level(1-10)'] > 7]

        recs = []

        if len(high_screen) > len(df_clean) * 0.3:
            recs.append(html.Li(
                f"⏱️ {len(high_screen)} utilisateurs ({len(high_screen)/len(df_clean)*100:.1f}%) "
                "ont un temps d'écran élevé (>6h). Recommandation: Encourager des pauses régulières."
            ))

        if len(low_happiness) > 0:
            recs.append(html.Li(
                f"😔 {len(low_happiness)} utilisateurs ont un faible indice de bonheur. "
                "Recommandation: Promouvoir l'exercice physique et la qualité du sommeil."
            ))

        if len(high_stress) > 0:
            recs.append(html.Li(
                f"😰 {len(high_stress)} utilisateurs ont un niveau de stress élevé. "
                "Recommandation: Augmenter les jours sans réseaux sociaux."
            ))

        # Corrélation exercice-bonheur
        corr_exercise_happiness = df_clean['Exercise_Frequency(week)'].corr(
            df_clean['Happiness_Index(1-10)']
        )
        if corr_exercise_happiness > 0.3:
            recs.append(html.Li(
                f"💪 L'exercice est positivement corrélé au bonheur (r={corr_exercise_happiness:.2f}). "
                "Recommandation: Promouvoir l'activité physique régulière."
            ))

        return html.Ul(recs)
    return []


# Lancer l'application
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
