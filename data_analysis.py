"""
Analyse complète du dataset Mental Health & Social Media Balance
Application des techniques des TPs: Exploration, Clustering, Réduction dimensionnelle, Réseaux
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import community.community_louvain as community_louvain
import warnings
warnings.filterwarnings('ignore')


class MentalHealthAnalysis:
    """Classe pour l'analyse complète du dataset Mental Health"""

    def __init__(self, filepath):
        """Initialisation et chargement des données"""
        self.df = pd.read_csv(filepath)
        self.df_clean = None
        self.df_normalized = None
        self.numerical_cols = None
        self.clusters = None
        self.pca_results = None
        self.tsne_results = None
        self.network = None
        self.communities = None

    # ==================== TP1: EXPLORATION ET NETTOYAGE ====================

    def explore_data(self):
        """Exploration initiale des données (TP1)"""
        print("="*80)
        print("EXPLORATION DES DONNÉES")
        print("="*80)

        print("\n1. Premières lignes du dataset:")
        print(self.df.head())

        print("\n2. Informations générales:")
        print(self.df.info())

        print("\n3. Statistiques descriptives:")
        print(self.df.describe())

        print("\n4. Valeurs manquantes:")
        print(self.df.isnull().sum())

        return self.df.describe()

    def clean_data(self):
        """Nettoyage des données (TP1)"""
        print("\n" + "="*80)
        print("NETTOYAGE DES DONNÉES")
        print("="*80)

        self.df_clean = self.df.copy()

        # Supprimer les valeurs manquantes
        initial_rows = len(self.df_clean)
        self.df_clean = self.df_clean.dropna()
        print(f"\nLignes supprimées (valeurs manquantes): {initial_rows - len(self.df_clean)}")

        # Identifier les colonnes numériques
        self.numerical_cols = ['Age', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)',
                               'Stress_Level(1-10)', 'Days_Without_Social_Media',
                               'Exercise_Frequency(week)', 'Happiness_Index(1-10)']

        # Vérifier les outliers
        print("\n5. Détection des outliers (méthode IQR):")
        for col in self.numerical_cols:
            Q1 = self.df_clean[col].quantile(0.25)
            Q3 = self.df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df_clean[col] < (Q1 - 1.5 * IQR)) |
                        (self.df_clean[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"   {col}: {outliers} outliers")

        return self.df_clean

    def correlation_analysis(self):
        """Analyse des corrélations (TP1)"""
        print("\n" + "="*80)
        print("ANALYSE DES CORRÉLATIONS")
        print("="*80)

        # Corrélation de Pearson
        pearson_corr = self.df_clean[self.numerical_cols].corr(method='pearson')
        print("\nMatrice de corrélation de Pearson:")
        print(pearson_corr.round(3))

        # Corrélation de Spearman
        spearman_corr = self.df_clean[self.numerical_cols].corr(method='spearman')
        print("\nMatrice de corrélation de Spearman:")
        print(spearman_corr.round(3))

        # Identifier les corrélations fortes
        print("\nCorrélations fortes (|r| > 0.5):")
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                if abs(pearson_corr.iloc[i, j]) > 0.5:
                    print(f"   {pearson_corr.columns[i]} <-> {pearson_corr.columns[j]}: "
                          f"{pearson_corr.iloc[i, j]:.3f}")

        return pearson_corr, spearman_corr

    def test_normality(self):
        """Test de normalité (TP1)"""
        print("\n" + "="*80)
        print("TESTS DE NORMALITÉ (Shapiro-Wilk)")
        print("="*80)

        results = {}
        for col in self.numerical_cols:
            stat, p_value = stats.shapiro(self.df_clean[col].sample(min(5000, len(self.df_clean))))
            results[col] = {'statistic': stat, 'p_value': p_value}
            normal = "OUI" if p_value > 0.05 else "NON"
            print(f"\n{col}:")
            print(f"   Statistique: {stat:.4f}")
            print(f"   p-value: {p_value:.4f}")
            print(f"   Distribution normale (α=0.05): {normal}")

        return results

    # ==================== TP2: CLUSTERING ====================

    def normalize_data(self):
        """Normalisation des données pour le clustering (TP2)"""
        print("\n" + "="*80)
        print("NORMALISATION DES DONNÉES")
        print("="*80)

        scaler = StandardScaler()
        self.df_normalized = pd.DataFrame(
            scaler.fit_transform(self.df_clean[self.numerical_cols]),
            columns=self.numerical_cols,
            index=self.df_clean.index
        )

        print("\nDonnées normalisées (moyenne=0, std=1)")
        print(self.df_normalized.describe().round(3))

        return self.df_normalized

    def find_optimal_clusters(self, max_k=10):
        """Trouver le nombre optimal de clusters (TP2)"""
        print("\n" + "="*80)
        print("RECHERCHE DU NOMBRE OPTIMAL DE CLUSTERS")
        print("="*80)

        silhouette_scores = []
        inertias = []
        K_range = range(2, max_k+1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.df_normalized)
            silhouette_scores.append(silhouette_score(self.df_normalized, labels))
            inertias.append(kmeans.inertia_)

        print("\nScores de silhouette par nombre de clusters:")
        for k, score in zip(K_range, silhouette_scores):
            print(f"   k={k}: {score:.4f}")

        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"\nNombre optimal de clusters (silhouette max): {optimal_k}")

        return optimal_k, silhouette_scores, inertias

    def apply_clustering(self, n_clusters=3):
        """Application de différentes méthodes de clustering (TP2)"""
        print("\n" + "="*80)
        print(f"APPLICATION DU CLUSTERING ({n_clusters} clusters)")
        print("="*80)

        self.clusters = {}

        # K-Means
        print("\n1. K-Means:")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters['kmeans'] = kmeans.fit_predict(self.df_normalized)
        sil_kmeans = silhouette_score(self.df_normalized, self.clusters['kmeans'])
        print(f"   Score de silhouette: {sil_kmeans:.4f}")

        # Gaussian Mixture
        print("\n2. Gaussian Mixture:")
        gmm = BayesianGaussianMixture(n_components=n_clusters, random_state=42)
        self.clusters['gmm'] = gmm.fit_predict(self.df_normalized)
        sil_gmm = silhouette_score(self.df_normalized, self.clusters['gmm'])
        print(f"   Score de silhouette: {sil_gmm:.4f}")

        # DBSCAN
        print("\n3. DBSCAN:")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.clusters['dbscan'] = dbscan.fit_predict(self.df_normalized)
        n_clusters_db = len(set(self.clusters['dbscan'])) - (1 if -1 in self.clusters['dbscan'] else 0)
        n_noise = list(self.clusters['dbscan']).count(-1)
        print(f"   Nombre de clusters trouvés: {n_clusters_db}")
        print(f"   Points de bruit: {n_noise}")
        if n_clusters_db > 1:
            mask = self.clusters['dbscan'] != -1
            sil_dbscan = silhouette_score(self.df_normalized[mask], self.clusters['dbscan'][mask])
            print(f"   Score de silhouette: {sil_dbscan:.4f}")

        return self.clusters

    def interpret_clusters(self, method='kmeans'):
        """Interprétation des clusters (TP2)"""
        print("\n" + "="*80)
        print(f"INTERPRÉTATION DES CLUSTERS ({method.upper()})")
        print("="*80)

        df_with_clusters = self.df_clean.copy()
        df_with_clusters['Cluster'] = self.clusters[method]

        # Statistiques par cluster
        cluster_stats = df_with_clusters.groupby('Cluster')[self.numerical_cols].agg(['mean', 'count'])
        print("\nStatistiques par cluster:")
        print(cluster_stats.round(2))

        # Interprétation textuelle
        print("\nInterprétation des profils:")
        for cluster_id in sorted(df_with_clusters['Cluster'].unique()):
            if cluster_id == -1:  # Bruit pour DBSCAN
                continue
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
            size = len(cluster_data)

            avg_screen = cluster_data['Daily_Screen_Time(hrs)'].mean()
            avg_sleep = cluster_data['Sleep_Quality(1-10)'].mean()
            avg_stress = cluster_data['Stress_Level(1-10)'].mean()
            avg_happiness = cluster_data['Happiness_Index(1-10)'].mean()
            avg_exercise = cluster_data['Exercise_Frequency(week)'].mean()

            print(f"\nCluster {cluster_id} (n={size}):")
            print(f"   - Temps d'écran: {avg_screen:.1f}h/jour")
            print(f"   - Qualité sommeil: {avg_sleep:.1f}/10")
            print(f"   - Niveau stress: {avg_stress:.1f}/10")
            print(f"   - Indice bonheur: {avg_happiness:.1f}/10")
            print(f"   - Exercice: {avg_exercise:.1f} fois/semaine")

            # Profil textuel
            if avg_screen > 6:
                screen_desc = "usage intensif des écrans"
            elif avg_screen > 4:
                screen_desc = "usage modéré des écrans"
            else:
                screen_desc = "usage limité des écrans"

            if avg_happiness > 7:
                mood_desc = "niveau de bonheur élevé"
            elif avg_happiness > 5:
                mood_desc = "niveau de bonheur moyen"
            else:
                mood_desc = "niveau de bonheur faible"

            print(f"   → Profil: {screen_desc}, {mood_desc}")

        return df_with_clusters

    # ==================== TP5: RÉDUCTION DIMENSIONNELLE ====================

    def apply_pca(self, n_components=2):
        """Application de PCA (TP5)"""
        print("\n" + "="*80)
        print("RÉDUCTION DIMENSIONNELLE - PCA")
        print("="*80)

        pca = PCA(n_components=n_components)
        self.pca_results = pca.fit_transform(self.df_normalized)

        print(f"\nVariance expliquée par composante:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"   PC{i+1}: {var*100:.2f}%")
        print(f"   Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")

        print(f"\nContribution des variables aux composantes principales:")
        components_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=self.numerical_cols
        )
        print(components_df.round(3))

        return self.pca_results, pca

    def apply_tsne(self, n_components=2, perplexity=30):
        """Application de t-SNE (TP5)"""
        print("\n" + "="*80)
        print("RÉDUCTION DIMENSIONNELLE - t-SNE")
        print("="*80)

        # PCA d'abord pour réduire la dimension (bonne pratique)
        if self.df_normalized.shape[1] > 50:
            pca_50 = PCA(n_components=50)
            data_reduced = pca_50.fit_transform(self.df_normalized)
            print(f"PCA préliminaire: {self.df_normalized.shape[1]} -> 50 dimensions")
        else:
            data_reduced = self.df_normalized

        print(f"\nt-SNE avec perplexité={perplexity}")
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        self.tsne_results = tsne.fit_transform(data_reduced)

        print("   Embedding terminé")

        return self.tsne_results

    # ==================== TP3/TP4: ANALYSE DE RÉSEAU ====================

    def create_similarity_network(self, threshold=0.7):
        """Création d'un réseau de similarité entre utilisateurs (TP4/TP5)"""
        print("\n" + "="*80)
        print("CRÉATION DU RÉSEAU DE SIMILARITÉ")
        print("="*80)

        # Calculer la matrice de similarité cosinus
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(self.df_normalized)

        # Appliquer le seuil
        similarity_matrix[similarity_matrix < threshold] = 0
        np.fill_diagonal(similarity_matrix, 0)  # Enlever les auto-boucles

        print(f"\nSeuil de similarité: {threshold}")
        print(f"Nombre total de paires possibles: {len(self.df_clean) * (len(self.df_clean) - 1) // 2}")
        print(f"Nombre de connexions créées: {(similarity_matrix > 0).sum() // 2}")

        # Créer le graphe NetworkX
        self.network = nx.from_numpy_array(similarity_matrix)

        # Ajouter les attributs des nœuds
        for i, idx in enumerate(self.df_clean.index):
            self.network.nodes[i]['user_id'] = self.df_clean.loc[idx, 'User_ID']
            self.network.nodes[i]['age'] = self.df_clean.loc[idx, 'Age']
            self.network.nodes[i]['gender'] = self.df_clean.loc[idx, 'Gender']
            self.network.nodes[i]['happiness'] = self.df_clean.loc[idx, 'Happiness_Index(1-10)']
            self.network.nodes[i]['platform'] = self.df_clean.loc[idx, 'Social_Media_Platform']

        # Enlever les nœuds isolés
        self.network.remove_nodes_from(list(nx.isolates(self.network)))

        print(f"\nGraphe créé:")
        print(f"   Nœuds: {self.network.number_of_nodes()}")
        print(f"   Arêtes: {self.network.number_of_edges()}")

        return self.network

    def analyze_network(self):
        """Analyse des propriétés du réseau (TP3/TP4)"""
        print("\n" + "="*80)
        print("ANALYSE DU RÉSEAU")
        print("="*80)

        if self.network is None or self.network.number_of_nodes() == 0:
            print("Pas de réseau à analyser (augmenter le seuil de similarité)")
            return None

        # Propriétés de base
        n_nodes = self.network.number_of_nodes()
        n_edges = self.network.number_of_edges()
        density = nx.density(self.network)

        print(f"\n1. Propriétés de base:")
        print(f"   Nombre de nœuds: {n_nodes}")
        print(f"   Nombre d'arêtes: {n_edges}")
        print(f"   Densité: {density:.4f}")

        # Degré
        degrees = dict(self.network.degree())
        avg_degree = np.mean(list(degrees.values()))
        print(f"\n2. Degré:")
        print(f"   Degré moyen: {avg_degree:.2f}")
        print(f"   Degré max: {max(degrees.values())}")
        print(f"   Degré min: {min(degrees.values())}")

        # Composantes connexes
        if nx.is_connected(self.network):
            print(f"\n3. Connexité:")
            print(f"   Le graphe est connexe")
            clustering_coef = nx.average_clustering(self.network)
            print(f"   Coefficient de clustering: {clustering_coef:.4f}")

            avg_path_length = nx.average_shortest_path_length(self.network)
            diameter = nx.diameter(self.network)
            print(f"   Longueur moyenne des chemins: {avg_path_length:.4f}")
            print(f"   Diamètre: {diameter}")
        else:
            components = list(nx.connected_components(self.network))
            print(f"\n3. Connexité:")
            print(f"   Le graphe n'est pas connexe")
            print(f"   Nombre de composantes: {len(components)}")
            print(f"   Taille de la plus grande composante: {len(max(components, key=len))}")

            # Analyser la plus grande composante
            largest_cc = self.network.subgraph(max(components, key=len)).copy()
            clustering_coef = nx.average_clustering(largest_cc)
            print(f"   Coefficient de clustering (plus grande composante): {clustering_coef:.4f}")

        # Centralités (sur la plus grande composante)
        if not nx.is_connected(self.network):
            largest_cc = self.network.subgraph(max(nx.connected_components(self.network), key=len)).copy()
        else:
            largest_cc = self.network

        print(f"\n4. Centralités (sur la plus grande composante):")

        degree_centrality = nx.degree_centrality(largest_cc)
        betweenness_centrality = nx.betweenness_centrality(largest_cc)
        closeness_centrality = nx.closeness_centrality(largest_cc)

        # Top 5 nœuds par centralité
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top 5 - Degré: {[node for node, _ in top_degree]}")

        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top 5 - Betweenness: {[node for node, _ in top_betweenness]}")

        top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top 5 - Closeness: {[node for node, _ in top_closeness]}")

        # Stocker les centralités comme attributs
        nx.set_node_attributes(self.network, degree_centrality, 'degree_centrality')
        nx.set_node_attributes(self.network, betweenness_centrality, 'betweenness_centrality')
        nx.set_node_attributes(self.network, closeness_centrality, 'closeness_centrality')

        return {
            'density': density,
            'avg_degree': avg_degree,
            'clustering_coef': clustering_coef,
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality
        }

    def detect_communities(self):
        """Détection de communautés (TP4)"""
        print("\n" + "="*80)
        print("DÉTECTION DE COMMUNAUTÉS (Louvain)")
        print("="*80)

        if self.network is None or self.network.number_of_nodes() == 0:
            print("Pas de réseau à analyser")
            return None

        # Algorithme de Louvain
        self.communities = community_louvain.best_partition(self.network)

        n_communities = len(set(self.communities.values()))
        modularity = community_louvain.modularity(self.communities, self.network)

        print(f"\nNombre de communautés détectées: {n_communities}")
        print(f"Modularité: {modularity:.4f}")

        # Distribution des communautés
        community_sizes = {}
        for node, comm in self.communities.items():
            community_sizes[comm] = community_sizes.get(comm, 0) + 1

        print(f"\nTaille des communautés:")
        for comm, size in sorted(community_sizes.items()):
            print(f"   Communauté {comm}: {size} nœuds")

        # Ajouter comme attribut
        nx.set_node_attributes(self.network, self.communities, 'community')

        return self.communities

    def save_results(self, output_prefix='results'):
        """Sauvegarder les résultats de l'analyse"""
        print("\n" + "="*80)
        print("SAUVEGARDE DES RÉSULTATS")
        print("="*80)

        # Sauvegarder le dataset avec clusters
        if self.clusters is not None and 'kmeans' in self.clusters:
            df_results = self.df_clean.copy()
            df_results['Cluster_KMeans'] = self.clusters['kmeans']
            if 'gmm' in self.clusters:
                df_results['Cluster_GMM'] = self.clusters['gmm']
            if 'dbscan' in self.clusters:
                df_results['Cluster_DBSCAN'] = self.clusters['dbscan']

            output_file = f'{output_prefix}_with_clusters.csv'
            df_results.to_csv(output_file, index=False)
            print(f"\nDataset avec clusters sauvegardé: {output_file}")

        # Sauvegarder les résultats PCA
        if self.pca_results is not None:
            pca_df = pd.DataFrame(
                self.pca_results,
                columns=[f'PC{i+1}' for i in range(self.pca_results.shape[1])]
            )
            pca_df.to_csv(f'{output_prefix}_pca.csv', index=False)
            print(f"Résultats PCA sauvegardés: {output_prefix}_pca.csv")

        # Sauvegarder le réseau
        if self.network is not None and self.network.number_of_nodes() > 0:
            nx.write_gexf(self.network, f'{output_prefix}_network.gexf')
            print(f"Réseau sauvegardé: {output_prefix}_network.gexf (format Gephi)")

        print("\nTous les résultats ont été sauvegardés !")


def main():
    """Fonction principale pour exécuter toute l'analyse"""
    print("="*80)
    print("PROJET: ANALYSE MENTAL HEALTH & SOCIAL MEDIA BALANCE")
    print("="*80)

    # Charger et initialiser
    filepath = 'mental_health_social_media.csv'  # Adapter le nom du fichier
    analysis = MentalHealthAnalysis(filepath)

    # TP1: Exploration et nettoyage
    analysis.explore_data()
    analysis.clean_data()
    analysis.correlation_analysis()
    analysis.test_normality()

    # TP2: Clustering
    analysis.normalize_data()
    optimal_k, sil_scores, inertias = analysis.find_optimal_clusters(max_k=8)
    analysis.apply_clustering(n_clusters=optimal_k)
    analysis.interpret_clusters(method='kmeans')
    analysis.interpret_clusters(method='gmm')

    # TP5: Réduction dimensionnelle
    analysis.apply_pca(n_components=2)
    analysis.apply_tsne(n_components=2, perplexity=30)

    # TP3/TP4: Analyse de réseau
    analysis.create_similarity_network(threshold=0.7)
    analysis.analyze_network()
    analysis.detect_communities()

    # Sauvegarder tous les résultats
    analysis.save_results(output_prefix='mental_health_results')

    print("\n" + "="*80)
    print("ANALYSE TERMINÉE !")
    print("="*80)

    return analysis


if __name__ == "__main__":
    analysis = main()