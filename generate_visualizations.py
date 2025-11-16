"""
Génération de visualisations statiques pour le rapport
Crée des graphiques haute qualité pour la présentation et le rapport écrit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Configuration du style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class VisualizationGenerator:
    """Génère des visualisations de haute qualité pour le rapport"""

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.df_clean = self.df.dropna()
        self.numerical_cols = ['Age', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)',
                               'Stress_Level(1-10)', 'Days_Without_Social_Media',
                               'Exercise_Frequency(week)', 'Happiness_Index(1-10)']

        # Normalisation
        scaler = StandardScaler()
        self.df_normalized = pd.DataFrame(
            scaler.fit_transform(self.df_clean[self.numerical_cols]),
            columns=self.numerical_cols
        )

    def generate_all_plots(self, output_dir='figures'):
        """Génère toutes les visualisations"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("Génération des visualisations...")

        # 1. Vue d'ensemble
        self.plot_overview(f'{output_dir}/01_overview.png')

        # 2. Corrélations
        self.plot_correlations(f'{output_dir}/02_correlations.png')

        # 3. Distributions
        self.plot_distributions(f'{output_dir}/03_distributions.png')

        # 4. Relations clés
        self.plot_key_relationships(f'{output_dir}/04_relationships.png')

        # 5. Clustering
        self.plot_clustering_analysis(f'{output_dir}/05_clustering.png')

        # 6. PCA
        self.plot_pca_analysis(f'{output_dir}/06_pca.png')

        # 7. Profils par plateforme
        self.plot_platform_analysis(f'{output_dir}/07_platforms.png')

        # 8. Profils par genre
        self.plot_gender_analysis(f'{output_dir}/08_gender.png')

        print(f"\nToutes les visualisations ont été sauvegardées dans le dossier '{output_dir}/'")

    def plot_overview(self, filename):
        """Vue d'ensemble des données"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Vue d\'ensemble du dataset Mental Health & Social Media',
                     fontsize=16, fontweight='bold', y=0.995)

        # Distribution de l'âge
        axes[0, 0].hist(self.df_clean['Age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution de l\'âge', fontweight='bold')
        axes[0, 0].set_xlabel('Âge')
        axes[0, 0].set_ylabel('Fréquence')
        axes[0, 0].axvline(self.df_clean['Age'].mean(), color='red',
                           linestyle='--', label=f'Moyenne: {self.df_clean["Age"].mean():.1f}')
        axes[0, 0].legend()

        # Temps d'écran vs Bonheur
        scatter = axes[0, 1].scatter(self.df_clean['Daily_Screen_Time(hrs)'],
                                     self.df_clean['Happiness_Index(1-10)'],
                                     c=self.df_clean['Stress_Level(1-10)'],
                                     cmap='RdYlGn_r', alpha=0.6, s=50)
        axes[0, 1].set_title('Temps d\'écran vs Indice de bonheur', fontweight='bold')
        axes[0, 1].set_xlabel('Temps d\'écran quotidien (heures)')
        axes[0, 1].set_ylabel('Indice de bonheur (1-10)')
        cbar = plt.colorbar(scatter, ax=axes[0, 1])
        cbar.set_label('Niveau de stress')

        # Distribution par plateforme
        platform_counts = self.df_clean['Social_Media_Platform'].value_counts()
        axes[1, 0].bar(range(len(platform_counts)), platform_counts.values,
                       color=sns.color_palette("Set2", len(platform_counts)))
        axes[1, 0].set_title('Répartition par plateforme', fontweight='bold')
        axes[1, 0].set_xlabel('Plateforme')
        axes[1, 0].set_ylabel('Nombre d\'utilisateurs')
        axes[1, 0].set_xticks(range(len(platform_counts)))
        axes[1, 0].set_xticklabels(platform_counts.index, rotation=45, ha='right')

        # Statistiques clés
        axes[1, 1].axis('off')
        stats_text = f"""
        STATISTIQUES CLÉS
        
        Nombre d'utilisateurs: {len(self.df_clean)}
        
        Moyennes:
        • Âge: {self.df_clean['Age'].mean():.1f} ans
        • Temps d'écran: {self.df_clean['Daily_Screen_Time(hrs)'].mean():.1f} h/jour
        • Qualité sommeil: {self.df_clean['Sleep_Quality(1-10)'].mean():.1f}/10
        • Niveau stress: {self.df_clean['Stress_Level(1-10)'].mean():.1f}/10
        • Indice bonheur: {self.df_clean['Happiness_Index(1-10)'].mean():.1f}/10
        • Exercice: {self.df_clean['Exercise_Frequency(week)'].mean():.1f} fois/semaine
        
        Répartition genre:
        {self.df_clean['Gender'].value_counts().to_string()}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12,
                        verticalalignment='center', family='monospace')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Vue d'ensemble sauvegardée: {filename}")

    def plot_correlations(self, filename):
        """Matrice de corrélation"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Matrices de corrélation', fontsize=16, fontweight='bold')

        # Pearson
        corr_pearson = self.df_clean[self.numerical_cols].corr(method='pearson')
        sns.heatmap(corr_pearson, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, ax=axes[0], cbar_kws={'label': 'Corrélation'})
        axes[0].set_title('Corrélation de Pearson', fontweight='bold')

        # Spearman
        corr_spearman = self.df_clean[self.numerical_cols].corr(method='spearman')
        sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, ax=axes[1], cbar_kws={'label': 'Corrélation'})
        axes[1].set_title('Corrélation de Spearman', fontweight='bold')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Corrélations sauvegardées: {filename}")

    def plot_distributions(self, filename):
        """Distributions de toutes les variables numériques"""
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle('Distribution des variables', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i, col in enumerate(self.numerical_cols):
            # Histogramme avec KDE
            axes[i].hist(self.df_clean[col], bins=20, color='steelblue',
                         edgecolor='black', alpha=0.7, density=True)

            # KDE
            from scipy import stats
            kde = stats.gaussian_kde(self.df_clean[col])
            x_range = np.linspace(self.df_clean[col].min(), self.df_clean[col].max(), 100)
            axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

            axes[i].set_title(col, fontweight='bold')
            axes[i].set_xlabel('Valeur')
            axes[i].set_ylabel('Densité')
            axes[i].legend()

            # Ajouter les statistiques
            mean_val = self.df_clean[col].mean()
            median_val = self.df_clean[col].median()
            axes[i].axvline(mean_val, color='green', linestyle='--',
                            label=f'Moyenne: {mean_val:.2f}', alpha=0.7)
            axes[i].axvline(median_val, color='orange', linestyle='--',
                            label=f'Médiane: {median_val:.2f}', alpha=0.7)

        # Masquer les axes inutilisés
        for i in range(len(self.numerical_cols), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Distributions sauvegardées: {filename}")

    def plot_key_relationships(self, filename):
        """Relations clés entre variables"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Relations clés entre variables', fontsize=16, fontweight='bold')

        # 1. Temps d'écran vs Bonheur (coloré par stress)
        scatter1 = axes[0, 0].scatter(self.df_clean['Daily_Screen_Time(hrs)'],
                                      self.df_clean['Happiness_Index(1-10)'],
                                      c=self.df_clean['Stress_Level(1-10)'],
                                      cmap='RdYlGn_r', s=50, alpha=0.6)
        axes[0, 0].set_xlabel('Temps d\'écran (h/jour)')
        axes[0, 0].set_ylabel('Indice de bonheur (1-10)')
        axes[0, 0].set_title('Temps d\'écran vs Bonheur (couleur = stress)')
        plt.colorbar(scatter1, ax=axes[0, 0], label='Stress')

        # Ligne de tendance
        z = np.polyfit(self.df_clean['Daily_Screen_Time(hrs)'],
                       self.df_clean['Happiness_Index(1-10)'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.df_clean['Daily_Screen_Time(hrs)'].min(),
                             self.df_clean['Daily_Screen_Time(hrs)'].max(), 100)
        axes[0, 0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # 2. Exercice vs Bonheur
        scatter2 = axes[0, 1].scatter(self.df_clean['Exercise_Frequency(week)'],
                                      self.df_clean['Happiness_Index(1-10)'],
                                      c=self.df_clean['Age'],
                                      cmap='viridis', s=50, alpha=0.6)
        axes[0, 1].set_xlabel('Fréquence d\'exercice (fois/semaine)')
        axes[0, 1].set_ylabel('Indice de bonheur (1-10)')
        axes[0, 1].set_title('Exercice vs Bonheur (couleur = âge)')
        plt.colorbar(scatter2, ax=axes[0, 1], label='Âge')

        # 3. Qualité sommeil vs Stress
        scatter3 = axes[1, 0].scatter(self.df_clean['Sleep_Quality(1-10)'],
                                      self.df_clean['Stress_Level(1-10)'],
                                      c=self.df_clean['Daily_Screen_Time(hrs)'],
                                      cmap='plasma', s=50, alpha=0.6)
        axes[1, 0].set_xlabel('Qualité du sommeil (1-10)')
        axes[1, 0].set_ylabel('Niveau de stress (1-10)')
        axes[1, 0].set_title('Sommeil vs Stress (couleur = temps d\'écran)')
        plt.colorbar(scatter3, ax=axes[1, 0], label='Temps écran (h)')

        # 4. Jours sans SM vs Bonheur
        axes[1, 1].scatter(self.df_clean['Days_Without_Social_Media'],
                           self.df_clean['Happiness_Index(1-10)'],
                           c=self.df_clean['Exercise_Frequency(week)'],
                           cmap='coolwarm', s=50, alpha=0.6)
        axes[1, 1].set_xlabel('Jours sans réseaux sociaux')
        axes[1, 1].set_ylabel('Indice de bonheur (1-10)')
        axes[1, 1].set_title('Détox digitale vs Bonheur (couleur = exercice)')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Relations clés sauvegardées: {filename}")

    def plot_clustering_analysis(self, filename):
        """Analyse de clustering complète"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('Analyse de Clustering', fontsize=16, fontweight='bold')

        # 1. Elbow curve et silhouette
        ax1 = fig.add_subplot(gs[0, :2])
        K_range = range(2, 9)
        inertias = []
        silhouettes = []

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.df_normalized)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.df_normalized, labels))

        ax1_twin = ax1.twinx()
        line1 = ax1.plot(K_range, inertias, 'b-o', linewidth=2, markersize=8, label='Inertie')
        line2 = ax1_twin.plot(K_range, silhouettes, 'r-s', linewidth=2, markersize=8, label='Silhouette')

        ax1.set_xlabel('Nombre de clusters')
        ax1.set_ylabel('Inertie', color='b')
        ax1_twin.set_ylabel('Score de Silhouette', color='r')
        ax1.set_title('Méthode du coude et Score de Silhouette', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)

        # Légende combinée
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

        # 2. Clustering avec 3 clusters (PCA)
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.df_normalized)

        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(self.df_normalized)

        ax2 = fig.add_subplot(gs[0, 2])
        for i in range(optimal_k):
            mask = clusters == i
            ax2.scatter(pca_results[mask, 0], pca_results[mask, 1],
                        label=f'Cluster {i}', alpha=0.6, s=30)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title(f'Clustering K-Means (k={optimal_k})', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Silhouette plot détaillé
        ax3 = fig.add_subplot(gs[1, :])

        silhouette_vals = silhouette_samples(self.df_normalized, clusters)
        y_lower = 10

        for i in range(optimal_k):
            cluster_silhouette_vals = silhouette_vals[clusters == i]
            cluster_silhouette_vals.sort()

            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.viridis(float(i) / optimal_k)
            ax3.fill_betweenx(np.arange(y_lower, y_upper),
                              0, cluster_silhouette_vals,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax3.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax3.set_xlabel('Coefficient de Silhouette')
        ax3.set_ylabel('Cluster')
        ax3.set_title('Diagramme de Silhouette', fontweight='bold')
        ax3.axvline(x=silhouette_score(self.df_normalized, clusters),
                    color="red", linestyle="--", label='Moyenne')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4-6. Profils des clusters
        df_temp = self.df_clean.copy()
        df_temp['Cluster'] = clusters

        key_vars = ['Daily_Screen_Time(hrs)', 'Happiness_Index(1-10)', 'Stress_Level(1-10)']

        for idx, var in enumerate(key_vars):
            ax = fig.add_subplot(gs[2, idx])

            cluster_means = [df_temp[df_temp['Cluster'] == i][var].mean()
                             for i in range(optimal_k)]

            bars = ax.bar(range(optimal_k), cluster_means,
                          color=[plt.cm.viridis(float(i) / optimal_k) for i in range(optimal_k)])
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Valeur moyenne')
            ax.set_title(f'{var}', fontweight='bold')
            ax.set_xticks(range(optimal_k))

            # Ajouter les valeurs sur les barres
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom')

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Analyse de clustering sauvegardée: {filename}")

    def plot_pca_analysis(self, filename):
        """Analyse PCA détaillée"""
        pca = PCA()
        pca_full = pca.fit_transform(self.df_normalized)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Analyse en Composantes Principales (PCA)',
                     fontsize=16, fontweight='bold')

        # 1. Variance expliquée
        axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_)+1),
                       pca.explained_variance_ratio_ * 100,
                       alpha=0.7, color='steelblue')
        axes[0, 0].plot(range(1, len(pca.explained_variance_ratio_)+1),
                        np.cumsum(pca.explained_variance_ratio_) * 100,
                        'ro-', linewidth=2, markersize=8, label='Cumulée')
        axes[0, 0].set_xlabel('Composante principale')
        axes[0, 0].set_ylabel('Variance expliquée (%)')
        axes[0, 0].set_title('Variance expliquée par composante')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Biplot (PC1 vs PC2)
        pca_2d = PCA(n_components=2)
        pca_results = pca_2d.fit_transform(self.df_normalized)

        axes[0, 1].scatter(pca_results[:, 0], pca_results[:, 1],
                           c=self.df_clean['Happiness_Index(1-10)'],
                           cmap='RdYlGn', alpha=0.6, s=30)
        axes[0, 1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0, 1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
        axes[0, 1].set_title('Projection PCA (coloré par bonheur)')

        # Ajouter les vecteurs
        loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)
        for i, var in enumerate(self.numerical_cols):
            axes[0, 1].arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
                             head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)
            axes[0, 1].text(loadings[i, 0]*3.5, loadings[i, 1]*3.5, var,
                            fontsize=8, ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # 3. Loadings heatmap
        loadings_df = pd.DataFrame(
            pca_2d.components_.T,
            columns=['PC1', 'PC2'],
            index=self.numerical_cols
        )
        sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, ax=axes[1, 0], cbar_kws={'label': 'Loading'})
        axes[1, 0].set_title('Contributions des variables')

        # 4. Cercle des corrélations
        ax = axes[1, 1]
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
        ax.add_artist(circle)

        for i, var in enumerate(self.numerical_cols):
            ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                     head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.6)
            ax.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, var,
                    fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Cercle des corrélations')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Analyse PCA sauvegardée: {filename}")

    def plot_platform_analysis(self, filename):
        """Analyse par plateforme de réseaux sociaux"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Analyse par plateforme de réseaux sociaux',
                     fontsize=16, fontweight='bold')

        platforms = self.df_clean['Social_Media_Platform'].unique()

        # 1. Bonheur moyen par plateforme
        happiness_by_platform = self.df_clean.groupby('Social_Media_Platform')['Happiness_Index(1-10)'].mean().sort_values()
        axes[0, 0].barh(range(len(happiness_by_platform)), happiness_by_platform.values,
                        color=sns.color_palette("RdYlGn", len(happiness_by_platform)))
        axes[0, 0].set_yticks(range(len(happiness_by_platform)))
        axes[0, 0].set_yticklabels(happiness_by_platform.index)
        axes[0, 0].set_xlabel('Indice de bonheur moyen')
        axes[0, 0].set_title('Bonheur moyen par plateforme')
        axes[0, 0].grid(axis='x', alpha=0.3)

        # 2. Stress moyen par plateforme
        stress_by_platform = self.df_clean.groupby('Social_Media_Platform')['Stress_Level(1-10)'].mean().sort_values(ascending=False)
        axes[0, 1].barh(range(len(stress_by_platform)), stress_by_platform.values,
                        color=sns.color_palette("YlOrRd", len(stress_by_platform)))
        axes[0, 1].set_yticks(range(len(stress_by_platform)))
        axes[0, 1].set_yticklabels(stress_by_platform.index)
        axes[0, 1].set_xlabel('Niveau de stress moyen')
        axes[0, 1].set_title('Stress moyen par plateforme')
        axes[0, 1].grid(axis='x', alpha=0.3)

        # 3. Temps d'écran par plateforme
        screen_time_by_platform = self.df_clean.groupby('Social_Media_Platform')['Daily_Screen_Time(hrs)'].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(screen_time_by_platform)), screen_time_by_platform.values,
                       color=sns.color_palette("Blues_r", len(screen_time_by_platform)))
        axes[1, 0].set_xticks(range(len(screen_time_by_platform)))
        axes[1, 0].set_xticklabels(screen_time_by_platform.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Temps d\'écran moyen (h/jour)')
        axes[1, 0].set_title('Temps d\'écran moyen par plateforme')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Box plots comparatifs
        platform_data = [self.df_clean[self.df_clean['Social_Media_Platform'] == p]['Happiness_Index(1-10)'].values
                         for p in platforms]
        bp = axes[1, 1].boxplot(platform_data, labels=platforms, patch_artist=True)

        colors = sns.color_palette("Set2", len(platforms))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        axes[1, 1].set_xticklabels(platforms, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Indice de bonheur')
        axes[1, 1].set_title('Distribution du bonheur par plateforme')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Analyse par plateforme sauvegardée: {filename}")

    def plot_gender_analysis(self, filename):
        """Analyse par genre"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Analyse par genre', fontsize=16, fontweight='bold')

        genders = self.df_clean['Gender'].unique()

        # 1. Comparaison des moyennes
        metrics = ['Happiness_Index(1-10)', 'Stress_Level(1-10)',
                   'Daily_Screen_Time(hrs)', 'Exercise_Frequency(week)']

        gender_means = self.df_clean.groupby('Gender')[metrics].mean()

        x = np.arange(len(metrics))
        width = 0.25

        for i, gender in enumerate(genders):
            offset = width * (i - len(genders)/2 + 0.5)
            axes[0, 0].bar(x + offset, gender_means.loc[gender], width,
                           label=gender, alpha=0.8)

        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(['Bonheur', 'Stress', 'Temps écran', 'Exercice'],
                                   rotation=45, ha='right')
        axes[0, 0].set_ylabel('Valeur moyenne')
        axes[0, 0].set_title('Comparaison des métriques par genre')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Distribution de l'âge par genre
        for gender in genders:
            gender_data = self.df_clean[self.df_clean['Gender'] == gender]['Age']
            axes[0, 1].hist(gender_data, bins=15, alpha=0.6, label=gender)

        axes[0, 1].set_xlabel('Âge')
        axes[0, 1].set_ylabel('Fréquence')
        axes[0, 1].set_title('Distribution de l\'âge par genre')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. Scatter: Temps écran vs Bonheur par genre
        for gender in genders:
            gender_data = self.df_clean[self.df_clean['Gender'] == gender]
            axes[1, 0].scatter(gender_data['Daily_Screen_Time(hrs)'],
                               gender_data['Happiness_Index(1-10)'],
                               label=gender, alpha=0.6, s=50)

        axes[1, 0].set_xlabel('Temps d\'écran (h/jour)')
        axes[1, 0].set_ylabel('Indice de bonheur')
        axes[1, 0].set_title('Temps d\'écran vs Bonheur par genre')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. Violin plots comparatifs
        gender_data_list = [self.df_clean[self.df_clean['Gender'] == g]['Happiness_Index(1-10)'].values
                            for g in genders]

        parts = axes[1, 1].violinplot(gender_data_list, positions=range(len(genders)),
                                      showmeans=True, showmedians=True)

        axes[1, 1].set_xticks(range(len(genders)))
        axes[1, 1].set_xticklabels(genders)
        axes[1, 1].set_ylabel('Indice de bonheur')
        axes[1, 1].set_title('Distribution du bonheur par genre (Violin plot)')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Analyse par genre sauvegardée: {filename}")


def main():
    """Fonction principale"""
    print("="*80)
    print("GÉNÉRATION DES VISUALISATIONS POUR LE RAPPORT")
    print("="*80)

    filepath = 'mental_health_social_media.csv'
    viz = VisualizationGenerator(filepath)
    viz.generate_all_plots(output_dir='figures')

    print("\n" + "="*80)
    print("GÉNÉRATION TERMINÉE !")
    print("="*80)
    print("\nToutes les figures sont prêtes pour votre rapport et présentation.")
    print("Vous pouvez les trouver dans le dossier 'figures/'")


if __name__ == "__main__":
    main()