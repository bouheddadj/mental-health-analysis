# 📊 Projet: Mental Health & Social Media Balance Analysis

## 🎯 Objectif du projet

Ce projet applique les techniques d'analyse de données vues en cours (TPs 1-5) sur un dataset réel concernant la santé mentale et l'usage des réseaux sociaux. Il comprend:

- ✅ **Exploration et nettoyage des données** (TP1)
- ✅ **Analyse des corrélations** (TP1)
- ✅ **Clustering** avec KMeans, Gaussian Mixture et DBSCAN (TP2)
- ✅ **Réduction dimensionnelle** avec PCA et t-SNE (TP5)
- ✅ **Analyse de réseau** avec NetworkX (TP3/TP4)
- ✅ **Dashboard interactif** avec Dash et Plotly

---

## 📁 Structure du projet

```
projet/
├── app.py                              # Dashboard Dash (version Render)
├── data_analysis.py                    # Script d'analyse complète
├── generate_visualizations.py          # Génération de graphiques pour rapport
├── requirements.txt                    # Dépendances Python
├── runtime.txt                         # Version Python pour Render
├── render.yaml                         # Configuration Render
├── .gitignore                          # Fichiers à ignorer par Git
├── mental_health_social_media.csv      # Dataset
├── README.md                           # Ce fichier
```

---

## 🚀 Installation

### 1. Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### 2. Installation des dépendances

```bash
# Dans le terminal (IntelliJ ou autre)
pip install -r requirements.txt
```

### 3. Préparer le dataset

Placez votre fichier CSV dans le même dossier que les scripts Python, et nommez-le:
```
mental_health_social_media.csv
```

---

## 💻 Utilisation

### Option 1: Lancer l'analyse complète

Pour exécuter toutes les analyses (exploration, clustering, PCA, réseaux):

```bash
python data_analysis.py
```

**Résultats générés:**
- `mental_health_results_with_clusters.csv` - Dataset avec les clusters
- `mental_health_results_pca.csv` - Résultats de la PCA
- `mental_health_results_network.gexf` - Réseau (format Gephi)

### Option 2: Lancer le dashboard interactif

Pour lancer l'interface web interactive:

```bash
python app.py
```

Puis ouvrez votre navigateur à l'adresse: **http://localhost:8050**

---

## 📊 Fonctionnalités du Dashboard

Le dashboard comprend **6 onglets interactifs**:

### 1️⃣ **Exploration des données**
- 📈 Distribution des variables (histogramme + boxplot)
- 📋 Statistiques descriptives
- 🔥 Matrice de corrélation interactive

### 2️⃣ **Relations entre variables**
- 🎯 Scatter plots personnalisables
- 🎨 Coloration par genre, plateforme ou cluster
- 📏 Taille des points ajustable

### 3️⃣ **Clustering**
- 🎚️ Nombre de clusters ajustable (2-8)
- 📊 Score de silhouette
- 📉 Courbe d'Elbow
- 👥 Profils détaillés des clusters

### 4️⃣ **Réduction dimensionnelle**
- 🗺️ Visualisation PCA
- 🌐 Visualisation t-SNE (perplexité ajustable)
- 📊 Contribution des variables

### 5️⃣ **Analyse de réseau**
- 🕸️ Réseau de similarité entre utilisateurs
- 🎚️ Seuil de similarité ajustable
- 📊 Distribution des degrés
- 🏘️ Détection de communautés (Louvain)

### 6️⃣ **Insights & Recommandations**
- 💡 Corrélations importantes
- 👤 Profils d'utilisateurs identifiés
- 📱 Impact des plateformes
- 💪 Recommandations basées sur les données

---

## 🔬 Techniques appliquées (selon les TPs)

### TP1: Exploration et nettoyage
- ✅ Chargement et exploration des données
- ✅ Statistiques descriptives
- ✅ Détection de valeurs manquantes et outliers
- ✅ Corrélations de Pearson et Spearman
- ✅ Test de normalité (Shapiro-Wilk)
- ✅ Visualisations (histogrammes, scatter plots)

### TP2: Clustering
- ✅ Normalisation des données (StandardScaler)
- ✅ K-Means avec choix optimal du nombre de clusters
- ✅ Gaussian Mixture Model
- ✅ DBSCAN
- ✅ Score de silhouette
- ✅ Méthode du coude (Elbow method)
- ✅ Interprétation des clusters

### TP5: Réduction dimensionnelle
- ✅ PCA (Principal Component Analysis)
- ✅ Variance expliquée
- ✅ Contribution des variables
- ✅ t-SNE avec différentes perplexités
- ✅ Visualisation des embeddings

### TP3/TP4: Analyse de réseau
- ✅ Création de réseau de similarité
- ✅ Calcul de similarité cosinus
- ✅ Propriétés du graphe (densité, degré, clustering)
- ✅ Centralités (degré, betweenness, closeness)
- ✅ Composantes connexes
- ✅ Détection de communautés (Louvain)
- ✅ Export au format Gephi

---

## 📈 Exemples de résultats

### Insights typiques obtenus:

1. **Corrélations importantes:**
    - Temps d'écran ↔ Qualité du sommeil (négative)
    - Exercice ↔ Indice de bonheur (positive)
    - Stress ↔ Bonheur (négative)

2. **Profils d'utilisateurs identifiés:**
    - 🟢 Cluster 0: Usage limité, haute qualité de vie
    - 🟡 Cluster 1: Usage modéré, équilibre moyen
    - 🔴 Cluster 2: Usage intensif, stress élevé

3. **Impact des plateformes:**
    - Différences de bien-être selon les plateformes
    - Corrélation avec le temps d'écran

---

## 🛠️ Personnalisation

### Modifier le dataset

Pour utiliser un autre dataset, modifiez cette ligne dans les fichiers:

```python
df = pd.read_csv('votre_fichier.csv')
```

Et adaptez la liste des colonnes numériques:

```python
numerical_cols = ['colonne1', 'colonne2', ...]
```

### Ajuster les paramètres

Dans `data_analysis.py`, vous pouvez modifier:
- Nombre de clusters: `n_clusters`
- Seuil de similarité pour le réseau: `threshold`
- Perplexité pour t-SNE: `perplexity`

---

## 🎓 Concepts appliqués

### Statistiques
- Moyenne, médiane, écart-type, quartiles
- Corrélation de Pearson et Spearman
- Tests de normalité

### Machine Learning
- Clustering non supervisé
- Réduction de dimensionnalité
- Normalisation des données
- Métriques d'évaluation (silhouette score)

### Analyse de réseaux
- Théorie des graphes
- Mesures de centralité
- Détection de communautés
- Analyse de similarité

### Visualisation
- Graphiques interactifs (Plotly)
- Dashboard web (Dash)
- Cartes de chaleur
- Réseaux

---

## 📝 Notes importantes

1. **Performance**: Pour de gros datasets (>10000 lignes), t-SNE peut être lent. Utilisez PCA d'abord.

2. **Réseau**: Si le seuil de similarité est trop élevé, le réseau sera vide. Ajustez-le entre 0.5 et 0.8.

3. **Clusters**: Le nombre optimal de clusters dépend des données. Utilisez le score de silhouette comme guide.

4. **Dashboard**: Pour de meilleures performances, utilisez un échantillon des données si le dataset est très grand.

---

## 🐛 Dépannage

### Erreur: "No module named..."
```bash
pip install -r requirements.txt
```

### Erreur: "File not found"
Vérifiez que le fichier CSV est dans le bon dossier et bien nommé.

### Dashboard ne s'affiche pas
- Vérifiez que le port 8050 est libre
- Essayez: `app.run_server(debug=True, port=8051)`

### Calculs trop lents
- Réduisez la taille du dataset
- Diminuez le nombre d'itérations pour t-SNE
- Utilisez un seuil de similarité plus élevé pour le réseau

---

## 📚 Références

- **Pandas**: https://pandas.pydata.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Plotly/Dash**: https://dash.plotly.com/
- **NetworkX**: https://networkx.org/documentation/stable/
- **Cours**: Documents des TPs 1-5