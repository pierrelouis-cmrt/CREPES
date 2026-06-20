# Modèle Radiatif Simplifié à 1 Dimension

Ce projet contient deux scripts Python :
1.  `spectre_absorbance_ch4.py` : Un script pour calculer et visualiser le spectre d'absorption détaillé du méthane (CH4) en utilisant la bibliothèque scientifique RADIS.
2.  `modele3.py` : Un modèle climatique simplifié qui simule le transfert de rayonnement infrarouge à travers une colonne atmosphérique découpée en plusieurs couches, en se concentrant sur l'effet du CH4.

## Installation

Pour utiliser ces scripts, vous devez avoir Python 3.9+ installé sur votre système. Les dépendances peuvent être installées via `pip`.

1.  **Créez un environnement virtuel (recommandé)** :
    ```bash
    python -m venv .venv
    # Sur Windows
    .venv\Scripts\activate
    # Sur Linux/macOS
    source .venv/bin/activate
    ```

2.  **Installez les bibliothèques nécessaires** :
    ```bash
    pip install numpy radis matplotlib
    ```

## `modele3.py` : Explication du Modèle Physique et des Calculs

Ce script simule comment le rayonnement infrarouge émis par la surface de la Terre est absorbé et ré-émis par les couches de l'atmosphère contenant du méthane. L'objectif est de calculer le flux radiatif qui s'échappe vers l'espace et celui qui est renvoyé vers la surface.

### 1. Structure du Modèle

L'atmosphère est découpée verticalement en 6 couches (`COUCHES_DEPART`), chacune caractérisée par :
- Une altitude de base et de sommet.
- Une température moyenne fixe.
- Une pression à ses bords, calculée selon l'**Atmosphère Standard Internationale (ISA)**.
- Une concentration moyenne de CH4.

La surface de la Terre est modélisée comme un corps noir à une température fixe (`TEMPERATURE_SURFACE_K`).

### 2. Transfert Radiatif

Le modèle se base sur les principes suivants :

1.  **Émission de la surface** : La surface émet un rayonnement infrarouge comme un corps noir, dont l'intensité est régie par la **loi de Stefan-Boltzmann** (`Flux = σ * T⁴`).
2.  **Absorption par les gaz** : Le CH4 n'absorbe pas à toutes les longueurs d'onde. Le modèle définit une "bande d'absorption" (`BANDES_ABSORPTION`) où le gaz est actif. En dehors de cette bande, l'atmosphère est considérée comme transparente.
3.  **Absorption et Émission par les couches** : Dans la bande d'absorption, chaque couche atmosphérique peut :
    - **Absorber** une partie du rayonnement qui la traverse (venant du bas ou du haut).
    - **Émettre** son propre rayonnement (car elle a une température non nulle).
4.  **Propagation des flux** :
    - Un **flux montant** part de la surface, traverse les couches successives jusqu'au sommet de l'atmosphère.
    - Un **flux descendant** part du sommet de l'atmosphère (flux nul au départ) et se propage vers la surface, s'enrichissant de l'émission de chaque couche.

### 3. Calculs Clés et Formules

#### Luminance de Planck (`luminance_spectrale_planck`)

Le modèle calcule l'émission d'un corps noir en utilisant la loi de Planck, qui donne la luminance (intensité) du rayonnement pour une longueur d'onde et une température données.

#### Flux de Corps Noir dans une Bande (`flux_corps_noir_bande`)

Pour obtenir le flux total en W/m² dans une bande spectrale, le script intègre numériquement la luminance de Planck sur l'intervalle de longueurs d'onde de cette bande.

#### Épaisseur Optique (`epaisseur_optique`)

C'est le concept le plus important pour l'absorption. L'épaisseur optique (ou opacité), notée `τ` (tau), est un nombre sans dimension qui quantifie la capacité d'une couche à bloquer le rayonnement. Si `τ = 0`, la couche est transparente. Si `τ` est très grand, elle est opaque.

Dans ce modèle, elle est calculée avec la formule simplifiée suivante :

```
τ = a_bande * (C / C_ref) * (ΔP / P_surface)
```

Décortiquons cette formule :

-   `a_bande` (`coefficient_opacite`) : C'est un coefficient d'opacité effectif pour la bande spectrale. Il représente la "force" intrinsèque de l'absorption du gaz dans cette bande. C'est une valeur empirique qui simplifie une physique complexe.
-   `(C / C_ref)` (`concentration_ppm / reference_ppm`) : Ce terme module l'opacité en fonction de la concentration du gaz. `C` est la concentration actuelle de CH4 dans la couche, et `C_ref` est une concentration de référence (ici, pré-industrielle). Si la concentration double, l'opacité double.
-   `(ΔP / P_surface)` (`couche.epaisseur_pression_pa / PRESSION_SURFACE_PA`) : Ce terme représente la "masse d'air" relative de la couche. `ΔP` est la différence de pression entre le bas et le haut de la couche. En la divisant par la pression totale à la surface, on obtient une fraction qui représente la part de la masse totale de l'atmosphère contenue dans cette couche. Plus une couche est épaisse (en termes de pression), plus elle contient de molécules de gaz et plus son opacité est grande.

#### Transmission et Émissivité

Une fois l'épaisseur optique `τ` calculée, on en déduit :

-   **La transmission (`transmission_depuis_tau`)** : C'est la fraction du rayonnement qui réussit à traverser la couche.
    ```
    Transmission = exp(-D * τ)
    ```
    Où `D` est le `FACTEUR_DIFFUSIF` (ici 1.0), un facteur qui tient compte du fait que le rayonnement se propage dans toutes les directions et pas seulement à la verticale.

-   **L'émissivité (`emissivite_depuis_transmission`)** : C'est la capacité de la couche à émettre son propre rayonnement, par rapport à un corps noir parfait. Selon la **loi de Kirchhoff**, elle est égale à l'absorptivité.
    ```
    Émissivité = 1 - Transmission
    ```

### 4. Description des Fonctions Principales

-   `calculer_atmosphere_standard`: Calcule les profils de température et de pression en fonction de l'altitude, en se basant sur le modèle de l'Atmosphère Standard Internationale (ISA).

-   `concentration_moyenne_ppm_par_couche`: Calcule la concentration moyenne d'un gaz dans une couche, en pondérant par la masse d'air (pression) pour une meilleure justesse physique.

-   `creer_couches`: Initialise la structure de l'atmosphère en créant les 6 objets `CoucheAtmospherique` avec leurs propriétés (température, pression, concentration de CH4).

-   `epaisseur_optique`: Calcule l'opacité `τ` d'une couche pour une bande spectrale donnée, en utilisant la formule détaillée ci-dessus.

-   `propager_flux_montant`: Simule le trajet du rayonnement depuis la surface vers l'espace. Pour chaque couche, le flux entrant est partiellement transmis, et on y ajoute l'émission propre de la couche.
    `Flux_sortant = (Flux_entrant * Transmission) + (Emission_couche * Émissivité)`

-   `propager_flux_descendant`: Fait la même chose mais du haut vers le bas, pour calculer le rayonnement renvoyé vers la surface (le "back-radiation").

-   `calculer_flux`: Orchestre tous les calculs. Il calcule le flux sortant total (partie transparente + partie absorbée/ré-émise) et le flux descendant total.

-   `main`: La fonction principale qui exécute le modèle et affiche un résumé des couches, des opacités et des flux finaux.

### 5. Comment Lancer le Modèle

Le script est autonome. Pour l'exécuter, il suffit de lancer la commande suivante dans un terminal :

```bash
python modele3.py
```

Les résultats, incluant les flux et les opacités par couche, seront affichés dans la console.

### 6. Limites du Modèle

Ce script est un outil pédagogique et comporte de nombreuses simplifications par rapport aux modèles climatiques réels :

-   **Modèle à une dimension (1D)** : Il ne considère que les transferts verticaux de rayonnement et ignore complètement les transports horizontaux de chaleur par les vents et les courants marins.
-   **Bandes spectrales rectangulaires** : L'absorption du CH4 est modélisée par une seule bande "grise" avec un coefficient d'opacité unique. En réalité, l'absorption est un phénomène complexe composé de milliers de raies spectrales fines. Le script `spectre_absorbance_ch4.py` donne une vision plus réaliste de cette complexité.
-   **Températures des couches fixes** : Les températures des couches sont prédéfinies. Un modèle plus avancé calculerait ces températures à l'équilibre radiatif (ou radio-convectif).
-   **Absence de convection** : Le modèle est purement radiatif. Il ignore le transport de chaleur par convection, qui est le processus dominant dans la troposphère.
-   **Absence de nuages et d'aérosols** : Les nuages et les aérosols ont un impact majeur sur le bilan radiatif de la Terre (par réflexion du rayonnement solaire et absorption/émission de l'infrarouge), qui n'est pas pris en compte ici.
-   **Facteur de diffusivité** : Le `FACTEUR_DIFFUSIF` est utilisé pour approximer l'intégration sur tous les angles du rayonnement. Une valeur de 1.66 est souvent utilisée (approximation de Schwarzschild), mais ici il est fixé à 1.0 pour plus de simplicité.

---

## `spectre_absorbance_ch4.py` : Calcul de Spectre Détaillé

Ce script utilise la bibliothèque `radis` pour calculer un spectre d'absorption à haute résolution du CH4. Contrairement à `modele3.py` qui utilise une seule "bande" simplifiée, ce script résout des milliers de raies d'absorption individuelles.

### Fonctionnalités

-   Calcule le spectre pour des conditions de température, pression, concentration et longueur de trajet données.
-   Prend en compte les deux isotopes principaux du méthane (`¹²CH₄` et `¹³CH₄`).
-   Peut afficher le spectre sous forme de graphique.
-   Peut sauvegarder les données brutes (longueur d'onde, absorbance) dans un fichier CSV.

### Comment Lancer le Script

Le script accepte des arguments en ligne de commande pour personnaliser le calcul.

**Exemple d'utilisation :**

```bash
# Lancer avec les paramètres par défaut et afficher le graphique
python spectre_absorbance_ch4.py

# Changer la concentration de CH4 et sauvegarder le graphique et les données
python spectre_absorbance_ch4.py --ch4-ppm 4.0 --output mon_spectre.png --csv donnees_spectre.csv
```

Pour voir toutes les options disponibles :

```bash
python spectre_absorbance_ch4.py --help
```

<!--
[PROMPT_SUGGESTION]Comment puis-je modifier `modele3.py` pour simuler l'effet d'un doublement de la concentration de CH4 ?[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]Explique-moi plus en détail le fonctionnement de la fonction `calculer_atmosphere_standard` et les constantes utilisées.[/PROMPT_SUGGESTION]
