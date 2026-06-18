# Modele 5 temporaire - emission laterale par couche

Le modele 5 temporaire reprend le modele 3.1 et ajoute une sortie : ce que
chaque couche atmospherique emet lateralement vers les cotes.

Il garde donc le fonctionnement habituel :

```text
lat, lon -> colonne locale -> flux radiatifs verticaux
```

puis ajoute :

```text
couche k -> emission laterale sortante nord/sud/est/ouest
```

## Fichiers

| Fichier | Role |
| --- | --- |
| `modele5_temporaire.py` | Appelle le modele 3.1 et ajoute l'emission laterale sortante par couche. |
| `README.md` | Explique la formule, les sorties et les limites. |

## Principe

Le modele 3.1 calcule deja les flux radiatifs verticaux :

```text
flux montant   : surface/couches -> espace
flux descendant: couches -> surface
```

Ici, on ne regarde pas les voisins. On calcule seulement ce que la couche locale
emettrait vers ses faces laterales.

Pour chaque couche et chaque bande infrarouge du modele 3.1 :

```text
flux_lateral_bande = emissivite_couche_bande * flux_corps_noir_bande(T_couche)
```

Le flux lateral sortant par cote est ensuite :

```text
flux_lateral_par_cote =
    somme_sur_les_bandes(flux_lateral_bande)
```

Hypothese temporaire :

```text
nord = sud = est = ouest = flux_lateral_par_cote
```

Donc :

```text
flux_lateral_4_cotes = 4 * flux_lateral_par_cote
```

## Pourquoi les quatre cotes sont identiques

Dans cette version, on calcule l'emission propre de la couche, pas un echange
avec une colonne voisine.

La temperature, le CO2, la vapeur d'eau et l'emissivite de la couche sont les
memes quelle que soit la direction laterale. Donc le flux sortant nord, sud, est
et ouest est le meme.

Une future version pourrait ensuite propager ce flux vers les colonnes voisines,
mais ce n'est pas encore fait ici.

## Lancer

Depuis la racine du depot :

```bash
./.venv/bin/python "modele5 (temporaire)/modele5_temporaire.py" \
  --lat 48.8566 \
  --lon 2.3522 \
  --mois 7 \
  --temperature-surface 293.0 \
  --moyenne-journaliere-sw
```

Sortie JSON complete :

```bash
./.venv/bin/python "modele5 (temporaire)/modele5_temporaire.py" \
  --lat 48.8566 \
  --lon 2.3522 \
  --mois 7 \
  --json
```

## Sorties principales

Le resultat contient deux blocs :

```text
flux_radiatifs_3_1
emission_laterale_sortante
```

`flux_radiatifs_3_1` reprend les sorties du modele 3.1 :

- `SW_absorbe_surface` : solaire absorbe par la surface ;
- `LW_up_surface` : infrarouge emis par la surface ;
- `LW_down_surface` : infrarouge descendant emis par l'atmosphere ;
- `OLR` : infrarouge sortant vers l'espace ;
- `flux_net_radiatif_surface` : bilan radiatif net de surface.

`emission_laterale_sortante` ajoute, pour chaque couche :

- pression bas/haut de la couche ;
- temperature de la couche ;
- masse d'air de la couche ;
- `flux_sortant_lateral_par_cote_w_m2` ;
- `flux_sortant_lateral_4_cotes_w_m2` ;
- `flux_sortant_lateral_par_direction_w_m2` avec nord, sud, est, ouest ;
- diagnostics par bande infrarouge.

## Difference avec un lien horizontal

Ce dossier ne calcule plus :

```text
K_h * (T_voisin - T_centre)
```

Il calcule :

```text
ce que la couche centrale emet elle-meme sur les cotes
```

Le lien horizontal complet viendrait apres :

```text
emission laterale d'une couche
transmission/absorption vers la colonne voisine
effet sur la temperature de la colonne voisine
```

## Limites assumees

- Les temperatures des couches atmospheriques restent imposees par le paquet
  3.1.
- Les flux lateraux ne sont pas encore propages vers les cellules voisines.
- Les quatre cotes sont identiques par hypothese isotrope.
- L'unite est `W m-2` de face laterale, pas encore convertie en contribution
  par surface horizontale de cellule.
- Ce modele est temporaire : il sert a isoler clairement la sortie "emission
  laterale par couche" avant un vrai couplage horizontal.
