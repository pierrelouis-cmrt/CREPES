# Résumé théorique

Ce fichier résume les équations utilisées ou conservées dans le projet. Les
dérivations complètes restent dans `documents_sources/`.

## Moteur de surface actif

Base Carcajous modèle 4 :

$$
C\frac{dT_s}{dt}
= \Phi_{\mathrm{solaire}}
- Q_{\mathrm{latent}}
+ \sigma T_{\mathrm{atm}}^4
- \sigma T_s^4
$$

Le moteur utilise une intégration implicite de type Backward Euler.

## Soleil et albédo

Flux solaire absorbé :

$$
\Phi_{\mathrm{solaire}}
= S_0 \max(\cos i,0)(1-A_{\mathrm{nuages}})(1-A_{\mathrm{sol}})
$$

Les séries $A_{\mathrm{sol}}$ viennent des CSV mensuels. Les nuages viennent du
fichier CERES.

## Capacité thermique

Capacité surfacique principale, depuis l'humidité RZSM Carcajous :

$$
C = c_p \rho_{\mathrm{bulk}} e
$$

Si RZSM manque localement, le moteur utilise un fallback sec constant :

$$
C_{\mathrm{sec}} = c_{p,\mathrm{sec}}\rho_{\mathrm{bulk}}e
$$

## Chaleur latente

Flux latent Carcajous par continent :

$$
Q_{\mathrm{latent}} = \Delta h_{\mathrm{vap}}\rho_{\mathrm{eau}}E
$$

La variante Chevreaux d'évapotranspiration est documentée dans les PDF, mais
n'est pas le modèle actif.

## Convection active par défaut

Convection forcée Chevreaux :

$$
Re = \frac{\rho v L}{\mu}, \qquad
Nu = C_R Re^m Pr^n, \qquad
h = \frac{Nu\lambda}{L}
$$

Convection naturelle Ornithorynquietant :

$$
Ra = \frac{g\beta(T_s-T_{air})L^3}{\nu^2}Pr, \qquad
Nu = C_R |Ra|^{1/4}
$$

Dans les deux cas, le flux appliqué est :

$$
Q_{\mathrm{conv}} = h(T_s-T_{air})
$$

Par défaut, le moteur additionne le flux de convection forcée et le flux de
convection naturelle :

$$
Q_{\mathrm{conv,total}} = Q_{\mathrm{conv,forcee}} + Q_{\mathrm{conv,naturelle}}
$$

La désactivation des deux convections ou la sélection d'une seule convection est
documentée dans `README.md`.

## Modules conservés mais non branchés

La diffusion radiale est conservée dans `codes_python/physique/diffusion.py`,
mais non branchée au moteur car le flux calculé dans le script source reste
ambigu pour une lecture "flux de surface".

La partie gaz à effet de serre n'est pas intégrée pour l'instant.
