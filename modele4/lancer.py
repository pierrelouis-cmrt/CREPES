"""TUI de lancement des scripts du modele 4.

Ce fichier ne contient pas de logique physique. Il aide a choisir un scenario,
ajuster les parametres les plus courants, afficher la commande, puis lancer le
script existant avec ``subprocess``.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RACINE_PROJET = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


@dataclass(frozen=True)
class Parametre:
    option: str
    valeur: str
    description: str
    choix: tuple[str, ...] = ()
    drapeau: bool = False


@dataclass(frozen=True)
class Scenario:
    titre: str
    module: str
    description: str
    parametres: tuple[Parametre, ...]


@dataclass(frozen=True)
class Moteur:
    titre: str
    resume: str
    technique: tuple[str, ...]
    physique: tuple[str, ...]
    usage: tuple[str, ...]
    scenarios: tuple[Scenario, ...]


COMMUN_PHYSIQUE = (
    Parametre("--co2", "420", "CO2 en ppm transmis au modele radiatif."),
    Parametre(
        "--temperature-initiale",
        "",
        "Temperature imposee partout en K ; vide = donnees du paquet.",
    ),
    Parametre(
        "--convection",
        "toutes",
        "Convection de surface.",
        choix=("aucune", "forcee", "naturelle", "toutes"),
    ),
    Parametre("--facteur-latent", "1", "Multiplicateur du flux latent ; 0 le coupe."),
    Parametre("--vent", "2.5", "Vent en m/s pour la convection forcee."),
    Parametre("--rzsm-csv", "", "CSV d'humidite du sol ; vide = source par defaut."),
    Parametre("--paquet", "", "Paquet grille du modele 3 ; vide = source par defaut."),
    Parametre("--no-progress", "non", "Desactive la barre de progression.", drapeau=True),
)


SCENARIOS_RAPIDE = (
    Scenario(
        titre="Petite grille de developpement",
        module="modele4.rapide",
        description=(
            "Premier choix pour tester vite les parametres sur 4 x 8 cellules."
        ),
        parametres=(
            Parametre("--jours", "1", "Duree simulee en jours."),
            Parametre("--dt", "1800", "Pas interne en secondes."),
            Parametre("--sortie-heures", "4", "Frequence d'ecriture des cartes."),
            Parametre("--max-latitudes", "4", "Nombre de latitudes gardees."),
            Parametre("--max-longitudes", "8", "Nombre de longitudes gardees."),
            Parametre("--output", "modele4/sorties/rapide_dev.npz", "Fichier NPZ ecrit."),
            *COMMUN_PHYSIQUE,
            Parametre("--temperature-air", "288", "Temperature d'air fallback en K."),
        ),
    ),
    Scenario(
        titre="Grille globale sur 1 jour",
        module="modele4.rapide",
        description="Simulation vectorisee globale, cartes toutes les 4 heures.",
        parametres=(
            Parametre("--jours", "1", "Duree simulee en jours."),
            Parametre("--dt", "1800", "Pas interne en secondes."),
            Parametre("--sortie-heures", "4", "Frequence d'ecriture des cartes."),
            Parametre("--max-latitudes", "", "Sous-grille latitude ; vide = globale."),
            Parametre("--max-longitudes", "", "Sous-grille longitude ; vide = globale."),
            Parametre(
                "--output",
                "modele4/sorties/simulation_modele4_rapide.npz",
                "Fichier NPZ ecrit.",
            ),
            *COMMUN_PHYSIQUE,
            Parametre("--temperature-air", "288", "Temperature d'air fallback en K."),
        ),
    ),
    Scenario(
        titre="Grille globale sur 1 an",
        module="modele4.rapide",
        description="Simulation longue conseillee avec sortie quotidienne.",
        parametres=(
            Parametre("--jours", "365", "Duree simulee en jours."),
            Parametre("--dt", "1800", "Pas interne en secondes."),
            Parametre("--sortie-heures", "24", "Frequence d'ecriture des cartes."),
            Parametre("--max-latitudes", "", "Sous-grille latitude ; vide = globale."),
            Parametre("--max-longitudes", "", "Sous-grille longitude ; vide = globale."),
            Parametre("--output", "modele4/sorties/rapide_1an.npz", "Fichier NPZ ecrit."),
            *COMMUN_PHYSIQUE,
            Parametre("--temperature-air", "288", "Temperature d'air fallback en K."),
        ),
    ),
)


SCENARIOS_CLASSIQUE = (
    Scenario(
        titre="Mensuel global",
        module="modele4.modele4",
        description="12 cartes mensuelles avec recalcul radiatif local par cellule.",
        parametres=(
            Parametre("--mode", "mensuel", "Mode de sortie.", choix=("mensuel", "temporel")),
            Parametre("--dt", "1800", "Pas numerique en secondes."),
            Parametre("--max-latitudes", "", "Sous-grille latitude ; vide = globale."),
            Parametre("--max-longitudes", "", "Sous-grille longitude ; vide = globale."),
            Parametre(
                "--iterations-implicites",
                "4",
                "Iterations Newton par cellule.",
            ),
            Parametre(
                "--output",
                "modele4/sorties/simulation_modele4.npz",
                "Fichier NPZ ecrit.",
            ),
            *COMMUN_PHYSIQUE,
        ),
    ),
    Scenario(
        titre="Temporel petite grille",
        module="modele4.modele4",
        description="Integration pas-a-pas sur 4 x 8 cellules ; utile pour validation.",
        parametres=(
            Parametre("--mode", "temporel", "Mode de sortie.", choix=("mensuel", "temporel")),
            Parametre("--jours", "1", "Duree simulee en jours."),
            Parametre("--dt", "1800", "Pas numerique en secondes."),
            Parametre("--frequence-sortie-pas", "48", "Ecrit une carte tous les N pas."),
            Parametre("--max-latitudes", "4", "Nombre de latitudes gardees."),
            Parametre("--max-longitudes", "8", "Nombre de longitudes gardees."),
            Parametre(
                "--iterations-implicites",
                "4",
                "Iterations Newton par cellule.",
            ),
            Parametre("--output", "modele4/sorties/complet_dev.npz", "Fichier NPZ ecrit."),
            *COMMUN_PHYSIQUE,
        ),
    ),
    Scenario(
        titre="Une cellule, un pas",
        module="modele4.modele4",
        description="Test minimal : une cellule et un pas de 1800 s.",
        parametres=(
            Parametre("--mode", "temporel", "Mode de sortie.", choix=("mensuel", "temporel")),
            Parametre("--jours", "0.020833333333333332", "Duree ; ici 1 pas de 1800 s."),
            Parametre("--dt", "1800", "Pas numerique en secondes."),
            Parametre("--frequence-sortie-pas", "1", "Ecrit a chaque pas."),
            Parametre("--max-latitudes", "1", "Nombre de latitudes gardees."),
            Parametre("--max-longitudes", "1", "Nombre de longitudes gardees."),
            Parametre(
                "--iterations-implicites",
                "4",
                "Iterations Newton par cellule.",
            ),
            Parametre("--output", "modele4/sorties/complet_1cellule.npz", "Fichier NPZ ecrit."),
            *COMMUN_PHYSIQUE,
        ),
    ),
)


MOTEURS = (
    Moteur(
        titre="Rapide - modele4.rapide",
        resume="Moteur conseille pour travailler : vectorise, rapide, adapte aux longues runs.",
        technique=(
            "pre-calcule des champs mensuels avec le modele 3",
            "integre ensuite toute la grille avec numpy",
            "ne rappelle pas le modele 3 a chaque pas de temps",
        ),
        physique=(
            "atmosphere radiative mensuelle approximee pendant la simulation",
            "cycle jour/nuit conserve pour le court-onde",
            "LW montant et convection suivent T_surface en temps reel",
        ),
        usage=(
            "tests de parametres",
            "petites et grandes grilles",
            "simulations longues",
        ),
        scenarios=SCENARIOS_RAPIDE,
    ),
    Moteur(
        titre="Classique - modele4.modele4",
        resume="Moteur de reference : plus lent, plus direct physiquement.",
        technique=(
            "appelle le modele 3 dans la boucle de calcul",
            "recalcule les flux radiatifs cellule par cellule",
            "utilise un schema implicite avec iterations de Newton",
        ),
        physique=(
            "couplage radiatif local plus proche du modele 3",
            "flux calcules avec la temperature de surface courante",
            "toujours sans transport horizontal ni ocean dynamique",
        ),
        usage=(
            "test minimal sur une cellule",
            "validation sur petite grille",
            "comparaison avec le moteur rapide",
        ),
        scenarios=SCENARIOS_CLASSIQUE,
    ),
)


def est_oui(valeur: str) -> bool:
    return valeur.strip().lower() in {"o", "oui", "y", "yes", "1", "true"}


def invite(message: str) -> str:
    try:
        return input(message)
    except EOFError:
        return "q"


def pause() -> None:
    if sys.stdin.isatty():
        invite("\nEntree pour continuer...")


def valeurs_initiales(scenario: Scenario) -> dict[str, str]:
    return {parametre.option: parametre.valeur for parametre in scenario.parametres}


def construire_commande(scenario: Scenario, valeurs: dict[str, str]) -> list[str]:
    commande = [str(PYTHON), "-m", scenario.module]
    for parametre in scenario.parametres:
        valeur = valeurs.get(parametre.option, "").strip()
        if parametre.drapeau:
            if est_oui(valeur):
                commande.append(parametre.option)
            continue
        if valeur:
            commande.extend((parametre.option, valeur))
    return commande


def commande_affichee(commande: list[str]) -> str:
    affichage = list(commande)
    venv_python = RACINE_PROJET / ".venv" / "bin" / "python"
    try:
        if Path(affichage[0]).resolve() == venv_python.resolve():
            affichage[0] = "./.venv/bin/python"
    except OSError:
        pass
    return shlex.join(affichage)


def afficher_titre() -> None:
    print("\n" + "=" * 72)
    print("Modele 4 - lanceur TUI")
    print("=" * 72)
    print("Choisir d'abord le moteur, puis un cas d'usage courant.")
    print("Lancement conseille : ./.venv/bin/python -m modele4.lancer")


def afficher_menu() -> None:
    afficher_titre()
    print("\nMoteurs :")
    for indice, moteur in enumerate(MOTEURS, start=1):
        print(f"  {indice}. {moteur.titre}")
        print(f"     {moteur.resume}")
    print("\nCommandes : numero = choisir, c = comparer, p = parametres, q = quitter")


def afficher_liste(prefixe: str, lignes: tuple[str, ...]) -> None:
    for ligne in lignes:
        print(f"  {prefixe} {ligne}")


def afficher_moteur(moteur: Moteur) -> None:
    print(f"\n{moteur.titre}")
    print(moteur.resume)
    print("\nTechnique :")
    afficher_liste("-", moteur.technique)
    print("\nPhysique :")
    afficher_liste("-", moteur.physique)
    print("\nA utiliser pour :")
    afficher_liste("-", moteur.usage)


def afficher_comparaison_moteurs() -> None:
    afficher_titre()
    print("\nDifference essentielle :")
    print("  Rapide    : pre-calcule l'atmosphere mensuelle, puis integre vite la grille.")
    print("  Classique : recalcule la colonne radiative locale dans la boucle de calcul.")
    for moteur in MOTEURS:
        afficher_moteur(moteur)
    pause()


def afficher_menu_scenarios(moteur: Moteur) -> None:
    afficher_titre()
    afficher_moteur(moteur)
    print("\nCas d'usage :")
    for indice, scenario in enumerate(moteur.scenarios, start=1):
        print(f"  {indice}. {scenario.titre}")
        print(f"     {scenario.description}")
    print("\nCommandes : numero = choisir, p = parametres, r = retour, q = quitter")


def afficher_aide_parametres() -> None:
    afficher_titre()
    print("\nParametres utiles :")
    lignes = (
        ("--jours", "duree simulee en jours."),
        ("--dt", "pas de temps interne en secondes."),
        ("--sortie-heures", "frequence de sauvegarde du moteur rapide."),
        ("--mode", "mensuel = 12 cartes ; temporel = pas-a-pas."),
        ("--max-latitudes/--max-longitudes", "sous-grille de test ; vide = globale."),
        ("--output", "chemin du fichier .npz produit."),
        ("--co2", "concentration CO2 en ppm."),
        ("--temperature-initiale", "temperature forcee en K ; vide = donnees."),
        ("--convection", "aucune, forcee, naturelle ou toutes."),
        ("--facteur-latent", "0 coupe le latent ; 1 garde le reglage standard."),
        ("--vent", "vent en m/s pour la convection forcee."),
        ("--rzsm-csv", "humidite du sol ; vide = source par defaut."),
        ("--no-progress", "coupe les barres de progression."),
    )
    for option, description in lignes:
        print(f"  {option:<31} {description}")
    pause()


def afficher_parametres(scenario: Scenario, valeurs: dict[str, str]) -> None:
    print("\nParametres :")
    for indice, parametre in enumerate(scenario.parametres, start=1):
        valeur = valeurs.get(parametre.option, "")
        valeur_affichee = valeur if valeur else "(vide)"
        print(f"  {indice:>2}. {parametre.option:<24} {valeur_affichee:<28} {parametre.description}")


def modifier_parametres(scenario: Scenario, valeurs: dict[str, str]) -> None:
    while True:
        afficher_titre()
        print(f"\nScenario : {scenario.titre}")
        afficher_parametres(scenario, valeurs)
        print("\nNumero = modifier, t = tout parcourir, r = retour")
        choix = invite("> ").strip().lower()
        if choix in {"r", "retour", ""}:
            return
        if choix == "t":
            for parametre in scenario.parametres:
                modifier_un_parametre(parametre, valeurs)
            return
        if not choix.isdigit():
            print("Choix invalide.")
            pause()
            continue
        indice = int(choix)
        if not 1 <= indice <= len(scenario.parametres):
            print("Numero hors liste.")
            pause()
            continue
        modifier_un_parametre(scenario.parametres[indice - 1], valeurs)


def modifier_un_parametre(parametre: Parametre, valeurs: dict[str, str]) -> None:
    actuel = valeurs.get(parametre.option, "")
    print(f"\n{parametre.option}")
    print(f"  {parametre.description}")
    if parametre.choix:
        print("  Choix : " + ", ".join(parametre.choix))
    if parametre.drapeau:
        print("  Valeurs acceptees : oui/non")
    print("  Entree garde la valeur ; '-' vide le champ.")
    nouvelle = invite(f"  Nouvelle valeur [{actuel if actuel else 'vide'}] > ").strip()
    if nouvelle == "":
        return
    if nouvelle in {"-", "vide"}:
        valeurs[parametre.option] = ""
        return
    if parametre.choix and nouvelle not in parametre.choix:
        print("  Valeur ignoree : choix non reconnu.")
        pause()
        return
    valeurs[parametre.option] = nouvelle


def lancer_commande(commande: list[str]) -> int:
    print("\nExecution depuis : " + str(RACINE_PROJET))
    print(commande_affichee(commande))
    print("-" * 72)
    sys.stdout.flush()
    try:
        resultat = subprocess.run(commande, cwd=RACINE_PROJET, check=False)
    except KeyboardInterrupt:
        print("\nExecution interrompue.")
        return 130
    print("-" * 72)
    print(f"Code retour : {resultat.returncode}")
    pause()
    return int(resultat.returncode)


def gerer_scenario(scenario: Scenario) -> bool:
    valeurs = valeurs_initiales(scenario)
    while True:
        afficher_titre()
        print(f"\nScenario : {scenario.titre}")
        print(scenario.description)
        afficher_parametres(scenario, valeurs)
        commande = construire_commande(scenario, valeurs)
        print("\nCommande generee :")
        print("  " + commande_affichee(commande))
        print("\nActions : entree = lancer, m = modifier, r = retour, q = quitter")
        choix = invite("> ").strip().lower()
        if choix == "":
            lancer_commande(commande)
            return True
        if choix in {"m", "modifier"}:
            modifier_parametres(scenario, valeurs)
            continue
        if choix in {"r", "retour"}:
            return False
        if choix in {"q", "quit", "quitter"}:
            raise SystemExit(0)
        print("Choix invalide.")
        pause()


def gerer_moteur(moteur: Moteur) -> None:
    while True:
        afficher_menu_scenarios(moteur)
        choix = invite("> ").strip().lower()
        if choix in {"r", "retour"}:
            return
        if choix in {"q", "quit", "quitter"}:
            raise SystemExit(0)
        if choix in {"p", "parametres"}:
            afficher_aide_parametres()
            continue
        if not choix.isdigit():
            print("Choix invalide.")
            pause()
            continue
        indice = int(choix)
        if not 1 <= indice <= len(moteur.scenarios):
            print("Numero hors liste.")
            pause()
            continue
        gerer_scenario(moteur.scenarios[indice - 1])


def main() -> int:
    while True:
        afficher_menu()
        choix = invite("> ").strip().lower()
        if choix in {"q", "quit", "quitter"}:
            return 0
        if choix in {"c", "comparer", "a", "aide", "h", "help"}:
            afficher_comparaison_moteurs()
            continue
        if choix in {"p", "parametres"}:
            afficher_aide_parametres()
            continue
        if not choix.isdigit():
            print("Choix invalide.")
            pause()
            continue
        indice = int(choix)
        if not 1 <= indice <= len(MOTEURS):
            print("Numero hors liste.")
            pause()
            continue
        gerer_moteur(MOTEURS[indice - 1])


if __name__ == "__main__":
    raise SystemExit(main())
