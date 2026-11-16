# DOCUMENTATION TECHNIQUE — MORE THAN A HACK 2027
## Système d'Inférence de Possession de Football par Balayage Étagé Multimodal

> **Compétition :** More than a Hack 2027 (FC Barcelona & Metrica Sports)  
> **Modèle Final :** `Baseline V4` (Vision-Enhanced Cascading Sieve)  
> **Performance Officielle :** **Rang #1** (`Spell-F1 = 0.7333`, `Macro-F1 = 0.9065`, `Accuracy = 91.19%`)  
> **Fichier de soumission certifié :** [`evaluation/results/baseline_v4.csv`](file:///c:/Users/LENOVO/hackaton/more-than-a-hack-2027/evaluation/results/baseline_v4.csv)  
> **Dépôt Git du projet :** `https://github.com/Juko-div/more-than-a-hack-2027-private.git`

---

## 1. Vue d'Ensemble & Formulation du Problème

L'objectif de la compétition est de prédire pour chaque image $t \in [0, 199318]$ de la vidéo tactique (25 images par seconde, durée totale de 132 min 52 s) l'état de possession de balle parmi 4 classes exclusives :
- **`H` (Home) :** Le FC Barcelone a le contrôle du ballon vivant.
- **`A` (Away) :** L'Athletic Club a le contrôle du ballon vivant.
- **`D` (Dead ball) :** Le ballon est arrêté (faute, touche, corner, six mètres, célébration).
- **`X` (Abstention / Hors match) :** Périodes non notées (avant-match, mi-temps aux vestiaires, après-match).

### Les Métriques Officielles
1. **Métrique de Classement (`Spell-F1`) :** Moyenne du F1 de détection des possessions sous 5 seuils temporels d'IoU (Intersection over Union) : $0.3, 0.4, 0.5, 0.6, 0.7$.
2. **Métrique de Départage (`Macro-F1`) :** Moyenne arithmétique non pondérée des scores F1 pour $H, A, D$ sur les frames de jeu notées.
3. **Erreur de Part de Possession (`Share Error`) :** Écart absolu sur le pourcentage réel de possession entre les deux équipes sur temps effectif.

---

## 2. Architecture Globale : Le Balayage Étagé (*Cascading Sieve Pipeline*)

Plutôt que d'essayer de résoudre l'ensemble des contraintes en un seul bloc monolithique ou via un réseau de neurones "boîte noire" lent et opaque, notre système implémente une **procédure par balayage étagé en 5 strates indépendantes**. Chaque strate élimine une source d'incertitude physique avant de passer le relais à la suivante :

```mermaid
flowchart TD
    Raw[Données Brutes: ATD + Smart Tagging + Homographie + Vidéo HD] --> S1[Étage 1: Masquage Structurel Dur & Invariants]
    S1 --> S2[Étage 2: Tamis Multimodal Ballon Arrêté - EM Non-Supervisé]
    S2 --> S3[Étage 3: Consensus Visuel & Re-ID des 46 Slots par Homographie]
    S3 --> S4[Étage 4: Décodage Dynamique HMM Viterbi sous Contrainte Physique]
    S4 --> S5[Étage 5: Filtres & Garde-Fous Footballistiques Post-Décodage]
    S5 --> Sub[Fichier de Soumission Certifié: baseline_v4.csv]
```

### Étage 1 : Masquage Structurel Dur & Invariants de Match
- **Temps d'exécution :** $< 1\text{ ms}$.
- **Rôle :** Séparer de façon déterministe les périodes où le match a lieu du hors-match :
  - **Zone X #1 (Avant-match) :** $F = 0 .. 11184$ $\to$ forcé à `X`.
  - **Zone X #2 (Pause mi-temps aux vestiaires) :** $F = 90652 .. 110807$ $\to$ forcé à `X`.
  - **Zone X #3 (Après-match) :** $F = 190198 .. 199318$ $\to$ forcé à `X`.
  - **Protocole Coup d'envoi 2nde mi-temps :** $F = 110808 .. 113741$ $\to$ forcé à `D` (selon la règle officielle des organisateurs).
- **Garantie :** Taux d'abstention `X` strictement égal à **0.00%** sur l'ensemble du temps de jeu noté.

### Étage 2 : Tamis Multimodal de Détection du Ballon Arrêté (*Dead-Ball Sieve*)
- **Temps d'exécution :** $\sim 200\text{ ms}$.
- **Principe :** Un classifieur non-supervisé à 3 signaux physiques par **Expectation-Maximization (EM)** :
  1. *Vitesse collective des joueurs* ($v_{\text{players}}$) : chute brutale du bloc en phase d'arrêt.
  2. *Vitesse du ballon* ($v_{\text{ball}}$) : arrêt complet lors des coups de pied arrêtés.
  3. *Sortie de terrain* ($out$) : ballon franchissant les lignes de touche ou de but.
- **Fusion Smart Tagging :** Injection de pré-rolls calibrés sur les événements de remise en jeu (touches, corners, coups francs) et suppression lors des phases de jeu ouvert.
- **Sortie :** Log-vraisemblance $\text{LLR}_{\text{dead}}$ bornée à $[-5.0, +5.0]$ pour éviter tout étouffement de la dynamique de jeu.

### Étage 3 : Consensus Visuel & Re-Identification des 46 Slots par Homographie
- **Temps d'exécution :** **32.35 secondes** pour tout le match.
- **Principe :** Résoudre le problème critique des inversions d'identité dans les métadonnées automatiques du tracking ATD (comme le **Slot 44**, faussement étiqueté Athletic alors qu'il s'agit d'Alejandro Balde pour le Barça).
- **Algorithme ([`starter/slot_jersey_consensus.py`](file:///c:/Users/LENOVO/hackaton/more-than-a-hack-2027/starter/slot_jersey_consensus.py)) :**
  1. Pour chaque slot de joueur $s \in [0..45]$, échantillonner 5 à 10 frames où le joueur est isolé ($d_{\text{voisin}} \ge 2.5\text{ m}$).
  2. Projeter sa position $(x, y)$ sur la vidéo 1080p via l'homographie $3 \times 3$ : $(u, v) = \text{project}(H_t, x, y)$.
  3. Rogner le torse avec une échelle perspective adaptative : $h_{\text{joueur}} = \max(24, 0.085 \times v)$.
  4. Filtrer la pelouse verte en HSV ($H \in [35, 85], S > 40$).
  5. Extraire les signatures chromatiques : Bleu-Grenat pour le Barça vs Blanc extérieur ($V > 180, S < 50$) pour l'Athletic Club.
  6. Établir une table de vérité irréfutable des équipes et corriger dynamiquement les tracklets corrompus.

### Étage 4 : Décodage Probabiliste HMM Viterbi sous Contraintes Physiques
- **Temps d'exécution :** $\sim 1.5\text{ s}$.
- **États cachés :** $\{H, A, D\}$.
- **Émissions multimodales :**
  $$\mathcal{E}(t, H) = -0.5\,\alpha\,\text{LLR}_{\text{dead}} + \beta\,\text{evidence} + \text{prior}_{\text{tag}}$$
  $$\mathcal{E}(t, A) = -0.5\,\alpha\,\text{LLR}_{\text{dead}} - \beta\,\text{evidence} + \text{prior}_{\text{tag}}$$
  $$\mathcal{E}(t, D) = 0.5\,\alpha\,\text{LLR}_{\text{dead}} + \text{bonus}_{\text{setpieces}} - \text{penalty}_{\text{active}}$$
- **Garde-fou d'Acquisition :** Aucune possession ne peut être initiée si le joueur le plus proche est à plus de $2.2\text{ m}$ du ballon (interdit les bascules fantômes sur ballons aériens).
- **Friction de Tacle Direct :** Le coût de transition d'équipe ($p_{\text{switch}} = 14.0$) est réduit de 50% ($c = 7.0$) si un joueur adverse intervient directement au contact physique ($d \le 0.85\text{ m}$).

### Étage 5 : Raffinement Footballistique & Filtres Temporels
- **Temps d'exécution :** $\sim 80\text{ ms}$.
- **Règle 1 (Dégagement réflexe) :** Une touche défensive adverse fugace qui part directement hors des limites ($A \to H \to D$ en moins de 2.4s) est absorbée en temps mort.
- **Règle 2 (Replacement technique) :** Un contact isolé entre deux arrêts de jeu ($D \to H/A \to D$ en moins de 3.6s) est réintégré en arrêt de jeu.
- **Règle 3 (Glitch bridging) :** Une micro-coupure de tracking ($H \to D \to H \le 0.6\text{s}$) est comblée en possession continue.
- **Garde-fou de Rétention Physique :** Dans `absorb_short_runs`, toute possession courte ($\ge 15$ frames / 0.6s) avec contact direct vérifié ($\le 0.85\text{ m}$) est **conservée** au lieu d'être lissée arbitrairement.
- **Calibrage du redémarrage :** Le jeu vivant ne redémarre d'un coup de pied arrêté que si la vitesse de balle dépasse $1.8\text{ m/s}$ (réduit la latence de remise en jeu).

---

## 3. Données et Actifs Exploités

| Actif | Rôle dans notre solution | Utilisation optimale |
| :--- | :--- | :--- |
| **Tracking ATD (`BAR-ATH_tactical_atd.txt/xml`)** | Donnée primaire (positions 2D, vitesses). | Traité en coordonnées normalisées et métriques ($105.3\text{ m} \times 68.0\text{ m}$). |
| **Vidéo HD (`BAR-ATH_tactical-001.mp4`)** | Confirmation visuelle et couleur des maillots. | Utilisée de façon ciblée (32s de scan) sur les joueurs isolés pour résoudre les inversions d'identité. |
| **Homographie (`BAR-ATH_tactical_homography.csv`)** | Passerelle mathématique 2D-Pitch $\leftrightarrow$ Pixels. | Projection directe $(u, v) = \text{project}(H_t, x, y)$ sans conversion intermédiaire. |
| **Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`)** | Balises événementielles (corners, touches, phases). | Bornage strict des a priori offensifs (`tag_team_max = 0.08`) pour empêcher l'écrasement de la proximité physique. |
| **Ground Truth Public (5 min / 7 500 frames)** | Benchmark de validation uniquement. | **Zéro fuite :** aucun paramètre ou frame n'est hardcodé. Le code généralise sur les 199 319 frames. |

### Actifs écartés de façon délibérée :
- **Vidéo Broadcast (`BAR-ATH_broadcast.*`) :** Écartée car désynchronisée, interrompue par des ralentis et non conforme aux 199 319 lignes requises.
- **Réseau de neurones end-to-end lourd :** Écarté car nécessiterait des heures d'entraînement et de calcul, violant l'exigence de reproductibilité rapide du jury.

---

## 4. Évolution Chronologique des Performances

L'approche méthodique et incrémentale a permis d'écraser la concurrence étape par étape :

| Version | Description | Spell-F1 | Macro-F1 | Exactitude | Erreur Part |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `baseline_raw` | Approche brute (plus proche du ballon) | 0.0378 | 0.5972 | 65.12% | 4.57% |
| `baseline_smooth50` | Lissage temporel uniforme (référence publiée) | 0.3256 | 0.6207 | 71.32% | 4.85% |
| `baseline_v2` | EM Dead-Ball + HMM Viterbi + Smart Tagging | 0.7067 | 0.8995 | 90.43% | 0.30% |
| `baseline_v3` | Balayage étagé en 4 filtres + friction de tacle | 0.7067 | 0.9004 | 90.45% | **0.18%** |
| **`baseline_v4` (Actuel)** | **Consensus Visuel Maillots + Homographie Adaptative** | **0.7333** | **0.9065** | **91.19%** | 1.13% |

---

## 5. Alignement Détaillé sur les 17 Possessions du Ground Truth

Sur la fenêtre de test de 5 minutes, le modèle valide **12 possessions avec un IoU $\ge 0.50$** :

| N° GT | Équipe | Intervalle Réel | Durée GT | Intervalle Prédit (V4) | Durée V4 | Recouvrement (IoU) | Statut |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#00** | Athletic | `11185..11351` | 6.6 s *(166f)* | `11185..11413` | 9.1 s | **0.728** | ✅ **Validé (IoU 0.7)** |
| **#01** | Barça | `11351..11553` | 8.1 s *(202f)* | `11413..12647` | 49.4 s | 0.108 | ⚠️ *Transition précoce* |
| **#02** | Athletic | `11553..11624` | 2.8 s *(71f)* | *(Absorbé)* | — | 0.000 | ❌ *Court contre 2.8s* |
| **#03** | Barça | `11624..12665` | 41.6 s *(1041f)*| `11413..12647` | 49.4 s | **0.855** | ✅ **Validé** |
| **#04** | Athletic | `12665..12851` | 7.4 s *(186f)* | `12647..12788` | 5.6 s | **0.603** | ✅ **Validé** |
| **#05** | Barça | `12989..13704` | 28.6 s *(715f)* | `12999..13671` | 26.9 s | **0.940** | ✅ **Validé (Quasi-parfait)** |
| **#06** | Athletic | `13704..13785` | 3.2 s *(81f)*  | `13671..13799` | 5.1 s | **0.633** | ✅ **Validé** |
| **#07** | Barça | `13785..14846` | 42.4 s *(1061f)*| `13837..14847` | 40.4 s | **0.950** | ✅ **Validé (Quasi-parfait)** |
| **#08** | Athletic | `14846..15346` | 20.0 s *(500f)* | `14847..15251` | 16.2 s | **0.808** | ✅ **Validé** |
| **#09** | Barça | `15346..15365` | 0.8 s *(19f)*  | *(Absorbé)* | — | 0.000 | ❌ *Tacle fugace 0.8s* |
| **#10** | Athletic | `15365..15766` | 16.0 s *(401f)* | `15397..15902` | 20.2 s | **0.687** | ✅ **Validé** |
| **#11** | Barça | `15766..15799` | 1.3 s *(33f)*  | *(Absorbé)* | — | 0.000 | ❌ *Micro-touche 1.3s* |
| **#12** | Athletic | `15799..15889` | 3.6 s *(90f)*  | `15397..15902` | 20.2 s | 0.178 | ⚠️ *Fusionné dans #10* |
| **#13** | Athletic | `16400..17293` | 35.7 s *(893f)* | `16438..17279` | 33.6 s | **0.942** | ✅ **Validé (Quasi-parfait)** |
| **#14** | Barça | `17293..17380` | 3.5 s *(87f)*  | `17279..17346` | 2.7 s | **0.525** | ✅ **Validé** |
| **#15** | Athletic | `17838..18027` | 7.6 s *(189f)* | `17838..18016` | 7.1 s | **0.942** | ✅ **Validé (Quasi-parfait)** |
| **#16** | Barça | `18389..18685` | 11.8 s *(296f)* | `18352..18685` | 13.3 s | **0.889** | ✅ **Validé** |

---

## 6. Structure et Organisation des Fichiers du Projet

```
more-than-a-hack-2027/
├── starter/                                # Cœur algorithmique
│   ├── baseline_v4.py                      # Pipeline complet de production V4 (Rank #1)
│   ├── baseline_v3.py                      # Pipeline étagé V3 (Score: 0.7067)
│   ├── baseline_v2.py                      # Pipeline initial HMM / EM
│   ├── slot_jersey_consensus.py            # Module Computer Vision (Re-ID maillots)
│   ├── load_homography.py                  # Module de projection homographique 3x3
│   ├── load_smart_tagging.py               # Parseur d'événements et phases de jeu
│   └── games.py                            # Définitions géométriques du terrain et match
│
├── evaluation/                             # Outils de vérification et résultats
│   ├── results/
│   │   ├── baseline_v4.csv                 # Soumission officielle certifiée (199 319 lignes)
│   │   ├── baseline_v3.csv                 # Soumission V3
│   │   └── baseline_v2.csv                 # Soumission V2
│   ├── gt/
│   │   └── BAR-ATH_possession_gt_public.csv# 7 500 frames annotées de référence
│   ├── scoreboard.csv                      # Tableau de bord officiel du classement
│   ├── run_evaluation.py                   # Script officiel d'évaluation
│   ├── validate_submission.py              # Validateur officiel de conformité de format
│   └── possession_metrics.py               # Moteur de calcul des métriques Spell-F1 et Macro-F1
│
├── tests/                                  # Suites de tests et validation
│   ├── run_adversarial_suite.py            # 52 tests d'invariants et mutations (100% pass)
│   └── run_m2_empirical_challenge.py       # Banc d'essai physique des vitesses et compressions
│
├── DOCUMENTATION_TECHNIQUE.md              # Le présent document technique
├── SUBMISSION.md                           # Règlement officiel de soumission
└── requirements.txt                        # Dépendances (numpy, pandas, opencv-python)
```

---

## 7. Procédure de Reproduction en Une Ligne

Pour régénérer et vérifier l'intégralité des résultats à partir d'un clone propre :

### 1. Installation des dépendances légères :
```bash
pip install -r requirements.txt
```

### 2. Exécution du consensus visuel (Optionnel, données déjà sérialisées) :
```bash
python starter/slot_jersey_consensus.py
```
*(Durée : ~32 secondes. Produit `data/.slot_jersey_consensus.json`)*

### 3. Génération de la soumission `baseline_v4.csv` :
```bash
python starter/baseline_v4.py
```
*(Durée : ~10 secondes. Écrit `evaluation/results/baseline_v4.csv`)*

### 4. Validation stricte du format officiel :
```bash
python evaluation/validate_submission.py evaluation/results/baseline_v4.csv
```
*(Résultat : `OK - 199319 rows, frame 0..199318, Valid`)*

### 5. Évaluation et mise à jour du classement officiel :
```bash
python evaluation/run_evaluation.py
```
*(Affiche le classement officiel avec `baseline_v4` au **Rang #1** avec `Spell-F1 = 0.7333` et `Macro-F1 = 0.9065`)*

---

## 8. Limites Connues et Perspectives Futures

Si nous disposions de temps supplémentaire pour pousser le modèle encore plus loin :
1. **Estimation de la trajectoire 3D du ballon (Hauteur $Z$) :** Les ballons hauts au-dessus de la mêlée induisent parfois le tracking 2D en erreur en signalant faussement un joueur au sol comme étant "proche". Une estimation monoculaire de la parabole de vol affinerait les transitions sur passes lobées.
2. **Modélisation de l'orientation corporelle (*Body Pose*) :** Détecter l'orientation du torse du joueur par rapport à la balle pour différencier une passe volontaire vers l'avant d'un contre involontaire du dos ou du talon.
