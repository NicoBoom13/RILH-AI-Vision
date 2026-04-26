# Stage 3.a (Entities) — Design doc

> Originally written as "Phase 1.6 design doc" before the project's
> phase-numbering settled. The stage was Stage 1.d in the previous
> structure (entities under Phase 1), and is now Stage 3.a under
> Phase 3 (Entity recognition). The rink-aware on-ice/off-ice filter
> mentioned below is the work that promoted rink calibration into the
> new Phase 2.

## 1. Objectif

Partir d'une `tracks.json` Phase 1 fragmentée (ex. test12 : 250 tracks pour
~12 entités réelles) et produire une table d'**entités stables** : 1 entité
= 1 joueur réel = N tracks fusionnés.

Livrable cible sur test12 :
- **~12 entités** au lieu de 250 tracks (10 joueurs + ~2 arbitres)
- Chaque entité porte `team_id`, `jersey_number` (si OCR réussi),
  `is_goaltender`, la liste des `track_ids` fusionnés, et sa couverture
  temporelle (frame_start → frame_end)

Sans ça, les stats Phase 7 (distance, vitesse, heatmap) sont cassées parce
qu'on additionne des morceaux de joueurs différents sous un même ID.

## 2. Grille de lecture (3 horizons — ta synthèse)

| Horizon | Échelle | Signaux | Rôle | État dans notre pipeline |
|---|---|---|---|---|
| **Court** | frame à frame | Kalman + IoU + apparence | associer détections ↔ tracklets actives | ✅ **ByteTrack** (Phase 1) |
| **Moyen** | quelques secondes | Re-ID sport + couleur équipe | recoller tracklets après occlusion | ❌ **à construire en Phase 1.6** |
| **Long** | séquence / match | galerie embeddings + OCR numéro + contraintes équipe | garantir persistance d'identité + corriger | 🟡 **partiel** : OCR + merge-par-numéro déjà dans Phase 6, mais pas de galerie ni de contrainte d'équipe |

**Phase 1.6 = couche moyen-terme + complément long-terme** (contrainte
équipe hard + intégration OCR comme signal fort). La galerie complète
(mise-à-jour temps-réel sur un match entier) sera Phase 8 quand on voudra
du streaming live — pas pour maintenant.

## 3. Périmètre Phase 1.6

**Inclus** :
1. Extraction d'un **embedding d'apparence** (vecteur de ~512 réels) par
   track — on résume le look visuel de chaque morceau de tracking
2. Fusion des tracks via **clustering sous contraintes** (équipe +
   non-overlap temporel + max N par équipe)
3. **Intégration du signal OCR** déjà produit par `phase6_identify.py`
   (deux tracks avec le même numéro = très fort candidat à fusionner)
4. Sortie : **`tracks_entities.json`** — nouveau fichier qui complète
   `tracks.json` (on ne touche pas à Phase 1)
5. Adaptation `phase6_annotate.py` pour afficher la couleur + le numéro de
   **l'entité** (pas du track fragmenté)

**Exclus** (reportés plus tard) :
- Online / streaming (on reste en batch post-match)
- Galerie cumulative sur match entier (on traite clip par clip)
- Correction manuelle via UI (ça viendra avec Phase 7 web)
- Gestion multi-caméras (Phase 7+)

## 4. Composants — explications et choix

### 4.1 Embedding d'apparence (le "quoi on reconnaît")

**Ce que c'est** : un réseau de neurones qui prend une image de joueur en
entrée et sort un vecteur (typiquement 512 réels). Deux crops du *même*
joueur produisent des vecteurs *proches* dans cet espace ; deux joueurs
différents produisent des vecteurs *éloignés*. On mesure "proche" avec la
**distance cosinus** (angle entre vecteurs).

**Pourquoi pas les features YOLO** : on a testé en test11 (BoT-SORT avec
ReID activé qui utilise les features YOLO). Résultat : ne discrimine pas
entre joueurs de la même équipe (mêmes maillots). Logique — YOLO est
entraîné à *détecter* des personnes, pas à les *distinguer*.

**Trois options** :

| Option | Taille | Vitesse | Qualité | Dépendances |
|---|---|---|---|---|
| **OSNet** (torchreid) | ~30 Mo | rapide | bonne (entraîné pour Re-ID personne) | `torchreid` |
| **CLIP** (openai) | ~200 Mo | moyenne | très bonne mais généraliste | déjà indirectement via torch |
| **Hugging Face DINOv2-small** | ~90 Mo | moyenne | bonne, auto-supervisée | `transformers` |

**Recommandation** : **OSNet**. Spécifiquement conçu pour Re-ID personne,
~6× plus léger que CLIP, et l'état de l'art Re-ID sport (SoccerNet benchmark)
utilise des variantes d'OSNet. Si torchreid est capricieux à installer
(pas rare), fallback sur CLIP (via `clip-by-openai` ou `open_clip_torch`).

### 4.2 Agrégation par track (le "résumé d'un tracklet")

**Problème** : un track a 5–500 crops. On veut *un seul* vecteur par track.

**Options** :
- **Moyenne** des embeddings des crops — simple mais sensible aux outliers
  (crop flou, mal cadré)
- **Médoïde** — on choisit le crop dont l'embedding est le plus central
  dans le cloud du track. Robuste aux outliers.
- **Distribution** — on garde tous les crops et on compare des
  distributions (ex. Earth Mover's). Complexe, peu de gain attendu à
  cette échelle.

**Recommandation** : **médoïde** sur 5–10 crops sélectionnés (top-conf
détections, même logique que phase6_identify).

### 4.3 Construction du graphe (le "qui peut être fusionné avec qui")

On modélise la fusion comme un graphe :
- **Nœuds** = tracks (250 dans test12)
- **Arêtes** = paires `(track_a, track_b)` *éligibles* à la fusion

Une paire est éligible si :
- **Même team_id** (de Phase 1.5)
- **Pas de chevauchement temporel** (si les deux tracks existent simultanément
  au frame K, ils sont par définition deux joueurs différents)
- **Vote confidence team ≥ 0.67** sur les deux (sinon on n'a pas confiance
  dans leur label d'équipe, pas fiable pour merger)

Poids de l'arête :

```
w(a, b) = α · cos_sim(embedding_a, embedding_b)
        + β · 1_{same_jersey_number(a, b)}
        + γ · 1_{same_is_goaltender(a, b)}
```

avec α=1.0, β=10 (OCR très fort quand il existe), γ=0.5 (les gardiens
préfèrent se merger entre eux dans leur équipe).

### 4.4 Clustering sous contraintes (le "comment on fusionne")

**Option A — greedy merge itératif** (ma reco)
1. Trier les arêtes par poids décroissant
2. Pour chaque arête `(a, b)` :
   - Si fusionner les deux clusters respecte les contraintes
     (non-overlap total dans le cluster fusionné, taille ≤ max_per_team),
     fusionner.
   - Sinon, passer.
3. Fin quand plus d'arête valide.

Simple, déterministe, rapide. Utilisé dans plusieurs papiers de Re-ID
sport post-hoc.

**Option B — spectral clustering + projection sur contraintes** — plus
joli théoriquement mais implémentation plus lourde et pas de gain clair
à cette échelle (250 nœuds).

**Recommandation** : **Option A**.

### 4.5 Contraintes dures

Roller inline hockey : 4 skaters + 1 gardien par équipe, + 1–2 arbitres.

**Contrainte par équipe** (écrite dans le merge) :
```
max_skaters_per_team = 4    # sauf si remplacement → à assouplir plus tard
max_goalies_per_team = 1
```

Si une équipe a eu des *rotations* sur le clip (un skater remplacé), le
nombre réel de skaters dépasse 4. Pour un clip de 30 s pas de rotation ;
pour un match entier, il faudra relâcher à `max_skaters_per_team = 8`
(roster entier) et accepter quelques entités supplémentaires.

**Arbitres** : HockeyAI les étiquette "player" sur le roller (vu dans
test12) → ils se faufilent dans les contraintes d'équipe et occupent des
slots qui devraient aller à de vrais joueurs. Pour v1 on accepte : les
arbitres vont former quelques entités "parasites" au-delà du budget
d'équipe. À corriger via détection de rayures plus tard (piste 5).

### 4.6 Intégration du signal OCR

`phase6_identify.py` produit déjà `tracks_identified.json` avec un numéro
par track quand c'est lisible. Sur test12 : 26/250 tracks numérotés, 16
numéros distincts.

**Utilisation en Phase 1.6** :
- Les paires de tracks avec le **même numéro** reçoivent un **gros bonus
  de poids** (β=10) → quasi-sûr d'être fusionnées en premier.
- Par construction, la fonction `merge_tracks_by_number` de Phase 6 fait
  déjà ça, mais sans contrainte d'équipe. Phase 1.6 l'intègre proprement
  en plus du signal embedding.

**Effet attendu sur test12** : les 4 groupes déjà trouvés par Phase 6
(#5, #7×2, #10) deviennent des **seeds** forts pour le clustering. Ils
attirent les autres fragments non-numérotés de ces joueurs via la
similarité d'embedding.

### 4.7 Schéma de sortie

Nouveau fichier `runs/testNN/tracks_entities.json` :

```json
{
  "source_tracks": "runs/test12/tracks.json",
  "source_teams":  "runs/test12/tracks_teams.json",
  "source_ids":    "runs/test12/tracks_identified.json",
  "n_entities": 14,
  "entities": {
    "0": {
      "track_ids": [42, 186, 791, 886, 1062],
      "team_id": 1,
      "team_confidence": 0.92,
      "is_goaltender": false,
      "jersey_number": "7",
      "jersey_confidence": 0.58,
      "first_frame": 12,
      "last_frame": 1782,
      "total_frames_covered": 1203,
      "coverage_pct": 66.8
    },
    "...": "..."
  },
  "unmatched_track_ids": [15, 33, 201, ...]
}
```

`unmatched_track_ids` = les tracks qui n'ont pas pu être agrégés
(vote_confidence équipe trop basse, embedding aberrant, etc.). Ils
restent utilisables individuellement mais sont flaggés comme "bruit".

### 4.8 Adaptations downstream

**Obligatoire** — `phase6_annotate.py` v2 : lit `tracks_entities.json` si
présent et affiche la couleur + numéro de l'entité (au lieu du track).
Le track → entity mapping est `dict[track_id → entity_id]` construit à
la volée.

**Optionnel plus tard** — Phase 2 follow-cam pourrait pondérer le cadrage
par entité plutôt que par track (plus stable).

## 5. Flux de données

```
     Phase 1                       Phase 1.5              Phase 6 identify
tracks.json              tracks_teams.json          tracks_identified.json
     │                       │                              │
     └─────────┬─────────────┴──────────────────────────────┘
               ▼
        Phase 1.6
        tracks_entities.json
               │
               ▼
       Phase 6 annotate v2
       annotated_entities.mp4
```

Phase 1.6 ne consomme RIEN de nouveau côté modèles pré-entraînés — juste
OSNet (ou CLIP en fallback). Aucune donnée annotée nécessaire.

## 6. Validation

**Métriques quantitatives** (affichées en fin d'exécution) :
- `n_entities` : doit converger vers ~12–14 sur test12
- `coverage_pct` de la plus grosse entité : un vrai joueur qui joue tout
  le clip devrait donner une entité à 60–90 %. Si max < 30 %, c'est qu'on
  sur-fragmente.
- `n_unmatched` : idéalement < 50 (sur 250 tracks au départ, on tolère
  que les tracks courts sans signal restent isolés)
- Runtime : cible < 2 min sur test12 (CPU + embedding batchable)

**Métrique qualitative** :
- **Spot-check manuel** : pour les 5 plus grosses entités, ouvrir
  `annotated_entities.mp4` et vérifier que la boîte de couleur et le
  numéro restent sur le même joueur physique d'un bout à l'autre de son
  apparition à l'écran. C'est le test qui compte vraiment.

**Critère de succès** v1 sur test12 :
- ≥ 10 entités dans le budget roster (2 équipes × 4 skaters + 2 gardiens
  = 10 "vraies" entités attendues)
- 80 % des tracks source fusionnés dans une entité (les 20 % restants =
  arbitres, tracks très courts, bruits)
- 0 violation de contrainte : aucune entité avec un overlap temporel ou
  un team_id mixte

## 7. Questions ouvertes (à trancher)

Ces points méritent ton avis rapide avant que je code. Sinon je pars avec
mes recommandations par défaut.

| # | Question | Ma reco | Alternative |
|---|---|---|---|
| Q1 | Quel embedding ? | **OSNet via torchreid** | CLIP, DINOv2 |
| Q2 | Agrégation par track | **Médoïde sur 5–10 crops** | Moyenne, distribution |
| Q3 | Contrainte équipe | **Max 5 / équipe** (4 sk + 1 g) pour un clip 30s. Relâcher à 8 sur match complet | Max illimité (juste un prior) |
| Q4 | Non-overlap temporel | **Strict** (aucune frame partagée) | ≤ 5 % tolérance |
| Q5 | Arbitres | **Accepter** qu'ils forment des entités parasites en v1 | Essayer k=3 en Phase 1.5 pour les isoler (risqué) |
| Q6 | Gardiens | **Compter séparément** (max 1 par équipe) | Mélangés avec skaters |
| Q7 | Where to store the new file | **`runs/testNN/tracks_entities.json`** à côté des autres | sous-dossier `entities/` |

## 8. Plan de travail

| # | Livraison | Effort |
|---|---|---|
| 1 | Module d'extraction d'embeddings OSNet, batché sur les crops de phase6 existants | 0.5 j |
| 2 | Construction du graphe + greedy merge sous contraintes | 0.5 j |
| 3 | Intégration OCR + tests d'invariants (pas d'overlap, respect budget équipe) | 0.5 j |
| 4 | Sortie `tracks_entities.json` + stats + `phase6_annotate` v2 (color+#num par entité) | 0.5 j |
| 5 | Run sur test12 + spot-check manuel + itération | 0.5–1 j |

**Total : ~3 jours de travail concentré.** Un premier run partiel (étapes
1–3) est livrable en ~1 j si tu veux un signal intermédiaire avant de
continuer sur annotate.

## 9. Ce que Phase 1.6 *ne* résoudra pas

Pour que tu saches où s'arrête la promesse :
- **Arbitres mal isolés** (encore mélangés dans une équipe) — piste 5 de
  l'analyse initiale
- **Stats métriques** (distance, vitesse, heatmap) — ça arrive Phase 7,
  et ça a besoin de Phase 3 (homographie) qui reste bloquée
- **Fragmentation upstream de Phase 1** — on ne la résout pas, on la
  *corrige post-hoc*. Si à terme le tracker s'améliore (ex. fine-tune
  puck Phase 4), moins de travail pour Phase 1.6.
- **Rotations longue durée** (remplacements d'un skater à un autre dans
  la même équipe) — pour un clip 30s non-problématique ; pour un match
  complet il faudra relâcher la contrainte roster.
