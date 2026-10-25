# Injection HC exacte sur Flash Next — essai18 rejeté

Le [profil16](flash-profile-2026-10-12.md) motive une réduction du nombre de
soumissions GPU autour des connexions HC. `Hc::write` calcule le produit de
la branche par sa porte puis l'ajoute au flux résiduel. Le candidat fusionne
ces deux opérations en un kernel pour les flux/portes FP32, sans FMA : le
produit FP32 arrondi reste distinct de l'addition. Les injections FP16 ou
mixtes de la tête MTP gardent le graphe historique.

Aucun changement du cache d'experts, des lecteurs SSD, de la sélection
d'experts, des états, de la quantification ni des projections EXL3. Le
prototype de regroupement MTP15/15B reste retiré. L'overlap hot/cold reste
actif après le rejet17.

## Correction avant vitesse

La première version `7dc3cd14` passe les références HC mais refuse les flux
FP16 de la tête MTP :10 tests modèle réussissent et2 échouent. Ce résultat
est conservé dans [quality.json](measurements/flash-next-ssd-impl/opt-18-hc/quality.json)
et [qwen.log](measurements/flash-next-ssd-impl/opt-18-hc/qwen.log), avec
[source initiale](measurements/flash-next-ssd-impl/opt-18-hc/prototype-v1.patch).
Aucun débit n'est mesuré sur cette version.

La correction garde les formes vérifiées avant dispatch et réutilise le
graphe d'origine si le flux ou la porte est FP16. Deux régressions physiques
comparent les résultats au CPU et au graphe MLX :40 960 valeurs FP32 avec
branche FP16/FP32, strides, zéros, signes et cancellation ;51 200 valeurs
pour cinq combinaisons de types du chemin FP16/mixte. Les24 refus de
métadonnées invalides et6 fixtures HC indépendantes complètent ces tests.

Une mutation du shader utilisant une FMA est détectée, puis le shader est
restauré et le test repasse, avec binaire de test identique. Une seconde
mutation réintroduisant la garde FP32 fautive est détectée par la nouvelle
régression FP16, puis retirée. Preuves :
[mutation FMA](measurements/flash-next-ssd-impl/opt-18-hc/mutation.json),
[mutation de garde et reconstruction](measurements/flash-next-ssd-impl/opt-18-hc/repair/checks.json).

La source finale et les binaires sont identifiés dans
[identity.json](measurements/flash-next-ssd-impl/opt-18-hc/repair/identity.json) :
moteur `51e05452`, tests `2db4db15`. `cargo fmt --all --check`, Clippy strict
avec `mlx,chat`, build release livré,87 tests CPU/2 ignorés et107 tests
release/98 ignorés passent. Les tests physiques sont lancés explicitement,
séparément de ces suites.

Kani0.68.0 passe10 contrats CPU inchangés :1 118 obligations réussies,
3 inatteignables historiques et33 couvertures satisfaites. Le harnais du
chemin Rust→MLX→Metal utilise trois i8 symboliques indépendants, sans assume,
convertis en flottants et diffusés aux dimensions réelles ; le compilateur
Kani échoue avant les obligations (`intrinsics.rs:243`). CBMC6.11 ne reconnaît
pas le MSL. **Le kernel GPU n'est pas prouvé formellement** ; aucune
traduction ni fonction factice ne remplace ce parcours.
[Preuves et limites](measurements/flash-next-ssd-impl/opt-18-hc/proof-inspection.json).

La version corrigée passe13 tests Qwen physiques, l'oracle tenu à part de11
positions et3 tests SSD. Sur les16 positions indépendantes, logits complets
et états cible restent exacts ; tête MTP, cache-only, acceptation/rejet et
sessions passent aussi. Le bridge réel1/2/3/17 termine en256,81 s avec
configuration, cinq refus attendus et deux annulations/reprises.
[Exécution physique finale](measurements/flash-next-ssd-impl/opt-18-hc/repair/quality.json).

Le microbenchmark conserve les cinq paires, dont une défavorable :réductions
22,92 %,−8,76 %,9,30 %,15,29 %,17,65 %. Médianes170,78→148,12 µs/appel.
Ce signal variable ne valide aucun gain modèle. Pas de relevé VM propre au
micro ; aucun picRAM ou swapout ne lui est attribué.
[Données micro](measurements/flash-next-ssd-impl/opt-18-hc/repair/micro-summary.json).

## Protocole de vitesse préenregistré

Même M5/24 Gio, MLX0.32.2, checkpoint65c89531/EXL3 2,05 bpw que16/17 ;
cache5 Gio/89 slots, allocator512 Mio, quatre lecteurs, overlap1, contexte4096,
greedy, applications utilisateur conservées. Baseline immuable `bd89a7d6`.
Un modèle à la fois, aucun compilateur/proveur pendant les chronométrages.

Après oracles complets, états/transactions et bridge1/2/3/17 avec
configuration/annulation/reprise : microbenchmark de256 injections évaluées,
16 chauffes, cinq paires alternées. Ce diagnostic ne mesure pas le débit du
modèle. Ensuite MAIN/HOLD ×OFF/MTP, OFF d'abord ; chaque bras charge un seul
prompt/mode, chauffe64 tokens puis en mesure64 (63 décodés).

Le premier tour H-C doit gagner au moins1 % dans chaque cellule pour
lancer C-H puis H-C. Rétention :chaque paire≥1 %, étendue des trois gains≤3
points par cellule, pic candidat≤témoin+256 Mio, swapouts système≤256 Mio par
bras, réponses/compteurs/identités exacts. Conserver tous les bras et échecs ;
aucune somme avec les gains des lecteurs14B ou d'autres campagnes.

## Résultat du crible réel

| Cas | Témoin, tok/s | HC fusionné, tok/s | Écart |
|---|---:|---:|---:|
| Français OFF | 7,298 | 6,721 | −7,90 % |
| Code OFF | 7,252 | 7,150 | −1,40 % |
| Français MTP | 7,222 | 7,162 | −0,84 % |
| Code MTP | 7,145 | 7,176 | +0,43 % |

**Rejeté.** Le premier tour échoue au critère préenregistré ; huit bras
terminés et seize bras de confirmation non lancés. Les16 réponses gardent
leurs textes/hashes exacts ; aucun bras n'ajoute de swapout système. Pics
maximaux des deux variantes :8,684 Gio OFF et9,583 Gio MTP. Le plus grand
supplément mémoire candidat est inférieur à1 Mio. Aucune accélération HC
n'est retenue, aucun résultat du micro n'est présenté comme un gain modèle.

[Matrice et identités](measurements/flash-next-ssd-impl/opt-18-hc/speed/matrix.json),
[conditions et exécutions](measurements/flash-next-ssd-impl/opt-18-hc/speed/execution.json),
[résultat](measurements/flash-next-ssd-impl/opt-18-hc/speed/result.json).
[Prototype corrigé archivé](measurements/flash-next-ssd-impl/opt-18-hc/prototype-final.patch),
[shader archivé](measurements/flash-next-ssd-impl/opt-18-hc/prototype-final.metal).
La source rejetée a servi au [diagnostic19](flash-kernel-cache-2026-10-12.md),
puis a été retirée. Le moteur local est restauré et qualifié par identité
des octets exécutables, hors métadonnées Git/signature. Aucune installation
ni publication d'application.
