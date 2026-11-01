# T04 — pilote complet, absence de gain général qualifié

31 octobre 2026. **REJETÉ POUR INTÉGRATION ; code T04 retiré.** La qualité
numérique du candidat est qualifiée dans le domaine exercé ; sa performance
reste non concluante pour un gain général. Le critère « accélérer sans
ralentir le décode » n'est pas satisfait : AR est négatif, les petits signaux
MTP/DFlash restent mixtes et sensibles à la dérive. Aucun gain global ni
régression universelle causale n'est établi. Les vérifications finales du
code restauré sont **passées** ; aucune app
installée, aucun push ou publication n'est modifié.

## Candidat, protocole et preuves

Le port reprend T04 de [PR61](https://github.com/0xZKnw/mlxl3/pull/61),
tête vérifiée `a36ba5077bf386466660d6b478058fbe6880245c`, correction dtype
`40f45f8016ef68881bad9b6344fc6022b28fc038`, sans merge ni import de T17/T18.
Il étend la closure SwiGLU par le cast activation F16 et le produit suh F16,
garde H128 MLX et down EXL3 stock. Domaine : slots CPU non masqués/non groupés ;
Hot masqué/groupé inchangé. [Audit de source](audit-performance-source.md).

Même CLI dans tous les bras :
`81bc3a14d509224d77fd6d98d8dc64f9b840c0094f7376f5f004e924849732cd`.
Libtest qualifié :
`22073b88203fa4db0a3e2792f03f99c0e52f1a131c27f2f6d37c535ac10b384d`.
[Identité figée r3](final-identity-r3.json) ; sources locales modifiées
préexistantes identifiées, poids non intégralement hachés.

Apple M5/24 Gio, macOS27.2, MLX0.32.2 ; Qwen3.8 Flash-Next EXL3 2.05bpw
`65c89531`, draft DFlash2 `a22a1744`. Cache **3 Gio /53slots×48couches**,
4lecteurs, allocateur64Mio, MLX40ops/40MB, Hot/Wide/Unique ON, lookahead0,
greedy/contexte4096. T04 seul varie : A=0, C=1 ; T05/T17 OFF, T18 absent.

Pilote : **12processus, ordre ACCA par mode AR/MTP/DF**, chacun chauffe
MAIN puis HOLD et mesure MAIN puis HOLD une fois. Total **24chauffes +
24mesures =48réponses128**,127transitions de décode chacune, prompts87/82.
Réutilisation de préfixe OFF. Les débits ci-dessous sont calculés par
**somme des tokens ou transitions / somme des durées**, chauffes exclues,
sans addition des gains entre modes. [Inspection complète](pilot128-inspection.json),
[plans et rapports par bras](prepared/manifest.json).

## Résultats mesurés

Débits en tok/s ; Δ=(ON/OFF−1). Les arrondis du tableau ne remplacent pas
les valeurs complètes conservées dans l'inspection JSON.

| Mode | Prompt | Décode OFF | Décode ON | Δdécode | Préfill OFF | Préfill ON | Δpréfill |
|---|---|---:|---:|---:|---:|---:|---:|
| AR | MAIN | 10,6175 | 10,3925 | −2,12% | 10,2433 | 10,2378 | −0,05% |
| AR | HOLD | 10,3238 | 10,2028 | −1,17% | 10,3577 | 10,2720 | −0,83% |
| MTP | MAIN | 11,0429 | 10,9581 | −0,77% | 9,8162 | 9,6254 | −1,94% |
| MTP | HOLD | 10,0880 | 10,2448 | +1,55% | 9,7867 | 9,8425 | +0,57% |
| DFlash | MAIN | 11,4973 | 11,5669 | +0,61% | 10,0171 | 10,0585 | +0,41% |
| DFlash | HOLD | 10,8940 | 11,0317 | +1,26% | 10,0237 | 10,0171 | −0,07% |

| Mode, MAIN+HOLD | Décode OFF | Décode ON | Δdécode | Δpréfill |
|---|---:|---:|---:|---:|
| AR | 10,468587 | 10,296765 | **−1,641318%** | −0,428787% |
| MTP | 10,543890 | 10,589444 | +0,432035% | −0,738049% |
| DFlash | 11,187542 | 11,292933 | +0,942043% | +0,179787% |

Dérive du premier au dernier témoin OFF : AR MAIN−3,20%/HOLD−1,88%,
MTP MAIN+2,76%/HOLD+4,38%, DFlash MAIN+1,21%/HOLD−3,55%.
Elle limite l'attribution causale des faibles écarts. Aucun processus ou
prompt n'a été écarté ; les trois modes restent des comparaisons séparées.

## Qualité, mémoire et portée

- **48/48réponses du pilote qualifiées externes** : texte/hash128 complets
  indépendants, comptes/modes et compteurs homologues A/C identiques par
  mode/prompt/groupe. Processus et lecteurs joints ; aucun moteur actif
  de cette campagne après le dernier bras.
- [Bridge17](bridge17-inspection.json) :6processus AC,12chauffes+12mesures,
 24réponses17,16transitions, mêmes contrôles externes et compteurs A/C.
  Ce contrôle court ne remplace pas le pilote128.
- [Qualité physique](model-quality-inspection.json) :10sélections uniques
  réussies —4composants dont capture,1MoE réel isolé,5oracles modèle.
  Logits indépendants5×248320 et états, composants, paires/commits,
  MTPbudgets1/2/3/17, DFlash36préfixes B1..8. Le domaine numérique testé
  est fini ; la couverture63488gatesF16 ne constitue pas le produit
  cartésien exhaustif de toutes les activations/scales/F32.
- Capture réelle GU20 : **3→2dispatchs**, même H128 ;521317700octets
  cumulés sous1Gio. Aucun temps GPU gagné déduit de leur nombre.
  [Inspection Xcode](xcode-dispatch-inspection.json).
- **0nouveau Swapout depuis T04-w1**, ancre140865945600octets inchangée,
  secteur et séparation des mesures/compilations/proveurs. Picnatif rapporté
  du pilote **7902009944octets**, budgets RSS/footprint qualifiés aux
  observations ; cela ne décrit pas une RAM totale OS ni un pic continu universel.

L'admission initiale12Go a été refusée avant lancement ; ses plans et
identités sont [archivés](admission12-original/). L'amendement prospectif
[ADMISSION11-01](admission11-revision.json), avant tout modèle complet ou
chrono T04, a fixé **11Go avant chaque processus**,60s stables et toutes
les gardes swap/RSS/footprint inchangées. Aucun reset d'ancre, aucun mélange
avec des résultats T17 ou un protocole12Go. L'estimation libre+fichiers reste
heuristique, comme le rappelle l'incident T17 conservé.

[Vérifications du candidat avant retrait](verification.md) : CPU258passés/8ignorés,
MLX295passés/189ignorés,0échec, fmt/Clippy strict/buildCLI passés. Kani sur
la vraie garde CPU :166obligations/12couvertures, unwind17, rangs0..5 et
dimensions symboliques ; ce contrôle borné ne prouve pas MLX/Metal. Tentative
Kani voie MLX interrompue par ICE avant obligations ; CBMC FFI bloqué au
parsing libc++, CrossHair absent. CI distante de l'arbre local non publiée
**non exécutée** ; tests ignorés non assimilés à des passes.

L'audit avant décision n'a trouvé aucun défaut concret de portage,
fallback, synchronisation ou donation. Troisième input, métadonnées et
reshape supplémentaire sont identifiés, sans coût causal isolé. L'audit
final des48réponses n'a révélé aucun nouveau défaut concret ; le retrait
sélectif préenregistré a donc été appliqué.

[Retrait confirmé](retirement-result.json) : **8fichiers restaurés**, seul
le nouveau module test T04 supprimé après copie vérifiée ; **190/190empreintes
du périmètre pré-T04 identiques**, sans réinitialisation des autres locales.
T17/T05 sont préservés dans leur état antérieur. Les **9sources exactes du
candidat** sont [archivées](candidate-source/), avec
[patch complet](candidate-complete-local.patch),
[identité source](candidate-source-identity.json), binaires81bc/220 et toutes
les preuves conservées. Les sauvegardes de restauration reconstituées
après édition restent explicitement identifiées et correspondent aux hashes
pré-T04 figés.

Les contrôles du **code conservé** sont terminés : fmt/Clippy strict CPU et
MLX/build CLI passent ; **CPU256/0échec/8ignorés**, **MLX293/0échec/185ignorés**,
190 sources identiques au pré-T04. [Résultat final](retained-checks-inspection.json)
et [plan distinct](retained-checks-plan-result.json). CLI normal
`202f96dca5ceed6cd78f70c524ad48b324e2c35a347cd7dc0bcabb11baf141aa`
et libtest `a751cf592d9e28622d1c062a9f93213f4c55d8a6dcce0fa8b1c035be16c6b862`
sont byte-identiques à leurs versions pré-T04. Aucun GPU ni benchmark
supplémentaire après retrait ; aucun run CI distant de l'arbre non publié.
Les résultats de qualité et de débit ci-dessus concernent uniquement le
candidat archivé `81bc…/220…` et ses bras OFF/ON ; ils ne constituent pas
une nouvelle mesure du moteur restauré ni une mise à jour de l'application.
