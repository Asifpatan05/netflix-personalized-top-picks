# Stage 3 Prompt Note

## Claude Code prompt (Stage 3)

> **TODO: paste the full prompt verbatim.** It begins "Stage 3: Implement Qdrant and the Netflix Top Picks API". Only that opening line was available when this draft was made; the rest of the prompt must be copied from the original.

## What changed

- A known member is anyone who appears in `cleaned_viewing_history.csv`.
- Exclusion rule: any title a member has played at all is excluded from their recommendations (any-play exclusion).
- Members with no completed titles stay on the 200 path (they get a 200 response, not a 404).

## Checks

Status of each check. Tick a box only after the command has been run and its output seen.

- [ ] `GET /recommendations/M-10001` returns HTTP 200
- [ ] `GET /recommendations/M-99999` returns HTTP 404 (`samples/top_picks_404_M-99999.json`)
- [ ] `TT-0300` appears in `samples/top_picks_200_genre_matched.json`
- [ ] The member in that sample has a genre match with TT-0300 in the dataset
- [ ] All samples validate against the API response contract
