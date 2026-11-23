# Netflix Personalized Top Picks

## How to run

Install the required Python packages:

```bash
pip install pandas numpy
```

Run the cleaning script from the repository root, replacing the input filenames with the actual CSV filenames:

```bash
python cleaning.py --events "viewing_events.csv" --catalog "title_catalog.csv" --out "stage1_output"
```

The script generates:
- `cleaned_viewing_history.csv`
- `train_events.csv`
- `holdout_events.csv`

It also validates the expected split of 4,622 training events and 378 holdout events.
