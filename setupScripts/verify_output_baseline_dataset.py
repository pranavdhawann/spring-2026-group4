import json
import sys
from pathlib import Path

output_dir = Path(__file__).parent.parent / "data" / "baseline_data"
files = list(output_dir.glob("*.jsonl"))

if not files:
    print("No output files found in", output_dir)
    sys.exit(1)

for f in files:
    samples = [json.loads(line) for line in open(f)]
    print(f"=== {f.name} ===")
    print(f"  Total samples: {len(samples)}")
    print(f"  History window: {len(samples[0]['dates'])} days")
    print(f"  Forecast horizon: {len(samples[0]['target'])} days")
    print(f"  Sector: {samples[0]['sector']}")
    print(f"  Date range: {samples[0]['dates'][0]} -> {samples[-1]['dates'][-1]}")

    # --- Across ALL samples ---
    samples_with_table = 0
    samples_with_news = 0
    total_table_filled = 0
    total_table_cells = 0
    total_news_days = 0
    total_days = 0

    for s in samples:
        # Table data
        td_keys = list(s['table_data'][0].keys())
        filled = sum(1 for day in s['table_data'] for v in day.values() if v is not None)
        total = len(s['table_data']) * len(td_keys)
        total_table_filled += filled
        total_table_cells += total
        if filled > 0:
            samples_with_table += 1

        # News
        news_days = sum(1 for day in s['articles'] if day is not None)
        total_news_days += news_days
        total_days += len(s['articles'])
        if news_days > 0:
            samples_with_news += 1

    print(f"\n  TABLE DATA (across all samples):")
    print(f"    Samples with any table data: {samples_with_table}/{len(samples)} ({100*samples_with_table/len(samples):.1f}%)")
    print(f"    Overall cell fill rate: {total_table_filled}/{total_table_cells} ({100*total_table_filled/total_table_cells:.1f}%)")

    print(f"\n  NEWS (across all samples):")
    print(f"    Samples with any news: {samples_with_news}/{len(samples)} ({100*samples_with_news/len(samples):.1f}%)")
    print(f"    Total day-articles: {total_news_days}/{total_days} ({100*total_news_days/total_days:.1f}%)")

    # Time series spot check
    ts_nulls = sum(1 for s in samples for day in s['time_series'] for v in day.values() if v is None)
    ts_total = sum(len(s['time_series']) * len(s['time_series'][0]) for s in samples)
    print(f"\n  TIME SERIES:")
    print(f"    Null rate: {ts_nulls}/{ts_total} ({100*ts_nulls/ts_total:.1f}%)")

    # Target check
    null_targets = sum(1 for s in samples for t in s['target'] if t is None)
    total_targets = sum(len(s['target']) for s in samples)
    print(f"\n  TARGETS:")
    print(f"    Present: {total_targets - null_targets}/{total_targets} ({100*(total_targets-null_targets)/total_targets:.1f}%)")

    # Show a mid-range sample for sanity
    mid = len(samples) // 2
    ms = samples[mid]
    mid_filled = sum(1 for day in ms['table_data'] for v in day.values() if v is not None)
    mid_total = len(ms['table_data']) * len(ms['table_data'][0])
    mid_news = sum(1 for day in ms['articles'] if day is not None)
    print(f"\n  SPOT CHECK (sample {mid}, dates {ms['dates'][0]} -> {ms['dates'][-1]}):")
    print(f"    Table fill: {mid_filled}/{mid_total} ({100*mid_filled/mid_total:.1f}%)")
    print(f"    News days: {mid_news}/{len(ms['articles'])}")
    print(f"    Target: {ms['target']}")
    print()
