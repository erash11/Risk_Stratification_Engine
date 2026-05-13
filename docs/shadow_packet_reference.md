# Shadow Packet Reference

A "packet" here is not a separate folder or PDF packet. It is a review unit
represented by one row in the CSVs.

In this project, each packet is basically:

`channel_name + test_season_id`

Example:

`broad_30d__2023-2024`

That packet means: review the `broad_30d` shadow alert behavior for the full
source-eligible `2023-2024` athlete-season set.

## Files To Use

`exposure_load_shadow_adjudication_template.csv`

This is the file you fill out.

`exposure_load_shadow_review_packets.csv`

This defines the 12 packets and their summary counts.

`exposure_load_shadow_replay_log.csv`

This is the supporting detail behind the packets.

`exposure_load_shadow_stop_rules.csv`

This shows why some channel-season rows were stopped or excluded.

## How To Review One Packet

1. Pick one row in the adjudication template, for example `broad_30d__2023-2024`.
2. Look at its summary fields: episode count, captured events, missed events, burden.
3. Cross-check the matching channel/season in the replay log.
4. Ask: "Would this alert behavior have been useful, trustworthy, and actionable at the time?"
5. Fill only the blank adjudication fields: `alert_usefulness`, `outcome_confirmed`, `source_context_ok`, `action_taken`, and `notes`.

Practically, you are reviewing 12 channel-season packets, not individual
athletes one by one and not a bundle of separate files. Each row is a packet;
the other replay files are the evidence context for that row.
