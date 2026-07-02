Day 7 conclusion:
Manual-only accuracy: 0.95
TF-IDF-only accuracy: 0.80
Combined accuracy: 1.00

Manual features captured operational context such as CPU, memory, service, level and timeout flags.
TF-IDF captured textual meaning such as payout, shard lag, reads and static assets.
The combined model performed best because it used both operational signals and text signals.

Main lesson:
In DevOps log triage, structured operational features and text features are complementary, not competing approaches.
